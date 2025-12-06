import argparse
import pickle
import string
import re
from typing import Tuple, Dict, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from uncertainty.entailment import EntailmentDeberta


##############################################
# 0. Device
##############################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


##############################################
# 1. Text normalization and token F1
##############################################

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    # remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # remove extra spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


##############################################
# 2. Embedding-based similarity (GPU)
##############################################

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print(f"Loading sentence embedding model on {DEVICE}: {EMBED_MODEL_NAME}")
# sentence-transformers lets you set the device at init
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)


def embedding_similarity(pred: str, gold: str) -> float:
    # encode returns tensors on the configured device
    emb_pred = embedder.encode(pred, convert_to_tensor=True)
    emb_gold = embedder.encode(gold, convert_to_tensor=True)
    sim = util.cos_sim(emb_pred, emb_gold).item()
    return float(sim)


##############################################
# 3. NLI-based correctness (DeBERTa MNLI, GPU)
##############################################

print(f"Loading DeBERTa MNLI entailment model on {DEVICE}...")
# EntailmentDeberta must accept a `device` argument and move model to that device
nli_model = EntailmentDeberta(device=DEVICE)


# Map: 2 = entailment, 1 = neutral, 0 = contradiction (by their code)
def nli_label(premise: str, hypothesis: str) -> int:
    return nli_model.check_implication(premise, hypothesis)


##############################################
# 4. Final correctness rule
##############################################

# Thresholds – you can tune these later
TH_F1 = 0.8
TH_EMB = 0.85


def is_semantically_correct(pred: str, gold: str) -> Tuple[bool, Dict[str, Any]]:
    """Return (correct_flag, details_dict)."""

    f1 = token_f1(pred, gold)
    sim = embedding_similarity(pred, gold)

    # Layer 1: high lexical overlap
    if f1 >= TH_F1:
        return True, {"f1": f1, "sim": sim, "nli": None}

    # Layer 2: high embedding similarity
    if sim >= TH_EMB:
        return True, {"f1": f1, "sim": sim, "nli": None}

    # Layer 3: NLI entailment (gold -> pred)
    # You can also try symmetric or pred->gold; we pick gold->pred here.
    label = nli_label(gold, pred)  # 2: entailment, 1: neutral, 0: contradiction
    if label == 2:  # entailment
        return True, {"f1": f1, "sim": sim, "nli": label}

    # Otherwise incorrect
    return False, {"f1": f1, "sim": sim, "nli": label}


##############################################
# 5. Main evaluation: baseline SE branch
##############################################

def main(args: argparse.Namespace) -> None:
    print(f"Loading generations from: {args.generations_path}")
    with open(args.generations_path, "rb") as f:
        generations = pickle.load(f)

    total = 0
    correct = 0

    per_example = []  # for saving detailed results

    for ex_id, entry in generations.items():
        # Most likely answer (low temperature generation)
        pred = entry["most_likely_answer"]["response"]

        # Gold answer from reference
        # In their utils.get_reference, they store:
        # 'reference': {'answers': {'text': [...], 'answer_start': [...]}, 'id': ...}
        ref = entry["reference"]
        gold_answers = ref["answers"]["text"]
        if isinstance(gold_answers, list) and len(gold_answers) > 0:
            gold = gold_answers[0]
        else:
            gold = ""

        total += 1
        is_corr, details = is_semantically_correct(pred, gold)
        if is_corr:
            correct += 1

        per_example.append({
            "id": ex_id,
            "question": entry["question"],
            "pred": pred,
            "gold": gold,
            "correct": int(is_corr),
            "f1": details["f1"],
            "emb_sim": details["sim"],
            "nli_label": details["nli"],
        })

    acc = correct / total if total > 0 else 0.0
    print(f"\nSemantic correctness accuracy (baseline SE branch): {acc:.4f}")
    print(f"Total examples: {total}, Correct: {correct}")

    # Optionally save detailed per-example metrics
    if args.out_path is not None:
        print(f"Saving per-example results to: {args.out_path}")
        with open(args.out_path, "wb") as f:
            pickle.dump({
                "accuracy": acc,
                "results": per_example,
                "thresholds": {
                    "TH_F1": TH_F1,
                    "TH_EMB": TH_EMB
                }
            }, f)


##############################################
# 6. CLI entrypoint
##############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations_path",
        type=str,
        default="validation_generations.pkl",
        help="Path to *_generations.pkl from generate_answers.py (validation split).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="baseline_semantic_eval.pkl",
        help="Where to save detailed evaluation results (pickle).",
    )
    args = parser.parse_args()
    main(args)
