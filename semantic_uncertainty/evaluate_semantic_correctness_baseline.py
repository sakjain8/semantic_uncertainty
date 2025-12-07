import argparse
import pickle
import string
import re
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn import metrics

# <-- NEW: import SE + AUC helpers from your repo -->
from uncertainty.uncertainty_measures.semantic_entropy import (
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy_rao,
)
from uncertainty.utils.eval_utils import area_under_thresholded_accuracy

# If your EntailmentDeberta is elsewhere, keep your original import:
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

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
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
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)


def embedding_similarity(pred: str, gold: str) -> float:
    emb_pred = embedder.encode(pred, convert_to_tensor=True)
    emb_gold = embedder.encode(gold, convert_to_tensor=True)
    sim = util.cos_sim(emb_pred, emb_gold).item()
    return float(sim)


##############################################
# 3. NLI-based correctness (DeBERTa MNLI, GPU)
##############################################

print(f"Loading DeBERTa MNLI entailment model on {DEVICE}...")
nli_model = EntailmentDeberta(device=DEVICE)  # keep your original init


# Map: 2 = entailment, 1 = neutral, 0 = contradiction (by your code)
def nli_label(premise: str, hypothesis: str) -> int:
    return nli_model.check_implication(premise, hypothesis)


##############################################
# 4. Final correctness rule (same as before)
##############################################

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
    label = nli_label(gold, pred)  # 2: entailment, 1: neutral, 0: contradiction
    if label == 2:  # entailment
        return True, {"f1": f1, "sim": sim, "nli": label}

    # Otherwise incorrect
    return False, {"f1": f1, "sim": sim, "nli": label}


##############################################
# 5. Semantic Entropy H_SE from generations
##############################################

def compute_h_se_for_example(
    responses: List[tuple],
    question: str,
) -> float:
    """
    Compute semantic entropy H_SE for one QA example,
    using the original clustering + logsumexp logic.

    responses: list of (predicted_answer, token_log_likelihoods, embedding, accuracy)
               from generate_answers.py
    """
    if not responses:
        return 0.0

    # 1) per-sample log probabilities: sum of token log-likelihoods
    texts = []
    log_probs = []
    for (ans, token_lls, emb, acc) in responses:
        texts.append(str(ans))
        if token_lls is None or len(token_lls) == 0:
            log_probs.append(0.0)
        else:
            log_probs.append(float(np.sum(token_lls)))

    log_probs = np.array(log_probs)

    # 2) cluster by semantic meaning using DeBERTa NLI
    semantic_ids = get_semantic_ids(
        strings_list=texts,
        model=nli_model,  # reuse the same DeBERTa NLI model
        strict_entailment=False,
        example={"question": question},
    )

    # 3) aggregate to cluster space with normalized log-sum-exp
    log_cluster_probs = logsumexp_by_id(
        semantic_ids=semantic_ids,
        log_likelihoods=log_probs,
        agg="sum_normalized",
    )
    log_cluster_probs = np.array(log_cluster_probs)

    # 4) H_SE = -Σ p log p using predictive_entropy_rao
    H_SE = predictive_entropy_rao(log_cluster_probs)
    return float(H_SE)


##############################################
# 6. Main evaluation: baseline SE + metrics
##############################################

def main(args: argparse.Namespace) -> None:
    print(f"Loading generations from: {args.generations_path}")
    with open(args.generations_path, "rb") as f:
        generations = pickle.load(f)

    total = 0
    correct = 0
    f1_list = []
    hse_list = []
    error_labels = []  # 1 if error, 0 if correct

    per_example = []  # for saving detailed results

    for ex_id, entry in generations.items():
        # Most likely answer (low temperature generation)
        pred = entry["most_likely_answer"]["response"]

        # Gold answer from reference
        ref = entry["reference"]
        gold_answers = ref["answers"]["text"]
        if isinstance(gold_answers, list) and len(gold_answers) > 0:
            gold = gold_answers[0]
        else:
            gold = ""

        # Multi-turn question text (still one QA per example)
        question = entry.get("question", "")

        total += 1

        # 6a) semantic correctness by your 3-layer rule
        is_corr, details = is_semantically_correct(pred, gold)
        correct_flag = int(is_corr)
        correct += correct_flag
        f1_val = details["f1"]
        f1_list.append(f1_val)
        error_labels.append(1 - correct_flag)

        # 6b) semantic entropy H_SE from high-T responses
        responses = entry.get("responses", [])
        H_SE = compute_h_se_for_example(responses, question)
        hse_list.append(H_SE)

        per_example.append({
            "id": ex_id,
            "question": question,
            "pred": pred,
            "gold": gold,
            "correct": correct_flag,
            "f1": f1_val,
            "H_SE": H_SE,
            "emb_sim": details["sim"],
            "nli_label": details["nli"],
        })

    # -------------------
    # Aggregate metrics
    # -------------------
    acc = correct / total if total > 0 else 0.0
    mean_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    print(f"\nSemantic correctness accuracy (baseline SE branch): {acc:.4f}")
    print(f"Total examples: {total}, Correct: {correct}")
    print(f"Average token F1 (prediction vs gold): {mean_f1:.4f}")

    hse_arr = np.array(hse_list, dtype=float)
    err_arr = np.array(error_labels, dtype=int)

    # AUROC(H_SE → error)
    try:
        auroc = metrics.roc_auc_score(err_arr, hse_arr)
    except Exception:
        auroc = float("nan")
    print(f"AUROC (H_SE → error): {auroc:.4f}")

    # Area under thresholded accuracy (same metric as original repo)
    # accuracies = 1 - error_labels
    accuracies = 1 - err_arr
    auta = area_under_thresholded_accuracy(accuracies, hse_arr)
    print(f"Area under thresholded accuracy: {auta:.4f}")

    # -------------------
    # Save everything
    # -------------------
    if args.out_path is not None:
        print(f"Saving per-example results to: {args.out_path}")
        with open(args.out_path, "wb") as f:
            pickle.dump(
                {
                    "accuracy": acc,
                    "mean_f1": mean_f1,
                    "auroc_hse_error": auroc,
                    "auta_hse": auta,
                    "results": per_example,
                    "thresholds": {
                        "TH_F1": TH_F1,
                        "TH_EMB": TH_EMB,
                    },
                },
                f,
            )


##############################################
# 7. CLI entrypoint
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