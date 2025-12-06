import logging
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from uncertainty.uncertainty_measures.memory_sematic_entropy import (
    DialogueMemory,
    memory_conditioned_semantic_entropy,
)
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.utils.eval_utils import f1_score

logging.basicConfig(level=logging.INFO)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two distributions p and q.
    Both should be 1D numpy arrays that sum to 1.
    """
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def load_generations(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(
    xlsx_path: str = "synthetic_multiturn_squad.xlsx",
    generations_path: str = "validation_generations.pkl",
    out_pkl: str = "mcse_eval.pkl",
):
    # 1) Load synthetic multi-turn dataset
    df = pd.read_excel(xlsx_path)

    # Construct ids matching generate_answers.py
    df["id"] = df.apply(
    lambda row: f"validation_{row['dialogue_id']}_d1_t{int(row['turn_id'])}",
    axis=1,
)
    logging.info("Loaded %d turns from synthetic multiturn SQuAD.", len(df))

    # 2) Load generations
    gens = load_generations(generations_path)
    logging.info("Loaded %d generations from %s", len(gens), generations_path)

    # 3) NLI model used inside MC-SE (semantic clustering + memory check)
    entail_model = EntailmentDeberta()

    results = {}  # ex_id -> per-example dict
    total = 0
    correct = 0

    # For drift per "slot" (base_qid), we remember the previous cluster distribution
    prev_cluster_dist_per_slot = {}  # base_qid -> np.array of probs

    # 4) Process dialogue by dialogue
    for dlg_id, group in tqdm(df.groupby("dialogue_id"), desc="Dialogues"):
        group = group.sort_values("turn_id")

        memory = DialogueMemory()
        slot_answers = {}  # base_qid -> earliest model answer

        for _, row in group.iterrows():
            ex_id = row["id"]
            turn_id = int(row["turn_id"])
            is_repeat = int(row["is_repeat"])
            base_qid = str(row["base_qid"])
            gold_answer = str(row["answer"])
            question = str(row["question"])

            if ex_id not in gens:
                logging.warning("Example id %s not in generations; skipping.", ex_id)
                continue

            gen_entry = gens[ex_id]
            most_likely = gen_entry["most_likely_answer"]
            model_answer = str(most_likely["response"])
            responses = gen_entry.get("responses", [])

            # --- correctness (same as baseline) ---
            f1_fact = f1_score(model_answer, gold_answer)
            factually_correct = int(f1_fact >= 0.8)
            total += 1
            correct += factually_correct

            # --- self-consistency on re-asks ---
            if is_repeat == 1 and base_qid in slot_answers:
                prev_answer = slot_answers[base_qid]
                f1_self = f1_score(model_answer, prev_answer)
                self_consistent = int(f1_self >= 0.8)
            else:
                prev_answer = None
                f1_self = None
                self_consistent = None

            # --- MC-SE computation ---
            mc = memory_conditioned_semantic_entropy(
                responses=responses,
                question_text=question,
                memory=memory,
                entail_model=entail_model,
                strict_entailment=False,
            )

            H_MC = mc["H_MC"]
            cluster_probs_mc = np.array(mc["cluster_probs_mc"]) if mc["cluster_probs_mc"] else np.array([1.0])

            # Contradiction score: fraction of samples that got down-weighted (w < 1)
            sample_weights = np.array(mc["sample_weights"])
            if len(sample_weights) > 0:
                contradiction_rate = float(np.mean(sample_weights < 0.99))
            else:
                contradiction_rate = 0.0

            # Drift: JSD to previous distribution for the same base_qid (if any)
            if base_qid in prev_cluster_dist_per_slot and len(cluster_probs_mc) == len(
                prev_cluster_dist_per_slot[base_qid]
            ):
                drift = float(jsd(cluster_probs_mc, prev_cluster_dist_per_slot[base_qid]))
            else:
                drift = 0.0

            prev_cluster_dist_per_slot[base_qid] = cluster_probs_mc

            # Final risk score (simple linear combination)
            alpha, beta, gamma = 1.0, 1.0, 1.0
            risk = alpha * H_MC + beta * contradiction_rate + gamma * drift

            # --- update memory AFTER computing MC-SE ---
            memory.add_fact(model_answer)
            if base_qid not in slot_answers:
                slot_answers[base_qid] = model_answer

            results[ex_id] = {
                "dialogue_id": dlg_id,
                "turn_id": turn_id,
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
                "is_repeat": is_repeat,
                "base_qid": base_qid,
                "f1_fact": f1_fact,
                "factually_correct": factually_correct,
                "f1_self_consistency": f1_self,
                "self_consistent": self_consistent,
                "H_MC": H_MC,
                "contradiction_rate": contradiction_rate,
                "drift": drift,
                "risk": risk,
            }

    # --- Print global summary like baseline ---
    acc = correct / total if total > 0 else 0.0
    print(f"Semantic correctness accuracy (MC-SE branch): {acc:.2f}")
    print(f"Total examples: {total}, Correct: {correct}")
    print(f"Saving per-example results to: {out_pkl}")

    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    logging.info("Saved MC-SE evaluation to %s", out_pkl)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx_path",
        type=str,
        default="synthetic_multiturn_squad.xlsx",
        help="Path to synthetic multiturn SQuAD Excel.",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default="validation_generations.pkl",
        help="Path to validation_generations.pkl from generate_answers.py.",
    )
    parser.add_argument(
        "--out_pkl",
        type=str,
        default="mcse_eval.pkl",
        help="Where to save the MC-SE per-example evaluation.",
    )

    args = parser.parse_args()
    main(
        xlsx_path=args.xlsx_path,
        generations_path=args.generations_path,
        out_pkl=args.out_pkl,
    )