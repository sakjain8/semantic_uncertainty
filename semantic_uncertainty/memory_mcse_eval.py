import argparse
import logging
import pickle
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from uncertainty.uncertainty_measures.memory_sematic_entropy import (
    memory_conditioned_semantic_entropy,
)
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.utils.eval_utils import (
    auroc,
    area_under_thresholded_accuracy,
)
from evaluate_semantic_correctness_baseline import is_semantically_correct

logging.basicConfig(level=logging.INFO)


def load_generations(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main(
    xlsx_path: str,
    generations_path: str,
    out_path: str,
):
    logging.info("Loading dataset: %s", xlsx_path)
    df = pd.read_excel(xlsx_path)

    # Ensure ID matches generation keys
    if "id" not in df.columns:
        df["id"] = df.apply(
            lambda r: f"{r['dialogue_id']}_t{int(r['turn_id'])}", axis=1
        )

    logging.info("Loading generations: %s", generations_path)
    gens = load_generations(generations_path)

    entail_model = EntailmentDeberta()

    records: List[Dict[str, Any]] = []
    h_mc_list = []
    is_false_list = []

    total = 0
    correct = 0

    # Process dialogue-wise
    for dlg_id, group in tqdm(df.groupby("dialogue_id"), desc="Dialogues"):
        group = group.sort_values("turn_id")

        # Memory indexed by base_qid
        memory_by_slot: Dict[str, List[str]] = {}

        for _, row in group.iterrows():
            # 🔴 MC-SE ONLY applies to repeat questions
            if row["turn_id"] == 1:
                continue

            ex_id = row["id"]
            base_qid = str(row["base_qid"])
            question = str(row["question"])
            gold_answer = str(row["answer"])

            if ex_id not in gens:
                continue

            gen = gens[ex_id]
            model_answer = gen["most_likely_answer"]["response"]
            responses = gen.get("responses", [])

            # --- correctness (IDENTICAL to baseline) ---
            is_corr, details = is_semantically_correct(
                model_answer, gold_answer
            )

            total += 1
            if is_corr:
                correct += 1
                is_false = 0
            else:
                is_false = 1

            # --- retrieve slot-specific memory ---
            memory_facts = memory_by_slot.get(base_qid, [])

            # --- MC semantic entropy ---
            mc = memory_conditioned_semantic_entropy(
                responses=responses,
                question_text=question,
                memory_facts=memory_facts,
                entail_model=entail_model,
                strict_entailment=False,
            )

            H_MC = mc["H_MC"]

            # --- update memory ONLY if correct ---
            if is_corr:
                memory_by_slot.setdefault(base_qid, []).append(model_answer)

            # --- record ---
            records.append(
                {
                    "dialogue_id": dlg_id,
                    "turn_id": int(row["turn_id"]),
                    "id": ex_id,
                    "base_qid": base_qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                    "semantically_correct": int(is_corr),
                    "H_MC": float(H_MC),
                }
            )

            h_mc_list.append(float(H_MC))
            is_false_list.append(is_false)

    if total == 0:
        raise RuntimeError("No repeat turns evaluated. Check dataset!")

    accuracy = correct / total
    print(f"\nSemantic correctness accuracy (MC-SE): {accuracy:.4f}")
    print(f"Total examples: {total}, Correct: {correct}")

    h_mc = np.array(h_mc_list)
    is_false = np.array(is_false_list)
    accuracies = 1 - is_false

    # --- metrics ---
    auroc_mc = auroc(is_false, h_mc)
    auta_mc = area_under_thresholded_accuracy(accuracies, h_mc)

    print(f"AUROC (H_MC → error): {auroc_mc:.4f}")
    print(f"Area under thresholded accuracy (H_MC): {auta_mc:.4f}")

    # --- save ---
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "accuracy": accuracy,
                "auroc": auroc_mc,
                "auta": auta_mc,
                "results": records,
            },
            f,
        )

    logging.info("Saved MC-SE evaluation to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx_path", type=str, required=True)
    parser.add_argument("--generations_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="mcse_eval.pkl")
    args = parser.parse_args()

    main(
        xlsx_path=args.xlsx_path,
        generations_path=args.generations_path,
        out_path=args.out_path,
    )