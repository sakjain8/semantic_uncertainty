import os
import pickle
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from uncertainty.uncertainty_measures.memory_sematic_entropy import (
    DialogueMemory,
    memory_conditioned_semantic_entropy,
)
from uncertainty.entailment import EntailmentDeberta   # our DeBERTa wrapper
from uncertainty.utils.eval_utils import f1_score


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER.info("Using device for MC-SE: %s", DEVICE)


def load_generations(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_baseline_eval(
    baseline_eval_path: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Load baseline_semantic_eval.pkl and index by id.

    Expected structure (from your baseline script):
        {
          "accuracy": float,
          "results": [
              {
                  "id": ...,
                  "question": ...,
                  "pred": ...,
                  "gold": ...,
                  "correct": 0/1,
                  "f1": float,
                  "emb_sim": float,
                  "nli_label": int or None,
              },
              ...
          ],
          "thresholds": {...}
        }
    """
    if baseline_eval_path is None or not os.path.exists(baseline_eval_path):
        LOGGER.warning("No baseline eval found at %s", baseline_eval_path)
        return {}

    LOGGER.info("Loading baseline semantic eval from: %s", baseline_eval_path)
    with open(baseline_eval_path, "rb") as f:
        data = pickle.load(f)

    results = data.get("results", [])
    by_id = {str(r["id"]): r for r in results}
    LOGGER.info("Loaded %d baseline eval entries", len(by_id))
    return by_id


def main(
    xlsx_path: str = "synthetic_multiturn_squad.xlsx",
    generations_path: str = "validation_generations.pkl",
    baseline_eval_path: Optional[str] = "baseline_semantic_eval.pkl",
    output_csv: str = "memory_consistency_results.csv",
    output_pkl: str = "memory_consistency_results.pkl",
):
    # 1) Load synthetic multiturn dataset (for dialogue / repeat structure)
    df = pd.read_excel(xlsx_path)
    df["id"] = df.apply(
        lambda row: f"{row['dialogue_id']}_t{int(row['turn_id'])}",
        axis=1,
    )
    LOGGER.info("Loaded %d turns from synthetic multiturn SQuAD.", len(df))

    # 2) Load generations
    gens = load_generations(generations_path)
    LOGGER.info("Loaded %d generations from %s", len(gens), generations_path)

    # 3) Load baseline semantic correctness (for comparison)
    baseline_by_id = load_baseline_eval(baseline_eval_path)

    # 4) NLI model for MC-SE
    entail_model = EntailmentDeberta(device=DEVICE)

    records = []

    # 5) Process dialogues sequentially
    for dlg_id, group in tqdm(df.groupby("dialogue_id"), desc="Dialogues"):
        group = group.sort_values("turn_id")
        memory = DialogueMemory()
        slot_answers = {}  # base_qid -> earliest answer

        for _, row in group.iterrows():
            ex_id = str(row["id"])
            turn_id = int(row["turn_id"])
            is_repeat = int(row["is_repeat"])
            base_qid = str(row["base_qid"])
            gold_answer = str(row["answer"])
            question = str(row["question"])

            if ex_id not in gens:
                LOGGER.warning("Example id %s not in generations; skipping.", ex_id)
                continue

            gen_entry = gens[ex_id]
            most_likely = gen_entry["most_likely_answer"]
            model_answer = str(most_likely["response"])
            responses = gen_entry.get("responses", [])

            # --------------------------
            # A. factual correctness (F1 vs gold)
            # --------------------------
            f1_fact = f1_score(model_answer, gold_answer)
            factually_correct = int(f1_fact >= 0.8)

            # --------------------------
            # B. self-consistency on re-asks
            # --------------------------
            if is_repeat == 1 and base_qid in slot_answers:
                prev_answer = slot_answers[base_qid]
                f1_self = f1_score(model_answer, prev_answer)
                self_consistent = int(f1_self >= 0.8)
            else:
                prev_answer = None
                f1_self = None
                self_consistent = None  # not applicable

            # --------------------------
            # C. memory-conditioned semantic entropy (MC-SE)
            # --------------------------
            mc = memory_conditioned_semantic_entropy(
                responses=responses,
                question_text=question,
                memory=memory,
                entail_model=entail_model,
                strict_entailment=False,
            )
            H_MC = mc["H_MC"]

            # update memory only after computing H_MC
            memory.add_fact(model_answer)
            if base_qid not in slot_answers:
                slot_answers[base_qid] = model_answer

            # --------------------------
            # D. Baseline semantic correctness (from previous eval)
            # --------------------------
            base = baseline_by_id.get(ex_id)
            if base is not None:
                base_correct = int(base["correct"])
                base_f1 = float(base.get("f1", np.nan))
                base_emb_sim = float(base.get("emb_sim", np.nan))
                base_nli = base.get("nli_label", None)
            else:
                base_correct = None
                base_f1 = None
                base_emb_sim = None
                base_nli = None

            records.append(
                {
                    "dialogue_id": dlg_id,
                    "turn_id": turn_id,
                    "id": ex_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                    "is_repeat": is_repeat,
                    "base_qid": base_qid,
                    # factual correctness
                    "f1_fact": f1_fact,
                    "factually_correct": factually_correct,
                    # self-consistency
                    "f1_self_consistency": f1_self,
                    "self_consistent": self_consistent,
                    # MC-SE
                    "H_MC": H_MC,
                    # baseline semantic correctness (for direct comparison)
                    "baseline_correct": base_correct,
                    "baseline_f1": base_f1,
                    "baseline_emb_sim": base_emb_sim,
                    "baseline_nli_label": base_nli,
                }
            )

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    LOGGER.info("Wrote %d rows to %s", len(out_df), output_csv)

    # also save as pkl for easier loading in notebooks
    with open(output_pkl, "wb") as f:
        pickle.dump({"results": records}, f)
    LOGGER.info("Pickle results written to %s", output_pkl)

    # Simple quick summary: how H_MC differs for correct vs incorrect (baseline)
    mask = out_df["baseline_correct"].notna()
    df_sub = out_df[mask]
    if not df_sub.empty:
        corr_acc = df_sub["baseline_correct"].mean()
        mean_H_correct = df_sub.loc[df_sub["baseline_correct"] == 1, "H_MC"].mean()
        mean_H_wrong = df_sub.loc[df_sub["baseline_correct"] == 0, "H_MC"].mean()
        LOGGER.info("On subset with baseline eval:")
        LOGGER.info("  Baseline accuracy: %.4f", corr_acc)
        LOGGER.info("  Mean H_MC (correct):   %.4f", mean_H_correct)
        LOGGER.info("  Mean H_MC (incorrect): %.4f", mean_H_wrong)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx_path",
        type=str,
        default="synthetic_multiturn_squad.xlsx",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default="validation_generations.pkl",
    )
    parser.add_argument(
        "--baseline_eval_path",
        type=str,
        default="baseline_semantic_eval.pkl",
        help="Output of evaluate_semantic_correctness_baseline.py",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="memory_consistency_results.csv",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default="memory_consistency_results.pkl",
    )

    args = parser.parse_args()
    main(
        xlsx_path=args.xlsx_path,
        generations_path=args.generations_path,
        baseline_eval_path=args.baseline_eval_path,
        output_csv=args.output_csv,
        output_pkl=args.output_pkl,
    )
