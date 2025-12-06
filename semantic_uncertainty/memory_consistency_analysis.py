import os
import pickle
import logging

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


def load_generations(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(
    xlsx_path: str = "synthetic_multiturn_squad.xlsx",
    generations_path: str = "validation_generations.pkl",
    output_csv: str = "memory_consistency_results.csv",
):
    # 1) Load synthetic multiturn dataset
    df = pd.read_excel(xlsx_path)
    df["id"] = df.apply(
        lambda row: f"{row['dialogue_id']}_t{int(row['turn_id'])}",
        axis=1,
    )
    logging.info("Loaded %d turns from synthetic multiturn SQuAD.", len(df))

    # 2) Load generations
    gens = load_generations(generations_path)
    logging.info("Loaded %d generations from %s", len(gens), generations_path)

    # 3) NLI model for clustering + memory weighting
    entail_model = EntailmentDeberta()

    records = []

    # 4) Process dialogues sequentially
    for dlg_id, group in tqdm(df.groupby("dialogue_id"), desc="Dialogues"):
        group = group.sort_values("turn_id")
        memory = DialogueMemory()
        slot_answers = {}  # base_qid -> earliest answer

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

            # 4a) factual correctness via F1 (improve later with embeddings/NLI)
            f1_fact = f1_score(model_answer, gold_answer)
            factually_correct = int(f1_fact >= 0.8)

            # 4b) self-consistency on re-asks
            if is_repeat == 1 and base_qid in slot_answers:
                prev_answer = slot_answers[base_qid]
                f1_self = f1_score(model_answer, prev_answer)
                self_consistent = int(f1_self >= 0.8)
            else:
                prev_answer = None
                f1_self = None
                self_consistent = None  # not applicable

            # 4c) MC-SE based on high-T responses
            mc = memory_conditioned_semantic_entropy(
                responses=responses,
                question_text=question,
                memory=memory,
                entail_model=entail_model,
                strict_entailment=False,
            )
            H_MC = mc["H_MC"]

            # 4d) update memory only *after* computing H_MC
            memory.add_fact(model_answer)
            if base_qid not in slot_answers:
                slot_answers[base_qid] = model_answer

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
                    "f1_fact": f1_fact,
                    "factually_correct": factually_correct,
                    "f1_self_consistency": f1_self,
                    "self_consistent": self_consistent,
                    "H_MC": H_MC,
                }
            )

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    logging.info("Wrote %d rows to %s", len(out_df), output_csv)


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
        "--output_csv",
        type=str,
        default="memory_consistency_results.csv",
    )

    args = parser.parse_args()
    main(
        xlsx_path=args.xlsx_path,
        generations_path=args.generations_path,
        output_csv=args.output_csv,
    )
