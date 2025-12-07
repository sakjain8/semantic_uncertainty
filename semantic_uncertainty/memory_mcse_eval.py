import argparse
import logging
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from uncertainty.uncertainty_measures.memory_semantic_entropy import (
    DialogueMemory,
    memory_conditioned_semantic_entropy,
)
from uncertainty.utils.eval_utils import (
    f1_score,
    auroc,
    area_under_thresholded_accuracy,
)
# We reuse your semantic correctness rule from the baseline script
from evaluate_semantic_correctness_baseline import is_semantically_correct

logging.basicConfig(level=logging.INFO)


def load_generations(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main(
    xlsx_path: str,
    generations_path: str,
    out_path: str,
) -> None:
    logging.info("Loading synthetic multiturn dataset from %s", xlsx_path)
    df = pd.read_excel(xlsx_path)

    # IMPORTANT: this must match how IDs are stored in validation_generations.pkl
    # In your previous run, keys looked like: 'validation_84fddccf_d1_t1'
    # and dialogue_id was e.g. 'validation_84fddccf_d1'
    if "id" not in df.columns:
        df["id"] = df.apply(
            lambda row: f"{row['dialogue_id']}_t{int(row['turn_id'])}",
            axis=1,
        )
    logging.info("Loaded %d turns from synthetic multiturn SQuAD.", len(df))

    logging.info("Loading generations from %s", generations_path)
    gens = load_generations(generations_path)
    logging.info("Loaded %d generation entries.", len(gens))

    records: List[Dict[str, Any]] = []

    total = 0
    correct = 0

    # We will also store H_MC for AUROC/AUTA
    h_mc_list: List[float] = []
    is_false_list: List[int] = []  # 1 = error, 0 = correct

    # Iterate dialogue by dialogue to maintain memory
    for dlg_id, group in tqdm(df.groupby("dialogue_id"), desc="Dialogues"):
        group = group.sort_values("turn_id")
        memory = DialogueMemory()      # fresh memory per dialogue

        for _, row in group.iterrows():
            ex_id = row["id"]
            question = str(row["question"])
            gold_answer = str(row["answer"])
            is_repeat = int(row["is_repeat"])
            base_qid = str(row["base_qid"])

            if ex_id not in gens:
                logging.warning("Example id %s not in generations; skipping.", ex_id)
                continue

            gen_entry = gens[ex_id]
            most_likely = gen_entry["most_likely_answer"]
            model_answer = str(most_likely["response"])

            # High-T responses used for entropy
            responses = gen_entry.get("responses", [])

            # 1) Semantic correctness (same rule as baseline)
            is_corr, details = is_semantically_correct(model_answer, gold_answer)
            total += 1
            if is_corr:
                correct += 1
                is_false = 0
            else:
                is_false = 1

            # 2) Memory-conditioned semantic entropy H_MC
            mc = memory_conditioned_semantic_entropy(
                responses=responses,
                question_text=question,
                memory=memory,
                entail_model=None,         # handled inside module if needed
                strict_entailment=False,
            )
            H_MC = mc["H_MC"]

            # 3) Update memory AFTER computing H_MC
            memory.add_fact(model_answer)

            # 4) Save per-example record
            records.append(
                {
                    "dialogue_id": dlg_id,
                    "turn_id": int(row["turn_id"]),
                    "id": ex_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                    "is_repeat": is_repeat,
                    "base_qid": base_qid,
                    "semantically_correct": int(is_corr),
                    "f1": details["f1"],
                    "emb_sim": details["sim"],
                    "nli_label": details["nli"],
                    "H_MC": float(H_MC),
                }
            )

            h_mc_list.append(float(H_MC))
            is_false_list.append(is_false)

    if total == 0:
        logging.error("No examples matched between XLSX and generations. Check IDs / paths.")
        return

    acc = correct / total
    print(f"\nSemantic correctness accuracy (MC-SE branch): {acc:.4f}")
    print(f"Total examples: {total}, Correct: {correct}")

    # Convert to numpy arrays for metrics
    h_mc_arr = np.array(h_mc_list)
    is_false_arr = np.array(is_false_list)
    accuracies_arr = 1 - is_false_arr

    # AUROC: how well H_MC separates errors vs correct
    try:
        h_mc_auroc = auroc(is_false_arr, h_mc_arr)
    except Exception as e:
        logging.error("Error computing AUROC for H_MC: %s", e)
        h_mc_auroc = float("nan")

    # Area under thresholded accuracy (selective QA style)
    try:
        h_mc_auta = area_under_thresholded_accuracy(
            accuracies_arr,
            h_mc_arr,
        )
    except Exception as e:
        logging.error("Error computing AUTA for H_MC: %s", e)
        h_mc_auta = float("nan")

    print(f"AUROC (H_MC → error): {h_mc_auroc:.4f}")
    print(f"Area under thresholded accuracy (H_MC): {h_mc_auta:.4f}")

    # Save everything to a pickle, similar to baseline_semantic_eval.pkl
    out_obj = {
        "accuracy": acc,
        "h_mc_auroc": h_mc_auroc,
        "h_mc_auta": h_mc_auta,
        "results": records,
    }

    with open(out_path, "wb") as f:
        pickle.dump(out_obj, f)

    logging.info("Saved MC-SE evaluation to %s", out_path)


if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx_path",
        type=str,
        default="synthetic_multiturn_squad.xlsx",
        help="Path to synthetic multi-turn SQuAD XLSX.",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default="validation_generations.pkl",
        help="Path to validation_generations.pkl from generate_answers.py",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="mcse_eval.pkl",
        help="Where to save MC-SE evaluation results.",
    )
    args = parser.parse_args()
    main(
        xlsx_path=args.xlsx_path,
        generations_path=args.generations_path,
        out_path=args.out_path,
    )