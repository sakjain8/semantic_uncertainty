import pickle
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

# ✅ Import your memory-conditioned SE implementation
from uncertainty.uncertainty_measures.memory_sematic_entropy import DialogueMemory, memory_conditioned_semantic_entropy

# ✅ Use the same entailment model as base SE pipeline
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta


def load_base_se(uncertainty_path: str, generations_path: str):
    """
    Load base semantic entropy and align it with example IDs from validation_generations.pkl.
    """
    with open(uncertainty_path, "rb") as f:
        u = pickle.load(f)

    if "uncertainty_measures" not in u:
        raise KeyError(
            "Key 'uncertainty_measures' not found in uncertainty_measures.pkl. "
            "Check that compute_uncertainty_measures.py ran correctly."
        )

    um = u["uncertainty_measures"]

    if "semantic_entropy" not in um:
        raise KeyError(
            "semantic_entropy not found in uncertainty_measures.pkl. "
            "Did you run generate_answers.py with --compute_uncertainties "
            "and ensure compute_uncertainty_measures.py did not crash?"
        )

    se_list = np.array(um["semantic_entropy"], dtype=float)

    with open(generations_path, "rb") as f:
        generations = pickle.load(f)

    # Dict iteration order == insertion order; compute_uncertainty_measures
    # iterates over validation_generations in this order, so we can align.
    ids_in_order = list(generations.keys())

    if len(ids_in_order) != len(se_list):
        raise ValueError(
            f"Length mismatch: {len(ids_in_order)} validation examples vs "
            f"{len(se_list)} semantic_entropy values."
        )

    id_to_se = dict(zip(ids_in_order, se_list))
    return generations, id_to_se


def load_semantic_correctness(baseline_eval_path: str):
    """
    Load semantic correctness labels from your baseline_semantic_eval.pkl.

    Expects structure:
      {
        "accuracy": float,
        "results": [
          {
            "id": ...,
            "correct": 0/1,
            ...
          },
          ...
        ]
      }
    """
    with open(baseline_eval_path, "rb") as f:
        base = pickle.load(f)

    if "results" not in base:
        raise KeyError(
            "baseline_semantic_eval.pkl missing 'results' key. "
            "Check that baseline_semantic_eval.py ran correctly."
        )

    results = base["results"]
    id_to_correct = {r["id"]: int(r["correct"]) for r in results}
    return id_to_correct


def compute_memory_conditioned_se(
    generations,
    entailment_model=None,
    use_gold_in_memory: bool = True,
):
    """
    Compute H_MC for all examples in validation_generations.pkl.

    Assumes each entry in `generations[ex_id]` looks like:
      {
        'question': ...,
        'context': ...,
        'most_likely_answer': {...},
        'reference': {...},
        'responses': [(pred, token_log_liks, emb, acc), ...],
        # optionally:
        # 'dialogue_id': ...,
        # 'turn_id': ...
      }

    Returns:
      id_to_hmc: dict[example_id -> H_MC value]
    """

    if entailment_model is None:
        entailment_model = EntailmentDeberta()

    # One memory per dialogue
    memories = defaultdict(DialogueMemory)

    H_MC_list = []
    ids_in_order = list(generations.keys())

    for ex_id in ids_in_order:
        entry = generations[ex_id]
        question = entry["question"]
        responses = entry["responses"]

        # Try to use dialogue_id if present; otherwise fall back to treating
        # each example as its own dialogue (no cross-turn memory).
        if "dialogue_id" in entry and entry["dialogue_id"] is not None:
            did = entry["dialogue_id"]
        else:
            did = ex_id  # degenerate: no shared memory across turns

        mem = memories[did]

        # Compute memory-conditioned SE for this turn
        mc_result = memory_conditioned_semantic_entropy(
            responses=responses,
            question_text=question,
            memory=mem,
            entail_model=entailment_model,
            strict_entailment=False,
        )

        H_MC_list.append(mc_result["H_MC"])

        # === Memory update rule ===
        # Option A (current): add gold answer to memory (safer for evaluation)
        if use_gold_in_memory:
            ref = entry["reference"]
            gold_texts = ref["answers"]["text"]
            if isinstance(gold_texts, list) and len(gold_texts) > 0:
                mem.add_fact(gold_texts[0])

        # Option B (alternative): use model's answer instead of gold
        # else:
        #     mem.add_fact(entry["most_likely_answer"]["response"])

    H_MC_arr = np.array(H_MC_list, dtype=float)
    id_to_hmc = dict(zip(ids_in_order, H_MC_arr))
    return id_to_hmc


def main():
    # ---- PATHS: adjust if needed ----
    generations_path = "validation_generations.pkl"
    uncertainty_path = "uncertainty_measures.pkl"
    baseline_eval_path = "baseline_semantic_eval.pkl"

    print("Loading base semantic entropy and generations...")
    generations, id_to_se = load_base_se(uncertainty_path, generations_path)

    print("Loading semantic correctness labels...")
    id_to_correct = load_semantic_correctness(baseline_eval_path)

    # Ensure overlap (in case some ids were filtered in baseline eval)
    common_ids = list(set(generations.keys()) & set(id_to_correct.keys()))
    common_ids.sort()

    print(f"Common examples between SE and correctness labels: {len(common_ids)}")
    if len(common_ids) == 0:
        raise ValueError("No overlapping IDs between generations and baseline_eval results.")

    # Build aligned arrays for base SE and correctness
    base_se = np.array([id_to_se[i] for i in common_ids], dtype=float)
    correct = np.array([id_to_correct[i] for i in common_ids], dtype=int)

    # Compute memory-conditioned SE
    print("Computing memory-conditioned semantic entropy (H_MC)...")
    entailment_model = EntailmentDeberta()
    id_to_hmc = compute_memory_conditioned_se(
        generations, entailment_model=entailment_model
    )

    H_MC = np.array([id_to_hmc[i] for i in common_ids], dtype=float)

    # ------------- EVALUATION NUMBERS YOU CARE ABOUT --------------

    # 1. Mean SE / H_MC for correct vs incorrect
    se_correct = base_se[correct == 1]
    se_wrong = base_se[correct == 0]

    hmc_correct = H_MC[correct == 1]
    hmc_wrong = H_MC[correct == 0]

    print("\n=== MEAN ENTROPY BY CORRECTNESS ===")
    print(f"Base SE  - mean(correct): {se_correct.mean():.4f}, mean(wrong): {se_wrong.mean():.4f}")
    print(f"H_MC     - mean(correct): {hmc_correct.mean():.4f}, mean(wrong): {hmc_wrong.mean():.4f}")

    # 2. AUROC: how well each entropy detects errors (higher is better)
    # Treat label=1 as "wrong" for ROC AUC; so we use (1 - correct).
    y = (1 - correct)  # 1 = error, 0 = correct

    try:
        auroc_se = roc_auc_score(y, base_se)
        auroc_hmc = roc_auc_score(y, H_MC)
        print("\n=== AUROC FOR ERROR DETECTION (1 = WRONG) ===")
        print(f"Base SE AUROC : {auroc_se:.4f}")
        print(f"H_MC   AUROC  : {auroc_hmc:.4f}")
    except Exception as e:
        print("\nCould not compute AUROC (probably not enough pos/neg examples):", e)

    # 3. (Optional) Correlation with error indicator
    if len(np.unique(y)) > 1:
        corr_se = np.corrcoef(y, base_se)[0, 1]
        corr_hmc = np.corrcoef(y, H_MC)[0, 1]
        print("\n=== CORRELATION WITH ERROR LABEL (1 = WRONG) ===")
        print(f"Base SE corr : {corr_se:.4f}")
        print(f"H_MC   corr  : {corr_hmc:.4f}")
    else:
        print("\nCorrelation not defined (only one class in y).")

    # 4. Save results for plotting / further analysis
    out = {
        "ids": common_ids,
        "base_se": base_se,
        "H_MC": H_MC,
        "correct": correct,
    }
    with open("se_vs_hmc_eval.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\nSaved detailed arrays to se_vs_hmc_eval.pkl")


if __name__ == "__main__":
    main()
