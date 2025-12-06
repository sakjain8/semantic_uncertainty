import numpy as np
from tqdm import tqdm

from uncertainty.data.synthetic_multiturn_squad import SyntheticMultiTurnSquadTurns

from uncertainty.uncertainty_measures.semantic_entropy import (
    get_semantic_ids,
    logsumexp_by_id,
    EntailmentDeberta,
)

from uncertainty.models.huggingface_models import HuggingfaceModel    # existing wrapper
from uncertainty.utils.eval_utils import f1_score                                       # you wrote earlier


DATA_PATH = "synthetic_multiturn_squad.xlsx"
MODEL_NAME = "Llama-2-7b-chat"   # example; pick your open model
N_SAMPLES = 10


def sample_with_logprobs(model, prompt: str, n_samples: int):
    """
    Pseudo-function: adapt to how semantic_uncertainty’s code already does sampling.
    Should return: list[str] answers, list[float] log_probs.
    """
    answers = []
    log_probs = []
    for _ in range(n_samples):
        # You likely have something like:
        # txt, loglik, _ = model.predict(prompt, temperature=0.7, return_logprobs=True)
        txt, loglik, _ = model.predict(prompt, temperature=0.7)  # change to your API
        answers.append(txt.strip())
        # loglik should be log p(answer|prompt); if not available, approximate
        log_probs.append(float(loglik))
    return answers, log_probs


def main():
    dataset = SyntheticMultiTurnSquadTurns(DATA_PATH)
    print("Loaded", len(dataset), "turns")

    # load base LM
    model = HuggingfaceModel(
        MODEL_NAME,
        stop_sequences='default',
        max_new_tokens=64
    )

    # load entailment model used in SE code (DeBERTa or GPT-4 etc.)
    from uncertainty.semantic_entropy import EntailmentDeberta
    entail_model = EntailmentDeberta()

    all_labels = []   # 1 = incorrect, 0 = correct
    all_se = []       # semantic entropy
    all_te = []       # token entropy (if you compute it)

    for ex in tqdm(dataset):
        q = ex["question"]
        gold = ex["answer"]

        # for now, no history, just plain question
        prompt = f"Q: {q}\nA:"

        samples, log_probs = sample_with_logprobs(model, prompt, N_SAMPLES)
        log_probs = np.array(log_probs)

        # semantic clustering
        semantic_ids = get_semantic_ids(
            strings_list=samples,
            model=entail_model,
            strict_entailment=False,
            example={"question": q}
        )
        # cluster log-probs
        log_cluster_probs = logsumexp_by_id(
            semantic_ids=semantic_ids,
            log_likelihoods=log_probs,
            agg='sum_normalized'
        )
        cluster_probs = np.exp(log_cluster_probs)
        cluster_probs = cluster_probs / cluster_probs.sum()

        # SE
        se = -np.sum(cluster_probs * np.log(cluster_probs + 1e-30))

        # token-level predictive entropy as baseline
        # here: Rao-style predictive entropy
        p_tokens = np.exp(log_probs)
        p_tokens = p_tokens / p_tokens.sum()
        te = -np.sum(p_tokens * log_probs)

        # choose final answer: cluster with highest prob, then best sample in that cluster
        best_cluster = int(np.argmax(cluster_probs))
        indices = [i for i, cid in enumerate(semantic_ids) if cid == best_cluster]
        # pick sample with highest log_prob in that cluster
        best_idx = max(indices, key=lambda i: log_probs[i])
        final_answer = samples[best_idx]

        # correctness via F1
        f1 = f1_score(final_answer, gold)
        label = 1 if f1 < 1.0 else 0   # 1 = incorrect
        all_labels.append(label)
        all_se.append(se)
        all_te.append(te)

    # compute AUROCs etc.
    from sklearn.metrics import roc_auc_score, average_precision_score

    all_labels = np.array(all_labels)
    all_se = np.array(all_se)
    all_te = np.array(all_te)

    print("Label stats:", np.bincount(all_labels))

    if len(np.unique(all_labels)) > 1:
        auroc_se = roc_auc_score(all_labels, all_se)
        auprc_se = average_precision_score(all_labels, all_se)
        auroc_te = roc_auc_score(all_labels, all_te)
        auprc_te = average_precision_score(all_labels, all_te)

        print("Semantic Entropy: AUROC =", auroc_se, "AUPRC =", auprc_se)
        print("Token Entropy:    AUROC =", auroc_te, "AUPRC =", auprc_te)
    else:
        print("Warning: labels all the same, AUROC undefined")


if __name__ == "__main__":
    main()
