import numpy as np
from typing import List, Dict, Any

from uncertainty.uncertainty_measures.semantic_entropy import (
    get_semantic_ids,
    logsumexp_by_id,
    EntailmentDeberta,
)


class DialogueMemory:
    """
    Simple dialogue memory: stores past answers as free-form text.
    Later you can upgrade this to facts/triples.
    """
    def __init__(self):
        self.facts: List[str] = []

    def add_fact(self, text: str):
        text = (text or "").strip()
        if text:
            self.facts.append(text)

    def get_facts(self) -> List[str]:
        return list(self.facts)


def _sample_logprob(token_log_likelihoods: List[float]) -> float:
    """
    Approximate log p(answer | x) as sum of per-token log-likelihoods.
    """
    if token_log_likelihoods is None:
        return 0.0
    return float(np.sum(token_log_likelihoods))


# memory_semantic_entropy.py

def _weight_sample_against_memory(
    sample_text: str,
    memory_facts: List[str],
    entail_model: EntailmentDeberta,
    contradiction_penalty: float = 0.1,
) -> float:
    """
    w(s, M) in [0,1].

    - If sample contradicts ANY of the provided memory_facts: down-weight.
    - Otherwise: weight = 1.0.

    NOTE: Caller should pass only *relevant* memory facts
          (e.g., previous answers for the same base_qid).
    """
    if not memory_facts:
        return 1.0

    for fact in memory_facts:
        s_to_m = entail_model.check_implication(sample_text, fact)
        m_to_s = entail_model.check_implication(fact, sample_text)

        # 0 = contradiction, 1 = neutral, 2 = entailment
        if s_to_m == 0 or m_to_s == 0:
            return contradiction_penalty

    return 1.0


def memory_conditioned_semantic_entropy(
    responses: List[tuple],
    question_text: str,
    memory_facts: List[str],          # <<< CHANGED: we pass facts directly
    entail_model: EntailmentDeberta,
    strict_entailment: bool = False,
) -> Dict[str, Any]:
    """
    Compute memory-conditioned semantic entropy H_MC for a single turn.

    responses: list of tuples:
      (predicted_answer, token_log_likelihoods, embedding, accuracy)

    memory_facts: list of strings. The *caller* decides what 'memory' means
                  (e.g., previous answers to same base_qid, or empty for no memory).

    Returns dict with:
      - H_MC, cluster_probs_mc, semantic_ids, sample_logprobs, sample_weights
    """
    if len(responses) == 0:
        return {
            "H_MC": 0.0,
            "cluster_probs_mc": [],
            "semantic_ids": [],
            "sample_logprobs": [],
            "sample_weights": [],
        }

    # 1. Collect sample texts and log-probs
    sample_texts, sample_logprobs = [], []
    for (predicted_answer, token_log_likelihoods, embedding, acc) in responses:
        sample_texts.append(str(predicted_answer))
        sample_logprobs.append(_sample_logprob(token_log_likelihoods))

    sample_logprobs = np.array(sample_logprobs)

    # 2. Cluster by semantic meaning using original SE clustering
    semantic_ids = get_semantic_ids(
        strings_list=sample_texts,
        model=entail_model,
        strict_entailment=strict_entailment,
        example={"question": question_text},
    )

    # 3. Compute weights w(s, M) from *slot-specific* memory
    sample_weights = []
    for txt in sample_texts:
        w = _weight_sample_against_memory(txt, memory_facts, entail_model)
        sample_weights.append(w)
    sample_weights = np.array(sample_weights)

    # 4. Combine log p(s|x) + log w(s,M)
    eps = 1e-8
    log_weights = np.log(sample_weights + eps)
    log_weighted = sample_logprobs + log_weights

    # 5. Aggregate to cluster space with log-sum-exp (same as original SE)
    log_cluster_probs_mc = logsumexp_by_id(
        semantic_ids=semantic_ids,
        log_likelihoods=log_weighted,
        agg="sum_normalized",
    )
    cluster_probs_mc = np.exp(log_cluster_probs_mc)
    cluster_probs_mc = cluster_probs_mc / cluster_probs_mc.sum()

    # 6. Memory-conditioned semantic entropy
    H_MC = -np.sum(cluster_probs_mc * np.log(cluster_probs_mc + 1e-30))

    return {
        "H_MC": float(H_MC),
        "cluster_probs_mc": cluster_probs_mc.tolist(),
        "semantic_ids": semantic_ids,
        "sample_logprobs": sample_logprobs.tolist(),
        "sample_weights": sample_weights.tolist(),
    }
