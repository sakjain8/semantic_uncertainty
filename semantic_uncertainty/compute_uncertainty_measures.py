"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb

from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils


utils.setup_logger()

EXP_DETAILS = 'experiment_details.pkl'


def compute_memory_weights(memory_text, responses, entailment_model, example):
    """
    Compute a memory-consistency weight w_j for each generated response r_j.

    entailment_model.check_implication(memory_text, r_j, example=example) returns:
        2 -> entailment
        1 -> neutral
        0 -> contradiction

    We map these to:
        entailment    -> 1.0
        neutral       -> 0.7
        contradiction -> 0.3
    """
    if not memory_text or not memory_text.strip():
        # No memory yet: all weights 1.0 (reduces to vanilla SE)
        return [1.0] * len(responses)

    weights = []
    for r in responses:
        try:
            label = entailment_model.check_implication(memory_text, r, example=example)
        except TypeError:
            # some entailment models don't use 'example', but accept **kwargs
            label = entailment_model.check_implication(memory_text, r)

        if label == 2:         # entailment
            w = 1.0
        elif label == 1:       # neutral
            w = 0.7
        else:                  # contradiction
            w = 0.3
        weights.append(w)
    return weights


def compute_mc_semantic_entropy(semantic_ids, log_liks_agg, weights):
    """
    Memory-Conditioned Semantic Entropy (MC-SE).

    Vanilla SE:
        - cluster generations into semantic_ids
        - convert log_liks_agg -> probabilities over samples
        - sum probability mass per semantic cluster
        - entropy over that cluster distribution

    MC-SE:
        p_j ∝ exp(log_liks_agg[j]) * w_j

        cluster probs π_k ∝ sum_{j in cluster k} p_j

        H_MC = - sum_k π_k log π_k
    """
    semantic_ids = list(semantic_ids)
    log_liks_agg = list(log_liks_agg)
    weights = list(weights)
    assert len(semantic_ids) == len(log_liks_agg) == len(weights)

    probs = []
    for logp, w in zip(log_liks_agg, weights):
        probs.append(np.exp(logp) * w)

    Z = np.sum(probs)
    if Z == 0 or not np.isfinite(Z):
        # fallback: vanilla distribution
        log_probs = log_liks_agg
    else:
        probs = np.array(probs) / Z
        unique_ids = sorted(set(semantic_ids))
        cluster_probs = []
        for uid in unique_ids:
            idxs = [i for i, sid in enumerate(semantic_ids) if sid == uid]
            cluster_probs.append(probs[idxs].sum())

        cluster_probs = np.array(cluster_probs)
        cluster_probs = cluster_probs / cluster_probs.sum()
        log_probs = np.log(cluster_probs + 1e-12)

    return predictive_entropy_rao(log_probs)


def main(args):

    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"

    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    else:
        logging.info('Reuse active wandb id.')

        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Train on run %s, eval on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)
        is_ood_eval = True
        api = wandb.Api()
        old_run_train = api.run(f'{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}')
        filename = 'train_generations.pkl'
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False
        if args.compute_p_ik or args.compute_p_ik_answerable:
            train_generations_pickle = restore('train_generations.pkl')
            with open(train_generations_pickle.name, 'rb') as infile:
                train_generations = pickle.load(infile)

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model (used for SE + semantic correctness + memory checks).
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')
    else:
        entailment_model = None  # you should keep compute_predictive_entropy=True for our setup

    if args.compute_p_true_in_compute_stage:
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)
        del pt_validation_dataset

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric(old_exp['args'].metric))
        del p_true_responses
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80 * '#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80 * '#')

    if args.recompute_accuracy:
        logging.warning('Recompute accuracy enabled (not used for p_true).')
        metric = utils.get_metric(args.metric)

    # Restore outputs from `generate_answers.py` run.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict['semantic_ids'] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0

    # Build ordered list of examples (dialogue_id, turn_index, tid)
    ordered_items = []
    for tid, ex in validation_generations.items():
        dlg_id = ex.get('dialogue_id')
        turn_idx = ex.get('turn_index', 0)
        ordered_items.append((dlg_id, turn_idx, tid))
    ordered_items.sort(key=lambda x: (x[0], x[1]))

    from collections import defaultdict as _dd
    dialogue_memories = _dd(list)

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # New semantic labels
    validation_semantic_is_true = []
    validation_hallucinated = []
    validation_memory_inconsistent = []

    # Loop over datapoints in dialogue order
    for idx, (dlg_id, turn_idx, tid) in enumerate(ordered_items):

        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        # Dialogue-level memory: previous gold system utterances
        if dialogue_memories[dlg_id]:
            memory_text = "\n".join(dialogue_memories[dlg_id])
        else:
            memory_text = ""

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        # === SEMANTIC CORRECTNESS (gold vs answer) ===
        gold_answers = example['reference']['answers']['text']
        if gold_answers:
            gold = gold_answers[0]
        else:
            gold = ""

        label_ans = entailment_model.check_implication(
            gold,
            most_likely_answer['response'],
            example=example,
        )
        semantic_correct = 0.0 if label_ans == 0 else 1.0
        validation_semantic_is_true.append(semantic_correct)

        # hallucination: only if it contradicts the gold
        hallucinated = 1.0 - semantic_correct
        validation_hallucinated.append(hallucinated)
        
        # use semantic correctness as the main "true" label
        validation_is_true.append(semantic_correct)

        # === MEMORY CONSISTENCY (context + memory vs answer) ===
        if memory_text:
            full_premise = context + "\n" + memory_text
        else:
            full_premise = context

        ctx_label = entailment_model.check_implication(
            full_premise,
            most_likely_answer['response'],
            example=example,
        )
        memory_inconsistent = 1.0 if ctx_label == 0 else 0.0
        validation_memory_inconsistent.append(memory_inconsistent)

        # Answerable / embeddings for p_ik, etc.
        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        # === Predictive & semantic entropy (including MC-SE) ===
        if args.compute_predictive_entropy:
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                entropies['context_entails_response'].append(
                    context_entails_response(context, responses, entailment_model))

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)
            result_dict['semantic_ids'].append(semantic_ids)

            # Cluster assignment entropy
            entropies['cluster_assignment_entropy'].append(
                cluster_assignment_entropy(semantic_ids))

            # Token log-likelihood aggregation
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Naive entropy
            entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))

            # Semantic entropy (vanilla SE)
            log_likelihood_per_semantic_id = logsumexp_by_id(
                semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            # === Memory-Conditioned Semantic Entropy (MC-SE) ===
            mem_weights = compute_memory_weights(
                memory_text=memory_text,
                responses=responses,
                entailment_model=entailment_model,
                example=example,
            )
            pe_mc = compute_mc_semantic_entropy(
                semantic_ids=semantic_ids,
                log_liks_agg=log_liks_agg,
                weights=mem_weights,
            )
            entropies['semantic_entropy_mc'].append(pe_mc)

            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            logging.info(80 * '#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        # Update dialogue memory with this turn's gold system utterance
        ref = example.get('reference')
        gold_answers_ref = None
        if ref is not None and 'answers' in ref and 'text' in ref['answers']:
            gold_answers_ref = ref['answers']['text']
        elif 'answers' in example and 'text' in example['answers']:
            gold_answers_ref = example['answers']['text']

        if gold_answers_ref:
            dialogue_memories[dlg_id].append(gold_answers_ref[0])

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    # === Summary metrics & labels ===
    logging.info('Semantic accuracy: %f', np.mean(validation_semantic_is_true))
    result_dict['validation_semantic_is_true'] = validation_semantic_is_true

    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    # Hallucinations: 1 = hallucinated (not semantically correct)
    result_dict['validation_hallucinated'] = validation_hallucinated
    logging.info('Hallucination rate: %f', np.mean(validation_hallucinated))

    # Memory inconsistency: 1 = contradicts context + memory
    result_dict['validation_memory_inconsistent'] = validation_memory_inconsistent
    logging.info('Memory inconsistency rate: %f', np.mean(validation_memory_inconsistent))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)
        # Make sure MC-SE is directly accessible
        result_dict['uncertainty_measures']['semantic_entropy_mc'] = entropies['semantic_entropy_mc']

    # === p_ik training and evaluation ===
    if args.compute_p_ik or args.compute_p_ik_answerable:
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    if args.compute_p_ik:
        logging.info('Starting training p_ik on train embeddings.')
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
        logging.info('Finished training p_ik on train embeddings.')

    if args.compute_p_ik_answerable:
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    utils.save(result_dict, 'uncertainty_measures.pkl')

    if args.compute_predictive_entropy and hasattr(entailment_model, "save_prediction_cache"):
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)
    main(args)
