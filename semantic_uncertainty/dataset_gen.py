import random
import hashlib
import pandas as pd
from datasets import load_dataset

random.seed(42)

OUT_XLSX = "synthetic_multiturn_squad.xlsx"

def load_squad_from_hf():
    squad = load_dataset("squad")  # train + validation
    examples = []

    for split in ["train", "validation"]:
        for row in squad[split]:
            context = row["context"]
            question = row["question"]
            answers = row["answers"]["text"]
            if not answers:
                continue
            answer = answers[0]  # first gold answer

            # make a stable id per paragraph using hash of context text
            ctx_hash = hashlib.md5(context.encode("utf-8")).hexdigest()[:8]
            context_id = f"{split}_{ctx_hash}"

            examples.append({
                "context_id": context_id,
                "context": context,
                "question": question,
                "answer": answer,
            })

    return examples


# -----------------------------------------------------------
# 2. Group QAs by their context (paragraph)
# -----------------------------------------------------------
def group_by_context(examples):
    buckets = {}
    for ex in examples:
        cid = ex["context_id"]
        buckets.setdefault(cid, []).append(ex)
    return buckets

# -----------------------------------------------------------
# 3. Build synthetic multi-turn dialogues
# -----------------------------------------------------------


def build_dialogues_from_context(
    context_id,
    ex_list,
    max_dialogues=2,
    q_per_dialogue=4,
    reask_prob=0.6,   # 60% dialogues will have a re-ask
):
    dialogues = []
    if len(ex_list) < q_per_dialogue:
        return dialogues

    random.shuffle(ex_list)
    n_chunks = min(max_dialogues, len(ex_list) // q_per_dialogue)

    for d_idx in range(n_chunks):
        chunk = ex_list[d_idx * q_per_dialogue : (d_idx + 1) * q_per_dialogue]
        dialogue_id = f"{context_id}_d{d_idx+1}"

        base_turns = []
        base_ids = []

        # 1) First, collect the original questions (no turn_id yet)
        for i, qa in enumerate(chunk):
            base_qid = f"{dialogue_id}_q{i+1}"
            base_turns.append({
                "dialogue_id": dialogue_id,
                # turn_id will be filled later
                "turn_id": None,
                "question": qa["question"],
                "answer": qa["answer"],
                "context_id": context_id,
                "is_repeat": 0,
                "repeat_of": "",
                "base_qid": base_qid,
            })
            base_ids.append(base_qid)

        # 2) Decide whether this dialogue gets a re-ask (only in 60% of dialogues)
        do_reask = random.random() < reask_prob

        if do_reask:
            # Choose which original question to re-ask
            re_idx = random.randint(0, len(chunk) - 1)
            re_qa = chunk[re_idx]

            # Choose where to insert the re-ask:
            # anywhere AFTER the original question, from second turn to last
            # Positions are indices in the final turn list (0-based).
            # We must ensure insertion_pos > re_idx so original comes first.
            insertion_pos = random.randint(re_idx + 1, len(chunk))  # can be at end

            reask_turn = {
                "dialogue_id": dialogue_id,
                "turn_id": None,  # fill later
                "question": "(Reask) " + re_qa["question"],
                "answer": re_qa["answer"],
                "context_id": context_id,
                "is_repeat": 1,
                "repeat_of": base_ids[re_idx],
                "base_qid": base_ids[re_idx],
            }

            # Insert the re-ask at the chosen position
            # Example: if insertion_pos == len(chunk) → goes at the end
            base_turns.insert(insertion_pos, reask_turn)

        # 3) Now assign turn_id sequentially based on final order
        turns_with_ids = []
        for turn_id, turn in enumerate(base_turns, start=1):
            turn_copy = dict(turn)
            turn_copy["turn_id"] = turn_id
            turns_with_ids.append(turn_copy)

        # 4) Add to global list
        dialogues.extend(turns_with_ids)

    return dialogues


# -----------------------------------------------------------
# 4. Main
# -----------------------------------------------------------
def main():
    print("Loading SQuAD from HuggingFace…")
    examples = load_squad_from_hf()
    print(f"Total QA pairs loaded: {len(examples)}")

    print("Grouping by context…")
    buckets = group_by_context(examples)

    all_rows = []
    for context_id, ex_list in buckets.items():
        rows = build_dialogues_from_context(
            context_id, ex_list, 
            max_dialogues=2,  # dialogues per context
            q_per_dialogue=4  # turns before re-ask
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"Built {df['dialogue_id'].nunique()} dialogues with {len(df)} turns.")

    print(f"Saving to {OUT_XLSX} …")
    df.to_excel(OUT_XLSX, index=False)

    print("DONE ✓")

if __name__ == "__main__":
    main()
