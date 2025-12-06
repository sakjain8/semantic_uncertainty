import pandas as pd
from datasets import Dataset, DatasetDict


def load_synthetic_multiturn_squad(path="synthetic_multiturn_squad.xlsx"):
    """
    Load the synthetic multi-turn SQuAD Excel as a HuggingFace-style DatasetDict
    with a 'validation' split that looks like SQuAD examples.
    """
    df = pd.read_excel(path)

    records = []
    for _, row in df.iterrows():
        q = str(row["question"])
        a = str(row["answer"])
        dlg_id = str(row["dialogue_id"])
        turn_id = int(row["turn_id"])

        ex_id = f"{dlg_id}_t{turn_id}"

        records.append({
            "id": ex_id,
            "context": "",  # no context for now (context-free setting)
            "question": q,
            "answers": {
                "text": [a],
                "answer_start": [0],   # dummy; required by squad metric
            },
            # you can keep extra info if you want:
            "dialogue_id": dlg_id,
            "turn_id": turn_id,
            "is_repeat": int(row["is_repeat"]),
            "base_qid": str(row["base_qid"]),
            "repeat_of": str(row["repeat_of"]) if not pd.isna(row["repeat_of"]) else "",
        })

    ds = Dataset.from_list(records)
    # We use only a 'validation' split; generate_answers expects a DatasetDict
    return DatasetDict({"validation": ds})
