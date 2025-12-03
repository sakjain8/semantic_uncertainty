# semantic_uncertainty/uncertainty/data/multiwoz.py

from typing import List, Dict, Tuple
from datasets import load_dataset


def load_multiwoz(seed: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    Load MultiWOZ v2.2 from HuggingFace and convert to the QA format
    used by the semantic_uncertainty repo.

    Each returned example corresponds to ONE SYSTEM turn and has:
      - context: all previous turns in the dialogue (USER + SYSTEM)
      - question: the last USER utterance before this SYSTEM turn
      - answers['text'][0]: the gold SYSTEM utterance at this turn
      - id: unique dialogue-turn identifier
    """

    # 1) Load HF dataset (you already used this)
    ds = load_dataset("pfb30/multi_woz_v22")

    def convert(split_name: str) -> List[Dict]:
        data: List[Dict] = []

        for dialog in ds[split_name]:
            dialogue_id = dialog["dialogue_id"]   # check: print one example if unsure
            turns = dialog["turns"]              # list of {speaker, utterance, ...}

            history: List[str] = []              # "SPEAKER: utt" strings
            last_user_utt: str = None            # cache last user utterance

            for turn in turns:
                speaker_raw = turn["speaker"]    # e.g. "USER" / "SYSTEM"
                utt = turn["utterance"]

                speaker = speaker_raw.upper().strip()
                history.append(f"{speaker}: {utt}")

                if speaker == "USER":
                    # we’ll treat this as the current "question"
                    last_user_utt = utt

                elif speaker == "SYSTEM":
                    # we only create an example on SYSTEM turns
                    if last_user_utt is None:
                        # if system speaks first (rare), skip
                        continue

                    # all previous turns (both USER & SYSTEM) before this reply
                    full_context = "\n".join(history[:-1])

                    # turn_id is usually provided; if not, fall back to index
                    turn_id = turn.get("turn_id", len(history) - 1)

                    ex = {
                        "id": f"{dialogue_id}_{turn_id}",
                        "context": full_context,
                        "question": last_user_utt,
                        "answers": {"text": [utt]},
                        # keep extra fields for memory work later
                        "dialogue_id": dialogue_id,
                        "turn_index": turn_id,
                    }
                    data.append(ex)

            # end for turn
        # end for dialog

        return data

    # 2) Convert splits. Repo usually uses train + validation for QA.
    train_data = convert("train")
    val_data = convert("validation")

    return train_data, val_data
