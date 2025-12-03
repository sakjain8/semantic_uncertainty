from typing import List, Dict, Tuple
from datasets import load_dataset


def load_multiwoz(seed: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    Load MultiWOZ v2.2 from HuggingFace (pfb30/multi_woz_v22) and convert to
    the QA format used by the semantic_uncertainty repo.

    Actual HF structure for each dialogue:

        {
          "dialogue_id": "PMUL4398.json",
          "services": [...],
          "turns": {
              "turn_id":   ["0", "1", ..., "11"],
              "speaker":   [0, 1, 0, 1, ...],   # 0 = USER, 1 = SYSTEM
              "utterance": ["i need a place...", "I have several options...", ...],
              "frames":    [...],
              "dialogue_acts": [...]
          }
        }

    We convert EACH SYSTEM turn into ONE QA example:

      context  = all previous turns (USER + SYSTEM), formatted "SPEAKER: text"
      question = last USER utterance before this SYSTEM turn
      answers['text'][0] = the SYSTEM utterance at this turn
    """

    ds = load_dataset("pfb30/multi_woz_v22")

    def convert(split_name: str) -> List[Dict]:
        data: List[Dict] = []

        for dialog in ds[split_name]:
            dialogue_id = dialog["dialogue_id"]
            turns = dialog["turns"]  # dict of lists

            turn_ids   = turns["turn_id"]    # list of strings
            speakers   = turns["speaker"]    # list of ints (0=user, 1=system)
            utterances = turns["utterance"]  # list of strings

            history: List[str] = []          # we store "SPEAKER: utterance"
            last_user_utt: str = None

            for idx in range(len(utterances)):
                spk_id = speakers[idx]
                utt = utterances[idx]
                turn_id = turn_ids[idx]

                speaker = "USER" if spk_id == 0 else "SYSTEM"

                # add this utterance to running history
                history.append(f"{speaker}: {utt}")

                if speaker == "USER":
                    # update the latest user utterance
                    last_user_utt = utt

                else:  # SYSTEM turn
                    if last_user_utt is None:
                        # if a system speaks before any user (unlikely), skip
                        continue

                    # context = everything BEFORE this system utterance
                    # history currently includes this system utt at the end, so:
                    full_context = "\n".join(history[:-1])

                    ex = {
                        "id": f"{dialogue_id}_{turn_id}",
                        "context": full_context,
                        "question": last_user_utt,
                        "answers": {"text": [utt]},
                        "dialogue_id": dialogue_id,
                        "turn_index": int(turn_id),
                    }
                    data.append(ex)

        return data

    train_data = convert("train")
    val_data = convert("validation")
    return train_data, val_data