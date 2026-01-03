import pandas as pd
import numpy as np
import argparse

from tqdm import tqdm
from collections import defaultdict

from uncertainty.utils import utils
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.memory_sematic_entropy import (
    DialogueMemory,
    memory_conditioned_semantic_entropy,
)

DATA_PATH = "synthetic_multiturn_squad_sample_10_dialogues.xlsx"
OUT_CSV = "mcse_belief_trajectories.csv"

NUM_SAMPLES = 8
TEMPERATURE = 0.8

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
def main():
    np.random.seed(args.seed)
    df = pd.read_excel(DATA_PATH)
    entail_model = EntailmentDeberta()
    model = utils.init_model(args)


    records = []

    for dlg_id, group in tqdm(df.groupby("dialogue_id"), desc="Dialogues"):
        group = group.sort_values("turn_id")
        memory = DialogueMemory()

        for _, row in group.iterrows():
            prompt = row["question"]

            responses = []
            for _ in range(NUM_SAMPLES):
                ans, loglik, emb = model.predict(prompt, TEMPERATURE)
                responses.append((ans, loglik, emb, 0.0))

            mc = memory_conditioned_semantic_entropy(
                responses=responses,
                question_text=prompt,
                memory_facts=memory.get_facts(),
                entail_model=entail_model,
            )

            H = mc["H_MC"]

            records.append({
                "dialogue_id": dlg_id,
                "turn_id": int(row["turn_id"]),
                "H_MC": H,
                "memory_size": len(memory.get_facts()),
                "num_clusters": len(set(mc["semantic_ids"]))
            })

            # commit most likely response
            best_ans, _, _ = model.predict(prompt, temperature=0.1)
            memory.add_fact(best_ans)

    pd.DataFrame(records).to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
