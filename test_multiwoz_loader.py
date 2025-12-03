# test_multiwoz_loader.py
from  semantic_uncertainty.uncertainty.data.multiwoz import load_multiwoz

if __name__ == "_main_":
    print("Loading MultiWOZ with load_multiwoz(seed=10)...")
    train_data, val_data = load_multiwoz(seed=10)
    print("Train examples:", len(train_data))
    print("Val examples:", len(val_data))

    print("\n=== Sample example from train_data[0] ===")
    ex = train_data[0]
    print("ID:", ex.get("id"))
    print("DIALOGUE_ID:", ex.get("dialogue_id"))

    print("\nCONTEXT:\n", ex.get("context"))
    print("\nQUESTION:", ex.get("question"))
    print("\nGOLD ANSWER:", ex.get("answers", {}).get("text", ["<no answer>"])[0])


