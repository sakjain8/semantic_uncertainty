from semantic_uncertainty.uncertainty.data.multiwoz import load_multiwoz

if __name__ == "__main__":
    train_data, val_data = load_multiwoz(seed=10)
    print("Train examples:", len(train_data))
    print("Val examples:", len(val_data))

    ex = train_data[0]
    print("\nSample example:")
    print("ID:", ex["id"])
    print("DIALOGUE_ID:", ex["dialogue_id"])
    print("\nCONTEXT:\n", ex["context"])
    print("\nQUESTION:", ex["question"])
    print("\nGOLD ANSWER:", ex["answers"]["text"][0])
