import pickle
import pandas as pd
import os

# Load pickle file
with open("models/coral_bow_stratified.pkl", "rb") as f:
    data = pickle.load(f)

# Create output directory
os.makedirs("data", exist_ok=True)

train_df = pd.DataFrame({
    "id": [f"train_{i}" for i in range(len(data["examples_train"]))],
    "label": data["labels_train"],
    "content": data["examples_train"]
})
train_path = "data/coral_train.tsv"
train_df.to_csv(train_path, sep="\t", index=False)

test_df = pd.DataFrame({
    "id": [f"test_{i}" for i in range(len(data["examples_test"]))],
    "label": data["labels_test"],
    "content": data["examples_test"]
})
test_path = "data/coral_test.tsv"
test_df.to_csv(test_path, sep="\t", index=False)

print(f"Wrote {len(train_df)} training samples to {train_path}")
print(f"Wrote {len(test_df)} test samples to {test_path}")
