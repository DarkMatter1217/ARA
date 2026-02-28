import json
with open("testing/labeled_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

small = data[:5]

with open("testing/labeled_dataset_small.json", "w") as f:
    json.dump(small, f, indent=2)