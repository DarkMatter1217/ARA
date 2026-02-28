import json
from collections import Counter

DATA_FILE = "testing/labeled_dataset_clean.json"

ALLOWED_LABELS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"}


def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)

    label_counter = Counter()
    invalid_labels = []
    null_labels = []
    duplicate_ids = set()
    seen_ids = set()

    for item in data:
        claim_id = item.get("claim_id")
        label = item.get("ground_truth")

        # Duplicate check
        if claim_id in seen_ids:
            duplicate_ids.add(claim_id)
        seen_ids.add(claim_id)

        # Null check
        if label is None:
            null_labels.append(claim_id)
            continue

        # Invalid label check
        if label not in ALLOWED_LABELS:
            invalid_labels.append((claim_id, label))
        else:
            label_counter[label] += 1

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total claims: {total}")
    print("\nLabel Distribution:")
    for label in ALLOWED_LABELS:
        count = label_counter[label]
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{label}: {count} ({percentage:.2f}%)")

    print("\n========== VALIDATION CHECKS ==========")

    if null_labels:
        print(f"\n❌ Null labels found: {len(null_labels)}")
    else:
        print("✅ No null labels.")

    if invalid_labels:
        print(f"\n❌ Invalid labels found: {len(invalid_labels)}")
        print("Examples:", invalid_labels[:5])
    else:
        print("✅ No invalid labels.")

    if duplicate_ids:
        print(f"\n❌ Duplicate claim_ids found: {len(duplicate_ids)}")
        print("Examples:", list(duplicate_ids)[:5])
    else:
        print("✅ No duplicate claim_ids.")

    print("\n======================================\n")


if __name__ == "__main__":
    main()