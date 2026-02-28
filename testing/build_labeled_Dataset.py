import json
import random
from datetime import datetime

INPUT_FILE = "testing/raw_claims_pool.json"
OUTPUT_FILE = "testing/labeled_dataset.json"
SAMPLE_SIZE = 500


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def main():
    # ----------------------------
    # Load extracted claims
    # ----------------------------
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # ----------------------------
    # Flatten claims
    # ----------------------------
    all_claims = []

    for doc in raw_data:
        doc_id = doc["doc_id"]
        for idx, claim in enumerate(doc["claims"], start=1):
            all_claims.append({
                "claim_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "claim": claim.strip(),
                "ground_truth": None
            })

    total_available = len(all_claims)
    log(f"Total extracted claims available: {total_available}")

    if total_available < SAMPLE_SIZE:
        raise ValueError(
            f"Not enough claims to sample {SAMPLE_SIZE}. Only {total_available} available."
        )

    # ----------------------------
    # Random Sampling
    # ----------------------------
    random.seed(42)  # reproducibility
    sampled_claims = random.sample(all_claims, SAMPLE_SIZE)

    # ----------------------------
    # Save labeled dataset template
    # ----------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sampled_claims, f, indent=2)

    log(f"Randomly sampled {SAMPLE_SIZE} claims.")
    log(f"Saved labeled dataset template to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()