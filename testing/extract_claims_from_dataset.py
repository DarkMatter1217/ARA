import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from datetime import datetime
from agents import extract_claims_agent

from dotenv import load_dotenv
load_dotenv()

INPUT_FILE = "testing/dataset.json"
OUTPUT_FILE = "testing/raw_claims_pool.json"
MAX_CLAIMS_PER_DOC = 8


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

import time 
def main():
    # ----------------------------
    # Load full dataset
    # ----------------------------
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    log(f"Loaded {len(dataset)} documents from dataset.json")

    # ----------------------------
    # Load existing results (if any)
    # ----------------------------
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        log(f"Loaded {len(existing_results)} previously processed documents")
    else:
        existing_results = []
        log("No existing raw_claims_pool.json found. Starting fresh.")

    # Build lookup set of completed doc_ids
    processed_doc_ids = {item["doc_id"] for item in existing_results}

    results = existing_results[:]  # preserve existing
    total_claims = sum(len(item["claims"]) for item in existing_results)

    # ----------------------------
    # Process only missing documents
    # ----------------------------
    for idx, doc in enumerate(dataset):
        doc_id = doc["doc_id"]

        if doc_id in processed_doc_ids:
            log(f"Skipping {doc_id} (already processed)")
            continue

        text = doc["text"]

        log(f"Extracting claims for {doc_id} ({idx+1}/{len(dataset)})")

        try:
            claims = extract_claims_agent(
                text=text,
                max_claims=MAX_CLAIMS_PER_DOC
            )

            claims = list(dict.fromkeys([c.strip() for c in claims]))

            total_claims += len(claims)

            results.append({
                "doc_id": doc_id,
                "filename": doc["filename"],
                "claims": claims
            })

            log(f"Extracted {len(claims)} claims from {doc_id}")

            # Save incrementally after each successful document
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            log("Saved progress.")

            print("Sleeping 60 seconds to reset Cerebras TPM window...")
            time.sleep(60)

        except Exception as e:
            log(f"ERROR extracting claims for {doc_id}: {e}")
            continue

    log("Claim extraction complete.")
    log(f"Total claims extracted: {total_claims}")
    log(f"Saved to {OUTPUT_FILE}")
if __name__ == "__main__":
    main()