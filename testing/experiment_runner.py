import csv
import json
import os
from datetime import datetime

from agents import verify_claims_agent
from tools import document_vector_search
from llm import get_llm

INPUT_FILE = "testing/labeled_dataset_clean.json"



# -----------------------------------------
# Baseline RAG
# -----------------------------------------
def baseline_rag_verifier(claim):

    chunks = document_vector_search(claim, top_k=5)
    context = "\n\n".join(chunks)
    prompt = f"""
    You are a careful fact-checking system.

    Claim:
    {claim}

    Context:
    {context}

    Decision Rules:

    SUPPORTED only if:
    - The context clearly and directly confirms the core factual content of the claim.

    CONTRADICTED only if:
    - The context clearly and directly refutes the claim.

    When uncertain → choose INSUFFICIENT.

    Return ONLY one label:
    SUPPORTED
    CONTRADICTED
    INSUFFICIENT
    """

    llm = get_llm("verification")
    response = llm.invoke(prompt)

    verdict_raw = response.content.strip().upper()

    if "SUPPORTED" in verdict_raw:
        return "SUPPORTED"
    elif "CONTRADICTED" in verdict_raw:
        return "CONTRADICTED"
    else:
        return "INSUFFICIENT"


# -----------------------------------------
# Confidence (aligned with graph.py)
# -----------------------------------------
import math
from urllib.parse import urlparse

def compute_claim_confidence(verdict, sources):

    if not sources:
        return 0

    # -------------------------
    # 1. Quality Score
    # -------------------------
    quality_map = {
        "HIGH": 1.0,
        "MEDIUM": 0.7,
        "LOW": 0.3
    }

    quality_scores = [
        quality_map.get(s.get("quality", "LOW"), 0.3)
        for s in sources
    ]

    quality_score = sum(quality_scores) / len(quality_scores)

    # -------------------------
    # 2. Quantity Score (log scaled)
    # -------------------------
    n = len(sources)
    quantity_score = min(1.0, math.log(1 + n) / math.log(1 + 8))

    # -------------------------
    # 3. Domain Diversity Score
    # -------------------------
    domains = set()

    for s in sources:
        try:
            domain = urlparse(s.get("url", "")).netloc
            if domain:
                domains.add(domain)
        except:
            continue

    diversity_score = min(1.0, len(domains) / 5)

    # -------------------------
    # Final Weighted Confidence
    # -------------------------
    confidence = (
        0.5 * quality_score
        + 0.3 * quantity_score
        + 0.2 * diversity_score
    )

    return round(confidence * 100)
# -----------------------------------------
# Safe + Resumeable Experiment Runner
# -----------------------------------------
def run_experiment(run_id=None, model_config="full_ara", limit=None):

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# -----------------------------------------
# ARA Source Dump File (ONLY for ARA)
# -----------------------------------------
    if model_config == "full_ara":
        SOURCE_DUMP_FILE = "testing/evaluation/ara_sources_dump.csv"
    else:
        SOURCE_DUMP_FILE = None
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if limit is not None:
        dataset = dataset[:limit]
    # -----------------------------------------
    # Separate Output Files
    # -----------------------------------------
    if model_config == "baseline_rag":
        OUTPUT_FILE = "testing/evaluation/baseline_rag_results.csv"
    else:
        OUTPUT_FILE = "testing/evaluation/ara_results.csv"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load existing rows to avoid duplicates
    existing_rows = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    existing_keys = {
        (row["claim_id"], row["model_config"])
        for row in existing_rows
    }

    for idx, item in enumerate(dataset):

        claim_id = item["claim_id"]
        doc_id = item["doc_id"]
        claim = item["claim"]
        gt = item["ground_truth"]
        key = (claim_id, model_config)

        if key in existing_keys:
            print(f"Skipping {claim_id} (already computed)")
            continue

        print(f"[{idx+1}/{len(dataset)}] Processing {claim_id}")

        try:

            if model_config == "baseline_rag":
                verdict = baseline_rag_verifier(claim)
                sources = []
            else:
                output = verify_claims_agent([claim])[0]
                verdict = output["verdict"]
                sources = output["sources"]

            confidence = compute_claim_confidence(verdict, sources)

            row = {
                "claim_id": claim_id,
                "doc_id": doc_id,
                "ground_truth": gt,
                "predicted_label": verdict,
                "confidence": confidence / 100.0,
                "model_config": model_config,
                "run_id": run_id,
            }

            # Incremental crash-safe write
            write_header = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0

            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            # -----------------------------------------
            # Dump Extracted Sources (ARA only)
            # -----------------------------------------
            if model_config == "full_ara" and sources:

                source_write_header = (
                    not os.path.exists(SOURCE_DUMP_FILE)
                    or os.path.getsize(SOURCE_DUMP_FILE) == 0
                )

                for src in sources:
                    source_row = {
                        "claim_id": claim_id,
                        "doc_id": doc_id,
                        "claim": claim,
                        "verdict": verdict,
                        "source_url": src.get("url"),
                        "source_quality": src.get("quality"),
                        "run_id": run_id,
                    }

                    with open(SOURCE_DUMP_FILE, "a", newline="", encoding="utf-8") as sf:
                        writer = csv.DictWriter(sf, fieldnames=source_row.keys())
                        if source_write_header:
                            writer.writeheader()
                            source_write_header = False
                        writer.writerow(source_row)# Optional: Dump sources for ARA
        except Exception as e:
            print(f"ERROR on {claim_id}: {e}")
            continue

    print("\nExperiment completed.")
    print(f"Run ID: {run_id}")