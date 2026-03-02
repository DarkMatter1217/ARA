from typing import List
from llm import get_llm
from prompts import (
    CHAT_PROMPT,
    SUMMARY_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    EXTERNAL_VERIFICATION_PROMPT,
)
from tools import reset_evidence_store, vector_search
import re

def classify_source_quality(url: str) -> tuple[str, str]:
    u = url.lower()

    if any(domain in u for domain in [
        "arxiv.org",
        "openreview.net",
        "aclweb.org",
        "ieee.org",
        "springer.com",
        "sciencedirect.com",
        "nature.com",
        "acm.org",
        "jstor.org",
        "pubmed.ncbi.nlm.nih.gov",
    ]):
        return "HIGH", "Peer-reviewed / academic source"

    if any(domain in u for domain in [
        ".edu",
        ".gov",
        "who.int",
        "nih.gov",
        "nasa.gov",
    ]):
        return "MEDIUM", "Institutional or government source"

    return "LOW", "Non-peer-reviewed or informal source"

import re

def latex_to_plain_math(text: str) -> str:
    replacements = {
        r"\\log": "log",
        r"\\alpha": "alpha",
        r"\\beta": "beta",
        r"\\delta": "delta",
        r"\\times": "*",
        r"\\cdot": "*",
        r"\^\{([^}]+)\}": r"^\1",
        r"_\{([^}]+)\}": r"_\1",
        r"\\frac\{([^}]+)\}\{([^}]+)\}": r"\1 / \2",
        r"\\text\{([^}]+)\}": r"\1",
    }

    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    return text

def normalize_output_for_chainlit(text: str) -> str:
    text = latex_to_plain_math(text)
    text = latex_to_plain_math(text)
    text = re.sub(r"\n([\-•*])", r"\n\n\1", text)
    return text.strip()

def chat_agent(context: str, query: str) -> str:
    llm = get_llm()

    prompt = CHAT_PROMPT.format(
        context=context,
        question=query,
    )

    response = llm.invoke(prompt)
    return normalize_output_for_chainlit(response.content.strip())

def summarize_agent(text: str) -> str:
    llm = get_llm("summarization")
    prompt = SUMMARY_PROMPT.format(text=text)
    raw = llm.invoke(prompt).content.strip()

    return normalize_output_for_chainlit(raw)

import json
def extract_claims_agent(text: str, max_claims: int = 5) -> List[str]:
    import json
    import re
    import time

    llm = get_llm("extraction")

    prompt = CLAIM_EXTRACTION_PROMPT.format(
        text=text,
        max_claims=max_claims,
    )

    for attempt in range(5):
        try:
            response = llm.invoke(prompt)
            response_text = response.content.strip()

            # Remove markdown fences
            if response_text.startswith("```"):
                response_text = response_text.replace("```json", "")
                response_text = response_text.replace("```", "")
                response_text = response_text.strip()

            # Extract JSON block
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                response_text = match.group(0)

            data = json.loads(response_text)
            claims = data.get("claims", [])

            if not isinstance(claims, list):
                raise ValueError("Claims must be a list.")

            return [str(c).strip() for c in claims][:max_claims]

        except Exception as e:
            if attempt == 4:
                raise ValueError(
                    f"Claim extraction failed after retries. Last error: {str(e)}"
                )

            time.sleep(2 * (attempt + 1))

    return []


def verify_claims_agent(claims: List[str]) -> List[dict]:
    from tools import (
        tavily_search,
        add_sources_to_evidence_store,
        retrieve_evidence_context
    )
    import json

    verification_llm = get_llm("verification")
    results = []

    for claim in claims:
        from tools import reset_evidence_store
        reset_evidence_store()
        # Step 1 — Generate Search Queries
        query_prompt = f"""
Generate 3 precise search queries to verify this claim.

Claim:
{claim}

Return JSON:
{{"queries": ["...", "..."]}}
"""
        q_response = verification_llm.invoke(query_prompt)
        q_text = q_response.content.strip()

        try:
            q_json = json.loads(q_text)
            queries = q_json.get("queries", [])
        except:
            queries = [claim]

        # Step 2 — Tavily Retrieval
        all_sources = []
        seen_urls = set()

        for q in queries:
            sources = tavily_search(q)

            for s in sources:
                url = s.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_sources.append(s)
        if all_sources:
            add_sources_to_evidence_store(all_sources)

        # Step 3 — Retrieve Relevant Evidence
        evidence_chunks = retrieve_evidence_context(claim, top_k=5)

        formatted_context = ""
        used_sources = []

        for chunk in evidence_chunks:
            formatted_context += f"\nSource: {chunk['url']}\n{chunk['content']}\n"
            used_sources.append({
                "url": chunk["url"],
                "quality": chunk["quality"]
            })

        # Step 4 — Reasoning
        verification_prompt = f"""
        You are a strict scientific fact-checking system.

        Claim:
        {claim}

        Evidence:
        {formatted_context}

        Decision Policy (HIGH PRECISION MODE):

        SUPPORTED only if:
        - At least TWO independent sources agree.
        - Evidence directly matches key entities, numbers, datasets, or metrics.
        - There is NO contradicting evidence.

        CONTRADICTED only if:
        - Reliable evidence directly refutes the claim.

        INSUFFICIENT if:
        - Evidence is partial.
        - Only one weak source exists.
        - Numerical values do not exactly match.
        - Sources are vague or secondary.
        - There is uncertainty.

        When uncertain → choose INSUFFICIENT.

        Return ONLY one label:
        SUPPORTED
        CONTRADICTED
        INSUFFICIENT
        """

        verdict_response = verification_llm.invoke(verification_prompt)
        verdict_raw = verdict_response.content.strip().upper()

        if "SUPPORTED" in verdict_raw:
            verdict = "SUPPORTED"
        elif "CONTRADICTED" in verdict_raw:
            verdict = "CONTRADICTED"
        else:
            verdict = "INSUFFICIENT"

        results.append({
            "claim": claim,
            "verdict": verdict,
            "sources": used_sources
        })

    return results    