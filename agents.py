from typing import List
from llm import get_llm
from prompts import (
    CHAT_PROMPT,
    SUMMARY_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    EXTERNAL_VERIFICATION_PROMPT,
)
from tools import vector_search
import re

def classify_source_quality(url: str) -> tuple[str, str]:
    """
    Classifies evidence source quality based on URL heuristics.
    Returns (quality_level, explanation).
    """

    u = url.lower()

    # High-quality academic sources
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

    # Medium-quality institutional sources
    if any(domain in u for domain in [
        ".edu",
        ".gov",
        "who.int",
        "nih.gov",
        "nasa.gov",
    ]):
        return "MEDIUM", "Institutional or government source"

    # Low-quality sources
    return "LOW", "Non-peer-reviewed or informal source"

import re

def latex_to_plain_math(text: str) -> str:
    """
    Converts common LaTeX math into readable plain-text math
    that renders correctly in Chainlit.
    """

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
    """
    - Converts LaTeX to plain math
    - Moves equations into code blocks
    - Fixes bullet formatting
    """

    text = latex_to_plain_math(text)

    # Lift equations into fenced code blocks
    
    text = latex_to_plain_math(text)

    # Fix bullet rendering (Chainlit markdown rule)
    text = re.sub(r"\n([\-•*])", r"\n\n\1", text)

    return text.strip()

# =========================
# CHAT AGENT
# =========================

def chat_agent(context: str, query: str) -> str:
    """
    Normal chatbot mode.
    Uses full document as context.
    Single LLM call.
    """
    llm = get_llm()

    prompt = CHAT_PROMPT.format(
        context=context,
        question=query,
    )

    response = llm.invoke(prompt)
    return normalize_output_for_chainlit(response.content.strip())


# =========================
# SUMMARIZE AGENT
# =========================
def summarize_agent(text: str) -> str:
    llm = get_llm()
    prompt = SUMMARY_PROMPT.format(text=text)
    raw = llm.invoke(prompt).content.strip()

    # ✅ MINIMAL UI FIX
    return normalize_output_for_chainlit(raw)

# =========================
# CLAIM EXTRACTION (FULL DOCUMENT)
# =========================
import json

def extract_claims_agent(text: str, max_claims: int = 5) -> List[str]:
    llm = get_llm()

    prompt = CLAIM_EXTRACTION_PROMPT.format(
        text=text,
        max_claims=max_claims,
    )

    response = llm.invoke(prompt).content.strip()

    try:
        data = json.loads(response)
        claims = data.get("claims", [])
    except Exception as e:
        raise ValueError(
            f"Claim extraction failed. Expected JSON but got:\n{response}"
        )

    if not isinstance(claims, list):
        raise ValueError("Claims must be a list of strings.")

    return claims[:max_claims]


# =========================
# EXTERNAL VERIFICATION AGENT (NEW)
# =========================

def verify_claims_agent(claims: List[str]) -> List[dict]:
    """
    External-only verification using Perplexity search.
    Returns structured JSON per claim.
    """
    llm = get_llm()
    results = []

    for claim in claims:
        prompt = EXTERNAL_VERIFICATION_PROMPT.format(claim=claim)

        response = llm.invoke(prompt)

        # Perplexity returns citations in additional_kwargs
        citations = response.additional_kwargs.get("citations", [])
        raw = response.content.strip().upper()

        if "SUPPORTED" in raw:
            verdict = "SUPPORTED"
        elif "CONTRADICTED" in raw:
            verdict = "CONTRADICTED"
        else:
            verdict = "INSUFFICIENT"
        clean_sources = []

        for c in citations:
            if isinstance(c, str):
                clean_sources.append(c)
            elif isinstance(c, dict) and "url" in c:
                clean_sources.append(c["url"])
        enriched_sources = []

        for src in clean_sources:
            quality, reason = classify_source_quality(src)
            enriched_sources.append({
                "url": src,
                "quality": quality,
                "reason": reason,
            })

        results.append({
            "claim": claim,
            "sources": enriched_sources,
            "verdict": verdict
        })


    formatted = []

    for item in results:
        block = f"""
    ### Claim
    {item['claim']}

    ### Verdict
    **{item['verdict']}**

    ### Sources
    """
        if item["sources"]:
            for src in item["sources"]:
                block += f"- {src}\n"
        else:
            block += "- No external sources found\n"

        formatted.append(block.strip())

    return results