from dataclasses import dataclass, field
from typing import List

@dataclass
class Claim:
    text: str
    verdict: str
    evidence: str

@dataclass
class GraphState:
    raw_text: str

    audit_report: str = ""
    draft_summary: str = ""

    claims: List[Claim] = field(default_factory=list)

    hallucinations_found: bool = False
    critique_notes: str = ""

    retry_count: int = 0
    max_retries: int = 2

    confidence: str = ""
    final_answer: str = ""
