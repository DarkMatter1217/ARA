from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

def compute_document_trust_score(
    claim_confidences: list[int],
    doc_verdict: str,
    contradicted_count: int
) -> int:
    if not claim_confidences:
        return 0

    base = sum(claim_confidences) / len(claim_confidences)

    # Verdict penalty
    if doc_verdict == "REJECTED":
        verdict_penalty = 30
    elif doc_verdict == "MANUAL_CHECK":
        verdict_penalty = 10
    else:
        verdict_penalty = 0

    # Contradiction penalty
    contradiction_penalty = contradicted_count * 15

    score = base - verdict_penalty - contradiction_penalty

    return max(0, min(100, int(round(score))))

def compute_claim_confidence(verdict: str, evidence_strength: str, source_count: int) -> int:
    # Base score from verdict
    if verdict == "SUPPORTED":
        base = 70
    elif verdict == "INSUFFICIENT":
        base = 40
    else:  # CONTRADICTED
        base = 10

    # Evidence bonus
    if evidence_strength == "HIGH":
        bonus = 20
    elif evidence_strength == "MEDIUM":
        bonus = 10
    elif evidence_strength == "LOW":
        bonus = 5
    else:
        bonus = 0

    # Source count bonus (max 10)
    source_bonus = min(10, source_count * 5)

    return min(100, base + bonus + source_bonus)


def aggregate_document_verdict(items):
    """
    Deterministically aggregates claim-level verdicts
    into a document-level verdict.
    """
    verdicts = [item["verdict"] for item in items]

    if "CONTRADICTED" in verdicts:
        return "REJECTED"

    if "INSUFFICIENT" in verdicts:
        return "MANUAL_CHECK"

    return "ACCEPTED"


from agents import (
    chat_agent,
    summarize_agent,
    extract_claims_agent,
    verify_claims_agent,
)


class GraphState(TypedDict):
    raw_text: str
    query: str
    claims: List[str]
    final_answer: object


# =========================
# CHAT GRAPH
# =========================

def build_chat_graph():
    graph = StateGraph(GraphState)

    def chat_node(state: GraphState):
        return {
            "final_answer": chat_agent(
                context=state["raw_text"],
                query=state["query"],
            )
        }

    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()


# =========================
# SUMMARIZE GRAPH
# =========================

def build_summarize_graph():
    graph = StateGraph(GraphState)

    def summarize_node(state: GraphState):
        return {
            "final_answer": summarize_agent(state["raw_text"])
        }

    graph.add_node("summarize", summarize_node)
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


# =========================
# VERIFY GRAPH (EXTERNAL ONLY)
# =========================

def build_verify_graph():
    graph = StateGraph(GraphState)

    def extract_node(state: GraphState):
        return {
            "claims": extract_claims_agent(state["raw_text"])
        }
    def verify_node(state: GraphState):
        items = verify_claims_agent(state["claims"])

        # --- Document-level aggregation ---
        doc_verdict = aggregate_document_verdict(items)

        # Count stats
        supported = sum(1 for i in items if i["verdict"] == "SUPPORTED")
        insufficient = sum(1 for i in items if i["verdict"] == "INSUFFICIENT")
        contradicted = sum(1 for i in items if i["verdict"] == "CONTRADICTED")

        # --- Document verdict block ---
        header = (
            f"## 📄 Document Verification Result\n"
            f"**{doc_verdict}**\n\n"
            f"**Document Trust Score:** {trust_score} / 100\n\n"
            f"**Summary:**\n"
            f"- Supported claims: {supported}\n"
            f"- Insufficient claims: {insufficient}\n"
            f"- Contradicted claims: {contradicted}\n"
        )

        # --- Claim-level blocks ---
        blocks = []
        claim_confidences = []

        for item in items:
            block = (
                f"### Claim\n"
                f"{item['claim']}\n\n"
                f"### Verdict\n"
                f"**{item['verdict']}**\n\n"
                f"### Sources\n"
            )

            if item["sources"]:
                strengths = []
                for src in item["sources"]:
                    block += (
                        f"- {src['url']} "
                        f"({src['quality']} – {src['reason']})\n"
                    )
                    strengths.append(src["quality"])

                # Evidence strength summary
                if "HIGH" in strengths:
                    strength = "HIGH"
                elif "MEDIUM" in strengths:
                    strength = "MEDIUM"
                else:
                    strength = "LOW"
                confidence = compute_claim_confidence(
                    verdict=item["verdict"],
                    evidence_strength=strength,
                    source_count=len(item["sources"])
                )
                claim_confidences.append(confidence)


                block += f"\n**Evidence Strength:** {strength}\n"
            else:
                block += "- No external sources found\n"
                block += "\n**Evidence Strength:** NONE\n"
                block += f"\n**Claim Confidence:** {confidence}%\n"

            blocks.append(block.strip())
        trust_score = compute_document_trust_score(
            claim_confidences=claim_confidences,
            doc_verdict=doc_verdict,
            contradicted_count=contradicted
        )

        final_output = header + "\n\n---\n\n" + "\n\n---\n\n".join(blocks)

        return {
            "final_answer": final_output
        }

    graph.add_node("extract_claims", extract_node)
    graph.add_node("verify_claims", verify_node)

    graph.set_entry_point("extract_claims")
    graph.add_edge("extract_claims", "verify_claims")
    graph.add_edge("verify_claims", END)

    return graph.compile()
