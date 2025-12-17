from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

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

        blocks = []
        for item in items:
            block = (
                f"### Claim\n"
                f"{item['claim']}\n\n"
                f"### Verdict\n"
                f"**{item['verdict']}**\n\n"
                f"### Sources\n"
            )

            if item["sources"]:
                for src in item["sources"]:
                    block += f"- {src}\n"
            else:
                block += "- No external sources found\n"

            blocks.append(block.strip())

        return {
            "final_answer": "\n\n---\n\n".join(blocks)
        }


    graph.add_node("extract_claims", extract_node)
    graph.add_node("verify_claims", verify_node)

    graph.set_entry_point("extract_claims")
    graph.add_edge("extract_claims", "verify_claims")
    graph.add_edge("verify_claims", END)

    return graph.compile()
