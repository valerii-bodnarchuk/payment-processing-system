"""
LangGraph investigation graph — the core orchestration layer.

Graph topology:
  start → collect → reason ⇄ tool_executor → synthesize → audit → END

The reason ↔ tool_executor loop is the ReAct pattern.
Conditional edge after reason_node routes based on:
1. LLM returned tool calls → execute them
2. LLM said INVESTIGATION_COMPLETE → go to synthesize
3. Iteration cap reached → force synthesize
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.state import InvestigationState
from agent.tools.registry import ALL_TOOLS
from agent.config import MAX_ITERATIONS
from agent.nodes import (
    _message_text,
    start_node,
    collect_node,
    reason_node,
    synthesize_node,
    audit_node,
)


def _route_after_reason(state: InvestigationState) -> str:
    """
    Conditional edge after the reason node. Three possible outcomes:
    1. "tools"      — LLM wants to call a tool
    2. "synthesize" — LLM says it's done (INVESTIGATION_COMPLETE)
    3. "synthesize" — iteration cap reached (safety net)
    """
    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        return "synthesize"

    messages = state.get("messages", [])
    if not messages:
        return "synthesize"

    last_message = messages[-1]

    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Check for completion signal
    # Bedrock returns content as a list of blocks; `in` against a list would
    # test element membership and never find the signal.
    content = _message_text(getattr(last_message, "content", None))
    if "INVESTIGATION_COMPLETE" in content:
        return "synthesize"

    # Default: if LLM responded with text but no tool calls and no completion signal,
    # treat as done (it may have given its analysis inline)
    return "synthesize"


def build_investigation_graph() -> StateGraph:
    """Construct and compile the investigation graph."""

    graph = StateGraph(InvestigationState)

    # ── Nodes ─────────────────────────────────────────────────
    graph.add_node("start", start_node)
    graph.add_node("collect", collect_node)
    graph.add_node("reason", reason_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("audit", audit_node)

    # ── Edges ─────────────────────────────────────────────────
    graph.set_entry_point("start")
    graph.add_edge("start", "collect")
    graph.add_edge("collect", "reason")

    graph.add_conditional_edges(
        "reason",
        _route_after_reason,
        {
            "tools": "tools",
            "synthesize": "synthesize",
        },
    )

    # After tool execution, loop back to reason
    graph.add_edge("tools", "reason")

    graph.add_edge("synthesize", "audit")
    graph.add_edge("audit", END)

    return graph.compile()


# Singleton — compiled once, reused across requests
investigation_graph = build_investigation_graph()
