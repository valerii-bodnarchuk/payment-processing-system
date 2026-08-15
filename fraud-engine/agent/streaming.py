"""
Server-Sent Events streaming for the investigation graph.

Same graph, same persistence, same verdict shape — just emits one event per
LangGraph node delta while the investigation is running. The audit trail still
lands in PostgreSQL via audit_node at the end; streaming is additive.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from agent.nodes import _message_text

logger = logging.getLogger("agent.streaming")


def _format_sse(event: str, data: dict) -> str:
    """Encode one SSE message. `data` is JSON-serialized with a `default=str`
    fallback so LangChain message objects and datetimes don't blow up the wire."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _summarize_delta(node_name: str, delta: dict | None) -> dict:
    """Project a node's state delta down to what's worth streaming.

    LangGraph emits the full slice of state each node returned; for the wire we
    keep only the last audit entry, a preview of any new messages, and the
    verdict if the synthesize node just produced one. This keeps the SSE frame
    small even when the underlying state carries multi-KB tool outputs.
    """
    delta = delta or {}
    summary: dict = {"node": node_name}

    audit = delta.get("audit_trail")
    if audit:
        summary["last_audit"] = audit[-1]

    messages = delta.get("messages") or []
    if messages:
        summary["messages"] = [
            {
                "type": type(m).__name__,
                "name": getattr(m, "name", None),
                "preview": _message_text(getattr(m, "content", None))[:200],
                "tool_calls": [
                    {"name": tc.get("name"), "args": tc.get("args")}
                    for tc in (getattr(m, "tool_calls", None) or [])
                ],
            }
            for m in messages
        ]

    verdict = delta.get("verdict")
    if verdict:
        summary["verdict"] = {
            "verdict": verdict.get("verdict"),
            "confidence": verdict.get("confidence"),
            "risk_level": verdict.get("risk_level"),
            "summary": verdict.get("summary"),
        }

    return summary


async def stream_investigation(
    transaction_id: int,
    trigger: str = "MANUAL",
) -> AsyncIterator[str]:
    """Run the investigation graph and yield SSE-formatted events.

    Event sequence (happy path):
      event: start    — investigation accepted
      event: node     — one per LangGraph node delta
      event: done     — final verdict + iterations used

    On unhandled exception:
      event: error    — message + transaction_id (stream then closes)
    """
    from agent.graph import investigation_graph

    yield _format_sse("start", {
        "transaction_id": transaction_id,
        "trigger": trigger,
    })

    final_verdict: dict | None = None
    final_iteration = 0

    try:
        async for update in investigation_graph.astream(
            {"transaction_id": transaction_id, "trigger": trigger},
            stream_mode="updates",
        ):
            # `update` is a single-key dict {node_name: delta}; in rare
            # parallel branches it can be multi-key, so iterate either way.
            for node_name, delta in update.items():
                summary = _summarize_delta(node_name, delta)
                yield _format_sse("node", summary)

                if delta:
                    if delta.get("verdict"):
                        final_verdict = delta["verdict"]
                    if "iteration" in delta:
                        final_iteration = delta["iteration"]
    except Exception as e:
        logger.exception("Streaming investigation failed")
        yield _format_sse("error", {
            "transaction_id": transaction_id,
            "detail": str(e)[:500],
        })
        return

    yield _format_sse("done", {
        "transaction_id": transaction_id,
        "verdict": final_verdict,
        "iterations_used": final_iteration,
    })
