"""
eval/guard.py — contamination guard for the eval harness.

The eval marker (EVAL::<hash>) is written only into columns the agent does NOT
serialise to the LLM (see eval/seed.py). This guard is the runtime proof of that
invariant: after the graph runs, it scans everything the LLM actually saw and
fails if the marker is anywhere in it.

The runner is expected to call assert_no_contamination(result, case_id=...)
after investigation_graph.ainvoke(...) and BEFORE scoring. (Runner integration
is intentionally not wired here.)
"""
from __future__ import annotations

from typing import Iterable

# Must match EVAL_MARKER_PREFIX in eval/seed.py.
EVAL_MARKER_PREFIX = "EVAL::"

# State keys that hold raw tool output (i.e. data that was fed back to the LLM).
# From agent/state.py — the "collected context" block.
_CONTEXT_KEYS = (
    "transaction_data",
    "seller_profile",
    "payout_timeline",
    "fraud_score_detail",
    "ledger_check",
    "similar_cases",
)


def assert_no_contamination(state: dict, case_id: str | None = None) -> None:
    """Raise ValueError if the eval marker reached any LLM-visible data.

    Scans state["messages"] (message content + tool-call args — where tool
    OUTPUTS land as ToolMessages in the ReAct loop) and the collected-context
    fields that hold raw tool JSON.
    """
    hits = sorted(_scan(state))
    if hits:
        loc = f" (case {case_id})" if case_id else ""
        raise ValueError(
            f"Contamination detected{loc}: eval marker {EVAL_MARKER_PREFIX!r} "
            f"reached LLM-visible data at: {', '.join(hits)}. "
            "A marker leaked into an exposed column — the golden fixture is "
            "invalid and any accuracy from this run is fake."
        )


def _scan(state: dict) -> set[str]:
    """Return the set of locations where the marker appears (empty if clean)."""
    hits: set[str] = set()

    for i, msg in enumerate(state.get("messages", []) or []):
        if _message_contaminated(msg):
            hits.add(f"messages[{i}]")

    for key in _CONTEXT_KEYS:
        if key in state and _contains_marker(state[key]):
            hits.add(f"state.{key}")

    return hits


def _message_contaminated(msg: object) -> bool:
    """Check a message across the forms the ReAct loop produces: LangChain
    BaseMessage objects (content / tool_calls / additional_kwargs / name) and
    plain dicts."""
    if isinstance(msg, dict):
        return _contains_marker(msg)

    for attr in ("content", "tool_calls", "additional_kwargs", "name"):
        if _contains_marker(getattr(msg, attr, None)):
            return True
    return False


def _contains_marker(obj: object) -> bool:
    """Recursively test whether EVAL:: appears anywhere in a value."""
    if obj is None:
        return False
    if isinstance(obj, str):
        return EVAL_MARKER_PREFIX in obj
    if isinstance(obj, dict):
        return any(_contains_marker(k) for k in obj.keys()) or any(
            _contains_marker(v) for v in obj.values()
        )
    if isinstance(obj, Iterable) and not isinstance(obj, (bytes, bytearray)):
        return any(_contains_marker(v) for v in obj)
    return EVAL_MARKER_PREFIX in str(obj)
