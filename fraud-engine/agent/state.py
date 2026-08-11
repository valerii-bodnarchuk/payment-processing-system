"""
Investigation state schema — the single source of truth flowing through the graph.

Design decisions:
- All fields optional except transaction_id and trigger — graph nodes populate them.
- audit_trail is append-only within a single investigation run.
- messages holds the LangChain message history for the ReAct loop.
- iteration is a safety counter to prevent infinite loops.
"""
from __future__ import annotations

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class InvestigationState(TypedDict, total=False):
    """Typed state for the fraud investigation graph."""

    # ── Input (required) ─────────────────────────────────────────
    transaction_id: int
    trigger: Literal["BLOCK", "REVIEW", "MANUAL"]

    # ── Collected context (tools populate these) ─────────────────
    transaction_data: dict | None
    seller_profile: dict | None
    payout_timeline: dict | None
    fraud_score_detail: dict | None
    ledger_check: dict | None
    similar_cases: dict | None

    # ── ReAct loop ───────────────────────────────────────────────
    # add_messages reducer: appends new messages instead of replacing
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int

    # ── Output ───────────────────────────────────────────────────
    verdict: dict | None
    audit_trail: list[dict]
    # True when the verdict did not come from a complete reasoning pass.
    # degradation_reason carries which failure caused it (see agent.nodes
    # DEGRADATION_*). Structured on purpose: a consumer must never have to
    # infer this from confidence or parse it out of the summary text.
    degraded: bool
    degradation_reason: str | None
    # InvestigationRun.id from audit_node's persistence step. None when
    # persistence is skipped or fails — it is best-effort, never fatal.
    run_id: int | None
