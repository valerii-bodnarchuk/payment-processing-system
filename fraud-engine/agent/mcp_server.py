"""
MCP server exposing the fraud investigation agent to other agents.

This is deliberately *not* a mirror of `POST /investigate`. That endpoint serves a
human (or a dashboard) and returns the full audit trail in one response. An MCP
host pays context tokens for every byte a tool returns, and it retries on its own
initiative. Two consequences shape this surface:

1. The tool returns a compact verdict only. The audit trail — easily thousands of
   tokens — lives behind `investigation://{run_key}/audit`, fetched only when the
   caller actually needs to justify the verdict.
2. Calls are idempotent by `run_key`. One investigation is several LLM round
   trips; an autonomous retry must not pay for a second one.

Run it:
    python -m agent.mcp_server
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger("agent.mcp_server")

mcp = FastMCP("fraud-investigation-agent")


# ── Idempotency state ────────────────────────────────────────────
#
# Process-local, so it dedupes retries within one server lifetime only.
# The durable version keys off the `InvestigationRun` table with a unique index
# on ("transactionId", "idempotencyKey"): the insert becomes the dedupe lock, and
# a conflict returns the existing run instead of starting a second one. That
# needs a Prisma migration and is out of scope here.
_COMPLETED: dict[str, "InvestigationResult"] = {}
_AUDIT_TRAILS: dict[str, list[dict]] = {}
_IN_FLIGHT: dict[str, asyncio.Task] = {}


class InvestigationResult(BaseModel):
    """Compact investigation result — the audit trail is fetched separately."""

    status: Literal["ok", "error"]
    run_key: str
    run_id: int | None = None
    transaction_id: int
    verdict: str | None = None
    risk_level: str | None = None
    confidence: float | None = None
    summary: str | None = None
    iterations_used: int = 0
    audit_uri: str | None = Field(
        default=None,
        description="MCP resource URI holding the full audit trail for this run.",
    )
    replayed: bool = False
    error: str | None = None
    retryable: bool | None = None


def _compute_run_key(
    transaction_id: int,
    trigger: str,
    idempotency_key: str | None,
) -> str:
    """Stable short key for one logical investigation.

    Without an explicit key we fall back to (transaction_id, trigger) so that a
    naive retry — same arguments, no key — still dedupes instead of re-running.
    """
    basis = (
        f"{transaction_id}:{idempotency_key}"
        if idempotency_key
        else f"{transaction_id}:{trigger}"
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _one_line(text: str, limit: int = 300) -> str:
    """Collapse an exception message to a single loggable/returnable line."""
    return " ".join(str(text).split())[:limit]


def _error_result(
    run_key: str,
    transaction_id: int,
    message: str,
    retryable: bool,
) -> InvestigationResult:
    return InvestigationResult(
        status="error",
        run_key=run_key,
        transaction_id=transaction_id,
        error=_one_line(message),
        retryable=retryable,
    )


async def _run_investigation(
    run_key: str,
    transaction_id: int,
    trigger: str,
) -> InvestigationResult:
    """Invoke the graph once and reduce its state to a compact result.

    Never raises: an MCP host sees a typed error payload it can branch on, not a
    stack trace it has to parse out of an exception string.
    """
    # Lazy import, matching agent/api.py — keeps the server startable without
    # OPENAI_API_KEY and off the LangGraph import cost until first use.
    from agent.graph import investigation_graph

    try:
        state = await investigation_graph.ainvoke({
            "transaction_id": transaction_id,
            "trigger": trigger,
        })
    except (TimeoutError, ConnectionError) as e:
        logger.warning(
            "Investigation transport failure for transaction %s: %s",
            transaction_id,
            _one_line(e),
        )
        return _error_result(
            run_key,
            transaction_id,
            f"{type(e).__name__}: {e}",
            retryable=True,
        )
    except Exception as e:
        logger.exception("Investigation failed for transaction %s", transaction_id)
        return _error_result(
            run_key,
            transaction_id,
            f"{type(e).__name__}: {e}",
            retryable=False,
        )

    verdict = state.get("verdict")
    if not verdict:
        logger.warning("Agent produced no verdict for transaction %s", transaction_id)
        return _error_result(
            run_key,
            transaction_id,
            "Agent produced no verdict.",
            retryable=True,
        )

    trail = state.get("audit_trail") or []
    _AUDIT_TRAILS[run_key] = list(trail)

    result = InvestigationResult(
        status="ok",
        run_key=run_key,
        run_id=state.get("run_id"),
        transaction_id=transaction_id,
        verdict=verdict.get("verdict"),
        risk_level=verdict.get("risk_level"),
        confidence=verdict.get("confidence"),
        summary=verdict.get("summary"),
        iterations_used=state.get("iteration", 0),
        audit_uri=f"investigation://{run_key}/audit",
    )
    # Only successes are cached. A failed run leaves no entry, so a retry with
    # the same key is free to actually retry.
    _COMPLETED[run_key] = result
    return result


@mcp.tool
async def investigate_transaction(
    transaction_id: int,
    trigger: Literal["BLOCK", "REVIEW", "MANUAL"] = "MANUAL",
    idempotency_key: str | None = None,
) -> InvestigationResult:
    """Investigate a flagged transaction and return a compact fraud verdict.

    Runs the full investigation agent: it collects transaction, seller, payout and
    ledger context, reasons over it with tools, and produces a verdict with a
    confidence score and a one-paragraph summary.

    The response is intentionally small. To read the evidence behind a verdict,
    fetch the `audit_uri` resource from the result — do that only when you need to
    justify or contest the verdict, since the trail is large.

    Pass `idempotency_key` when retrying: a repeated call with the same key
    returns the first run's result with `replayed=true` instead of paying for a
    second investigation.
    """
    run_key = _compute_run_key(transaction_id, trigger, idempotency_key)

    cached = _COMPLETED.get(run_key)
    if cached is not None:
        return cached.model_copy(update={"replayed": True})

    # A call arriving while an identical one is still running joins it rather
    # than starting a second investigation. Safe without a lock: everything from
    # here to create_task() runs without an await, so no other coroutine can
    # interleave.
    in_flight = _IN_FLIGHT.get(run_key)
    if in_flight is not None:
        joined = await in_flight
        return joined.model_copy(update={"replayed": True})

    task = asyncio.create_task(_run_investigation(run_key, transaction_id, trigger))
    _IN_FLIGHT[run_key] = task
    try:
        return await task
    finally:
        _IN_FLIGHT.pop(run_key, None)


@mcp.resource("investigation://{run_key}/audit")
async def investigation_audit(run_key: str) -> dict:
    """Full ordered audit trail for one investigation run.

    Every step the agent took, in order: context collected, tools called, and the
    reasoning that produced the verdict. Large — fetch only when the verdict needs
    to be justified or contested.
    """
    trail = _AUDIT_TRAILS.get(run_key)
    if trail is None:
        # Structured miss, not an exception — the host can react to it.
        return {
            "status": "error",
            "run_key": run_key,
            "error": (
                "Unknown run_key. Audit trails are held in memory for the "
                "lifetime of this server process; re-run the investigation."
            ),
            "audit_trail": [],
        }

    return {
        "status": "ok",
        "run_key": run_key,
        "entry_count": len(trail),
        "audit_trail": trail,
    }


if __name__ == "__main__":
    # stdio — the transport Cursor and Claude Desktop launch locally.
    # For a shared/networked deployment: mcp.run(transport="http", port=8765)
    mcp.run()
