"""
Graph nodes — each function is a node in the LangGraph state machine.

Nodes:
- start_node: validates input, initializes state
- collect_node: parallel data fetch (transaction + seller + timeline)
- reason_node: LLM ReAct step — analyze data, decide next action
- synthesize_node: LLM produces structured verdict
- audit_node: persists the audit trail
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.config import MAX_ITERATIONS, OPENAI_MODEL
from agent.prompts import REACT_SYSTEM_PROMPT, SYNTHESIS_PROMPT
from agent.persistence.audit import persist_investigation_run
from agent.state import InvestigationState
from agent.tools.nestjs_client import nestjs_get
from agent.tools.registry import ALL_TOOLS

logger = logging.getLogger("agent.nodes")


def _get_llm():
    """Lazy LLM init — avoids import-time API key requirement for tests."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


def _fallback_verdict(reason: str) -> dict:
    """Safe manual-review verdict when the LLM cannot produce a conclusion."""
    return {
        "verdict": "INCONCLUSIVE",
        "confidence": 0.1,
        "risk_level": "medium",
        "summary": f"Agent could not complete LLM reasoning: {reason}",
        "key_findings": ["LLM investigation step failed."],
        "evidence": [
            {
                "source": "agent_runtime",
                "fact": reason,
                "significance": "A human investigator must review the collected deterministic context.",
            }
        ],
        "recommended_actions": ["Manual review required — LLM investigation unavailable."],
    }


# ── Start ────────────────────────────────────────────────────────

async def start_node(state: InvestigationState) -> dict:
    """Validate input and initialize tracking fields."""
    tx_id = state["transaction_id"]
    trigger = state.get("trigger", "MANUAL")

    logger.info(f"Starting investigation for transaction {tx_id} (trigger: {trigger})")

    return {
        "iteration": 0,
        "audit_trail": [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "investigation_started",
                "transaction_id": tx_id,
                "trigger": trigger,
            }
        ],
        "verdict": None,
        "transaction_data": None,
        "seller_profile": None,
        "payout_timeline": None,
        "fraud_score_detail": None,
        "ledger_check": None,
        "similar_cases": None,
    }


# ── Collect ──────────────────────────────────────────────────────

async def collect_node(state: InvestigationState) -> dict:
    """
    Parallel data fetch — get transaction context before the reasoning loop.
    This is deterministic, not LLM-driven. Every investigation needs this data.

    The risk profile is fetched with excludePayoutId set to the payout under
    investigation. That is NOT an optimisation — it is required for the numbers
    the LLM goes on to forward to the fraud engine to be correct.

    The engine's daily_volume rule computes `seller_total_amount_24h + amount`,
    i.e. it adds the payout being scored back in itself. Feed it a window total
    that already contains that payout and it is counted twice, which inflates
    the recomputed score above the stored one and makes a false positive look
    corroborated.

    This node is where the exclusion has to happen. `get_seller_risk_profile`
    grew an `exclude_payout_id` parameter for the same reason, but the agent
    never calls that tool — the profile is already in its first message, so
    there is nothing left to fetch. Fixing the tool alone left the live path
    unfixed; only repeated eval runs made that visible.
    """
    import asyncio

    tx_id = state["transaction_id"]

    # Fetch transaction context (includes payouts, entries, disputes)
    tx_data = await nestjs_get(f"/investigate/transaction/{tx_id}")

    # Extract seller_id and the payout under investigation from the first
    # report. One transaction carries one payout in every current flow; if that
    # ever changes, the first report is also what seller_id is taken from, so
    # the two stay consistent by construction.
    seller_id = None
    payout_id = None
    if tx_data and not tx_data.get("error") and tx_data.get("payoutReports"):
        seller_id = tx_data["payoutReports"][0].get("sellerId")
        payout_id = tx_data["payoutReports"][0].get("payoutId")
    elif tx_data and not tx_data.get("error") and tx_data.get("hasPayouts") is False:
        pass  # No payouts — seller_id stays None

    # Parallel fetch seller profile + timeline if we have a seller
    seller_profile = None
    payout_timeline = None
    if seller_id:
        # No payout_id (a transaction with no payout report) means nothing to
        # exclude — the unfiltered total is then already the prior-window total.
        profile_params = {"excludePayoutId": payout_id} if payout_id else None
        seller_profile, payout_timeline = await asyncio.gather(
            nestjs_get(f"/admin/sellers/{seller_id}/risk-profile", params=profile_params),
            nestjs_get(f"/admin/sellers/{seller_id}/payout-timeline"),
        )

    # Build initial context message for the LLM
    context_parts = [f"## Investigation: Transaction #{tx_id}\n"]
    context_parts.append(f"**Trigger:** {state.get('trigger', 'MANUAL')}\n")

    if tx_data and not tx_data.get("error"):
        context_parts.append(f"**Transaction data:**\n```json\n{json.dumps(tx_data, indent=2, default=str)[:3000]}\n```\n")
    else:
        context_parts.append(f"**Transaction data:** ERROR — {tx_data}\n")

    if seller_profile and not seller_profile.get("error"):
        context_parts.append(f"**Seller risk profile:**\n```json\n{json.dumps(seller_profile, indent=2, default=str)[:2000]}\n```\n")
        if payout_id:
            # Stated next to the data, not just in the system prompt: this is the
            # figure that gets forwarded to the engine, and the engine adds the
            # payout's amount back in itself.
            context_parts.append(
                f"Note: `totalVolume24h` above EXCLUDES payout #{payout_id}, the one "
                f"under investigation. Pass it to get_fraud_score_explanation as-is.\n"
            )

    if payout_timeline and not payout_timeline.get("error"):
        context_parts.append(f"**Payout timeline:**\n```json\n{json.dumps(payout_timeline, indent=2, default=str)[:2000]}\n```\n")

    context_parts.append(
        "\nAnalyze this data. If you need more information, call a tool. "
        "If you have enough to form a verdict, respond with INVESTIGATION_COMPLETE."
    )

    context_message = HumanMessage(content="\n".join(context_parts))

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "context_collected",
        "transaction_data_loaded": tx_data is not None and not tx_data.get("error"),
        "seller_profile_loaded": seller_profile is not None and not (seller_profile or {}).get("error"),
        "payout_timeline_loaded": payout_timeline is not None and not (payout_timeline or {}).get("error"),
    }

    return {
        "transaction_data": tx_data,
        "seller_profile": seller_profile,
        "payout_timeline": payout_timeline,
        "messages": [
            SystemMessage(content=REACT_SYSTEM_PROMPT.format(max_iterations=MAX_ITERATIONS)),
            context_message,
        ],
        "audit_trail": state.get("audit_trail", []) + [audit_entry],
    }


# ── Reason (ReAct) ───────────────────────────────────────────────

async def reason_node(state: InvestigationState) -> dict:
    """
    LLM reasoning step. The model either:
    1. Calls a tool → routed to tool execution → loops back here
    2. Says INVESTIGATION_COMPLETE → routed to synthesize
    3. Hits iteration cap → forced to synthesize
    """
    new_iteration = state.get("iteration", 0) + 1
    try:
        llm = _get_llm().bind_tools(ALL_TOOLS)
        response = await llm.ainvoke(state["messages"])
    except Exception as e:
        detail = str(e)
        logger.exception("LLM reasoning failed")
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "llm_error",
            "stage": "reason",
            "iteration": new_iteration,
            "detail": detail[:500],
        }

        return {
            "messages": [AIMessage(content="INVESTIGATION_COMPLETE")],
            "iteration": new_iteration,
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
        }

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "llm_reasoning",
        "iteration": new_iteration,
        "has_tool_calls": bool(response.tool_calls),
        "content_preview": (response.content or "")[:200],
    }

    return {
        "messages": [response],
        "iteration": new_iteration,
        "audit_trail": state.get("audit_trail", []) + [audit_entry],
    }


# ── Synthesize ───────────────────────────────────────────────────

async def synthesize_node(state: InvestigationState) -> dict:
    """
    Final verdict generation. Uses a separate prompt that forces
    structured JSON output. Does NOT have tool access.
    """
    llm = _get_llm()

    # Tool messages already include similar-case retrieval output when the LLM
    # called find_similar_cases. Keep them in context as advisory evidence.
    synthesis_messages = [
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=(
            "Here is the full investigation context from the reasoning steps:\n\n"
            + "\n".join(
                msg.content or ""
                for msg in state.get("messages", [])
                if hasattr(msg, "content") and msg.content
            )[-6000:]  # trim to last ~6k chars to fit context
        )),
    ]

    try:
        response = await llm.ainvoke(synthesis_messages)
    except Exception as e:
        detail = str(e)
        logger.exception("LLM synthesis failed")
        verdict = _fallback_verdict(detail)
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "llm_error",
            "stage": "synthesize",
            "detail": detail[:500],
        }
        verdict_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "verdict_produced",
            "verdict": verdict.get("verdict"),
            "confidence": verdict.get("confidence"),
            "risk_level": verdict.get("risk_level"),
        }

        return {
            "verdict": verdict,
            "audit_trail": state.get("audit_trail", []) + [audit_entry, verdict_entry],
        }

    # Parse the JSON verdict
    verdict = None
    try:
        raw = response.content.strip()
        # Strip markdown fences if the LLM wraps them anyway
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        verdict = json.loads(raw.strip())
    except (json.JSONDecodeError, IndexError):
        logger.error(f"Failed to parse verdict JSON: {response.content[:500]}")
        verdict = {
            "verdict": "INCONCLUSIVE",
            "confidence": 0.1,
            "risk_level": "medium",
            "summary": "Agent failed to produce structured output. Raw response available in audit trail.",
            "key_findings": [],
            "evidence": [],
            "recommended_actions": ["Manual review required — agent output parsing failed."],
        }

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "verdict_produced",
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "risk_level": verdict.get("risk_level"),
    }

    return {
        "verdict": verdict,
        "audit_trail": state.get("audit_trail", []) + [audit_entry],
    }


# ── Audit ────────────────────────────────────────────────────────

async def audit_node(state: InvestigationState) -> dict:
    """
    Persist audit trail. For v1: log to stdout (structured JSON).
    Production: append to PostgreSQL audit table or append-only file.
    """
    trail = state.get("audit_trail", [])

    final_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "investigation_complete",
        "transaction_id": state["transaction_id"],
        "verdict": state.get("verdict", {}).get("verdict"),
        "total_iterations": state.get("iteration", 0),
        "total_audit_entries": len(trail) + 1,
    }

    full_trail = trail + [final_entry]
    state_for_persistence = dict(state)
    state_for_persistence["audit_trail"] = full_trail
    persist_result = await persist_investigation_run(state_for_persistence, full_trail)

    logger.info(
        json.dumps({
            "audit_trail": full_trail,
            "transaction_id": state["transaction_id"],
            "persistence": persist_result,
        }, default=str)
    )

    # run_id is None when persistence was skipped or failed — callers must
    # tolerate its absence rather than treat it as a broken run.
    return {"audit_trail": full_trail, "run_id": persist_result.get("run_id")}
