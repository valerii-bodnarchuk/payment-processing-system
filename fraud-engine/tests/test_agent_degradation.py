"""
Tests for degraded-verdict marking.

A degraded verdict is syntactically valid but was never reasoned to. These tests
pin the marker as structured state — a consumer must be able to read `degraded`
and `degradation_reason` directly, without inspecting confidence or summary text.

Graph-level, LLM mocked. No network, no database.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from agent.config import MAX_ITERATIONS
from agent.nodes import (
    DEGRADATION_ITERATIONS_EXHAUSTED,
    DEGRADATION_LLM_UNAVAILABLE,
    DEGRADATION_OUTPUT_UNPARSEABLE,
    _fallback_verdict,
    is_transient_degradation,
    reason_node,
    synthesize_node,
)

GOOD_VERDICT = {
    "verdict": "TRUE_POSITIVE",
    "confidence": 0.9,
    "risk_level": "high",
    "summary": "Velocity spike corroborated by dispute history.",
    "key_findings": ["5 payouts in 24h"],
    "evidence": [],
    "recommended_actions": ["Hold payout"],
}


def _llm_returning(content: str) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


def _llm_raising(exc: Exception) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=exc)
    return llm


def _base_state(**overrides) -> dict:
    state = {
        "transaction_id": 42,
        "trigger": "BLOCK",
        "messages": [SystemMessage(content="ctx")],
        "iteration": 1,
        "audit_trail": [],
    }
    state.update(overrides)
    return state


# ── Transient: the provider call failed ──────────────────────────

async def test_synthesis_llm_failure_marks_transient_degradation():
    llm = _llm_raising(ConnectionError("provider unreachable"))

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(_base_state())

    assert result["degraded"] is True
    assert result["degradation_reason"] == DEGRADATION_LLM_UNAVAILABLE
    assert is_transient_degradation(result["degradation_reason"])

    # The verdict payload carries the marker too, so it survives being read
    # in isolation (e.g. straight out of InvestigationRun.verdictPayload).
    assert result["verdict"]["degraded"] is True
    assert result["verdict"]["degradation_reason"] == DEGRADATION_LLM_UNAVAILABLE


async def test_reasoning_llm_failure_marks_degradation():
    """reason_node forces INVESTIGATION_COMPLETE — that must not read as clean."""
    llm = AsyncMock()
    llm.bind_tools = lambda tools: _llm_raising(TimeoutError("llm timed out"))

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await reason_node(_base_state())

    assert result["degraded"] is True
    assert result["degradation_reason"] == DEGRADATION_LLM_UNAVAILABLE


async def test_degradation_survives_a_successful_synthesis():
    """A good synthesis on partial context does not undo a failed reasoning leg."""
    llm = _llm_returning(json.dumps(GOOD_VERDICT))

    inherited = _base_state(
        degraded=True,
        degradation_reason=DEGRADATION_LLM_UNAVAILABLE,
    )
    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(inherited)

    # Synthesis succeeded, so the verdict itself is a real one …
    assert result["verdict"]["verdict"] == "TRUE_POSITIVE"
    # … but the run is still degraded, and keeps the earlier proximate cause.
    assert result["degraded"] is True
    assert result["degradation_reason"] == DEGRADATION_LLM_UNAVAILABLE
    assert result["verdict"]["degraded"] is True


# ── Non-transient: the loop ran out of iterations ────────────────

async def test_iterations_exhausted_marks_non_transient_degradation():
    """At the cap the router forces synthesis — the model never concluded."""
    llm = _llm_returning(json.dumps(GOOD_VERDICT))

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(_base_state(iteration=MAX_ITERATIONS))

    assert result["degraded"] is True
    assert result["degradation_reason"] == DEGRADATION_ITERATIONS_EXHAUSTED
    assert not is_transient_degradation(result["degradation_reason"])
    assert result["verdict"]["degraded"] is True
    assert result["verdict"]["degradation_reason"] == DEGRADATION_ITERATIONS_EXHAUSTED


async def test_unparseable_output_marks_non_transient_degradation():
    llm = _llm_returning("I think this is fraud, honestly.")

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(_base_state())

    assert result["degraded"] is True
    assert result["degradation_reason"] == DEGRADATION_OUTPUT_UNPARSEABLE
    assert not is_transient_degradation(result["degradation_reason"])


# ── The clean path stays clean ───────────────────────────────────

async def test_successful_run_is_not_marked_degraded():
    llm = _llm_returning(json.dumps(GOOD_VERDICT))

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(_base_state(iteration=2))

    assert result["degraded"] is False
    assert result["degradation_reason"] is None
    # No marker smuggled into the verdict payload either.
    assert "degraded" not in result["verdict"]
    assert result["verdict"]["verdict"] == "TRUE_POSITIVE"


# ── The marker is structured, not inferred ───────────────────────

def test_fallback_verdict_carries_a_structured_marker():
    verdict = _fallback_verdict("boom", DEGRADATION_LLM_UNAVAILABLE)

    assert verdict["degraded"] is True
    assert verdict["degradation_reason"] == DEGRADATION_LLM_UNAVAILABLE
    # The reason is a field, not something to be parsed back out of prose.
    assert DEGRADATION_LLM_UNAVAILABLE not in verdict["summary"]


def test_transient_and_non_transient_reasons_are_distinguishable():
    assert is_transient_degradation(DEGRADATION_LLM_UNAVAILABLE)
    assert not is_transient_degradation(DEGRADATION_ITERATIONS_EXHAUSTED)
    assert not is_transient_degradation(DEGRADATION_OUTPUT_UNPARSEABLE)
    assert not is_transient_degradation(None)


# ── Through the compiled graph ───────────────────────────────────

async def test_marker_reaches_final_state_through_the_graph():
    """End-to-end through the compiled graph: the marker must survive audit_node.

    Node-level tests above prove each site sets it; this proves nothing between
    synthesis and END drops it on the way out.
    """
    from agent.graph import build_investigation_graph

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=ConnectionError("provider unreachable"))
    llm.bind_tools = lambda tools: llm

    async def fake_get(path, *args, **kwargs):
        return {"transactionId": 42, "payoutReports": [], "hasPayouts": False}

    with (
        patch("agent.nodes._get_llm", return_value=llm),
        patch("agent.nodes.nestjs_get", side_effect=fake_get),
        patch(
            "agent.nodes.persist_investigation_run",
            AsyncMock(return_value={"persisted": False}),
        ),
    ):
        state = await build_investigation_graph().ainvoke(
            {"transaction_id": 42, "trigger": "BLOCK"}
        )

    assert state["degraded"] is True
    assert state["degradation_reason"] == DEGRADATION_LLM_UNAVAILABLE
    assert state["verdict"]["degraded"] is True


def test_confidence_does_not_identify_degradation():
    """Guards the hard rule: consumers must not infer degradation from confidence.

    A degraded fallback and a low-confidence genuine verdict are allowed to
    share a confidence value; only the marker separates them.
    """
    degraded = _fallback_verdict("boom", DEGRADATION_LLM_UNAVAILABLE)
    genuine_low_confidence = {**GOOD_VERDICT, "confidence": degraded["confidence"]}

    assert degraded["confidence"] == genuine_low_confidence["confidence"]
    assert degraded["degraded"] is True
    assert "degraded" not in genuine_low_confidence
