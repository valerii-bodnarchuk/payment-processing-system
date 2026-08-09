"""
Unit tests for agent/mcp_server.py.

The graph is mocked throughout — no live LLM, no database, no network.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from agent import mcp_server
from agent.mcp_server import (
    _compute_run_key,
    investigate_transaction,
    investigation_audit,
)


_DEFAULT_VERDICT = {
    "verdict": "TRUE_POSITIVE",
    "confidence": 0.88,
    "risk_level": "high",
    "summary": "Payout velocity spike with two open disputes.",
}

# Sentinel so tests can pass verdict=None explicitly to mean "no verdict".
_UNSET = object()


def _state(verdict=_UNSET, run_id: int | None = 77) -> dict:
    """A minimal graph output state."""
    return {
        "verdict": _DEFAULT_VERDICT if verdict is _UNSET else verdict,
        "audit_trail": [
            {"timestamp": "2026-08-09T10:00:00+00:00", "action": "investigation_start"},
            {"timestamp": "2026-08-09T10:00:04+00:00", "action": "context_collected"},
            {"timestamp": "2026-08-09T10:00:09+00:00", "action": "investigation_complete"},
        ],
        "iteration": 3,
        "run_id": run_id,
    }


@pytest.fixture(autouse=True)
def clear_module_state():
    """The idempotency caches are module-level — reset them between tests."""
    mcp_server._COMPLETED.clear()
    mcp_server._AUDIT_TRAILS.clear()
    mcp_server._IN_FLIGHT.clear()
    yield
    mcp_server._COMPLETED.clear()
    mcp_server._AUDIT_TRAILS.clear()
    mcp_server._IN_FLIGHT.clear()


def _mock_graph(return_value=None, side_effect=None) -> AsyncMock:
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(return_value=return_value, side_effect=side_effect)
    return graph


# ── Registration ─────────────────────────────────────────────────

async def test_tool_and_resource_template_are_registered():
    tools = await mcp_server.mcp.list_tools()
    assert [t.name for t in tools] == ["investigate_transaction"]

    templates = await mcp_server.mcp.list_resource_templates()
    assert "investigation://{run_key}/audit" in [t.uri_template for t in templates]


# ── Happy path ───────────────────────────────────────────────────

async def test_successful_call_returns_compact_result():
    graph = _mock_graph(return_value=_state())

    with patch("agent.graph.investigation_graph", graph):
        result = await investigate_transaction(transaction_id=42, trigger="BLOCK")

    assert result.status == "ok"
    assert result.transaction_id == 42
    assert result.verdict == "TRUE_POSITIVE"
    assert result.risk_level == "high"
    assert result.confidence == 0.88
    assert result.summary == "Payout velocity spike with two open disputes."
    assert result.iterations_used == 3
    assert result.run_id == 77
    assert result.replayed is False
    assert result.error is None

    assert result.audit_uri == f"investigation://{result.run_key}/audit"

    # Compact: the trail itself must not ride along in the tool response.
    assert "audit_trail" not in result.model_dump()

    graph.ainvoke.assert_awaited_once_with({"transaction_id": 42, "trigger": "BLOCK"})


async def test_missing_run_id_is_tolerated():
    """Persistence is best-effort — a None run_id is not an error."""
    graph = _mock_graph(return_value=_state(run_id=None))

    with patch("agent.graph.investigation_graph", graph):
        result = await investigate_transaction(transaction_id=42)

    assert result.status == "ok"
    assert result.run_id is None


# ── Idempotency ──────────────────────────────────────────────────

async def test_repeat_call_with_same_key_replays_without_reinvoking():
    graph = _mock_graph(return_value=_state())

    with patch("agent.graph.investigation_graph", graph):
        first = await investigate_transaction(
            transaction_id=42, idempotency_key="retry-abc"
        )
        second = await investigate_transaction(
            transaction_id=42, idempotency_key="retry-abc"
        )

    assert graph.ainvoke.await_count == 1
    assert first.replayed is False
    assert second.replayed is True
    assert second.run_key == first.run_key
    assert second.verdict == first.verdict
    assert second.audit_uri == first.audit_uri


async def test_naive_retry_without_key_also_dedupes():
    """No key supplied — (transaction_id, trigger) still keys the run."""
    graph = _mock_graph(return_value=_state())

    with patch("agent.graph.investigation_graph", graph):
        first = await investigate_transaction(transaction_id=42, trigger="REVIEW")
        second = await investigate_transaction(transaction_id=42, trigger="REVIEW")

    assert graph.ainvoke.await_count == 1
    assert second.replayed is True
    assert second.run_key == first.run_key


async def test_different_keys_run_separate_investigations():
    graph = _mock_graph(return_value=_state())

    with patch("agent.graph.investigation_graph", graph):
        first = await investigate_transaction(transaction_id=42, idempotency_key="a")
        second = await investigate_transaction(transaction_id=42, idempotency_key="b")

    assert graph.ainvoke.await_count == 2
    assert first.run_key != second.run_key
    assert second.replayed is False


async def test_concurrent_identical_calls_invoke_graph_once():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow_ainvoke(payload):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return _state()

    graph = AsyncMock()
    graph.ainvoke = slow_ainvoke

    with patch("agent.graph.investigation_graph", graph):
        first = asyncio.create_task(
            investigate_transaction(transaction_id=42, idempotency_key="race")
        )
        # Let the first call reach the graph and register itself as in-flight.
        await started.wait()

        second = asyncio.create_task(
            investigate_transaction(transaction_id=42, idempotency_key="race")
        )
        await asyncio.sleep(0)  # give the second call a chance to join

        release.set()
        first_result, second_result = await asyncio.gather(first, second)

    assert calls == 1
    assert first_result.status == "ok"
    assert second_result.status == "ok"
    assert second_result.run_key == first_result.run_key
    assert second_result.replayed is True
    assert not mcp_server._IN_FLIGHT  # cleaned up in finally


# ── Error handling ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "exc,expected_retryable",
    [
        (ConnectionError("nestjs unreachable"), True),
        (TimeoutError("llm call timed out"), True),
        (ValueError("bad tool payload"), False),
        (RuntimeError("graph exploded"), False),
    ],
)
async def test_graph_exception_becomes_structured_error(exc, expected_retryable):
    graph = _mock_graph(side_effect=exc)

    with patch("agent.graph.investigation_graph", graph):
        result = await investigate_transaction(transaction_id=42)

    assert result.status == "error"
    assert result.retryable is expected_retryable
    assert result.transaction_id == 42
    assert result.verdict is None
    assert result.audit_uri is None

    # One line, no stack trace leaking into the host's context.
    assert result.error
    assert "\n" not in result.error
    assert "Traceback" not in result.error
    assert "File \"" not in result.error


async def test_missing_verdict_is_a_retryable_error():
    graph = _mock_graph(return_value=_state(verdict=None))

    with patch("agent.graph.investigation_graph", graph):
        result = await investigate_transaction(transaction_id=42)

    assert result.status == "error"
    assert result.retryable is True
    assert "verdict" in result.error.lower()


async def test_failed_run_is_not_cached_so_retry_reruns():
    graph = _mock_graph(side_effect=ConnectionError("nestjs unreachable"))

    with patch("agent.graph.investigation_graph", graph):
        failed = await investigate_transaction(
            transaction_id=42, idempotency_key="retry-me"
        )
        assert failed.status == "error"

    graph_ok = _mock_graph(return_value=_state())
    with patch("agent.graph.investigation_graph", graph_ok):
        retried = await investigate_transaction(
            transaction_id=42, idempotency_key="retry-me"
        )

    assert graph_ok.ainvoke.await_count == 1
    assert retried.status == "ok"
    assert retried.replayed is False


# ── Audit resource ───────────────────────────────────────────────

async def test_audit_resource_returns_trail_for_known_key():
    graph = _mock_graph(return_value=_state())

    with patch("agent.graph.investigation_graph", graph):
        result = await investigate_transaction(transaction_id=42)

    audit = await investigation_audit(result.run_key)

    assert audit["status"] == "ok"
    assert audit["run_key"] == result.run_key
    assert audit["entry_count"] == 3
    assert [e["action"] for e in audit["audit_trail"]] == [
        "investigation_start",
        "context_collected",
        "investigation_complete",
    ]


async def test_audit_resource_unknown_key_returns_structured_error():
    audit = await investigation_audit("0000000000000000")

    assert audit["status"] == "error"
    assert audit["audit_trail"] == []
    assert "Unknown run_key" in audit["error"]


# ── Run key ──────────────────────────────────────────────────────

def test_run_key_is_stable_and_short():
    key = _compute_run_key(42, "MANUAL", "abc")
    assert key == _compute_run_key(42, "MANUAL", "abc")
    assert len(key) == 16
    # The explicit key wins over the trigger fallback.
    assert key == _compute_run_key(42, "BLOCK", "abc")
    assert key != _compute_run_key(43, "MANUAL", "abc")


# ── Protocol round-trip ──────────────────────────────────────────

async def test_round_trip_over_mcp_protocol():
    """Exercise the server through an in-memory MCP client.

    The tests above call the tool as a plain function, which skips schema
    validation, result serialisation and resource-URI routing. This one goes
    through the protocol, so it covers the parts a host actually touches.
    """
    graph = _mock_graph(return_value=_state())

    with patch("agent.graph.investigation_graph", graph):
        async with Client(mcp_server.mcp) as client:
            first = await client.call_tool(
                "investigate_transaction",
                {
                    "transaction_id": 42,
                    "trigger": "BLOCK",
                    "idempotency_key": "round-trip",
                },
            )

            # Structured output: deserialised model plus the raw JSON payload.
            assert first.data.status == "ok"
            assert first.data.verdict == "TRUE_POSITIVE"
            assert first.data.risk_level == "high"
            assert first.data.run_id == 77
            assert first.data.replayed is False
            assert first.structured_content["run_key"] == first.data.run_key
            assert "audit_trail" not in first.structured_content

            # The trail is reachable only via the URI the tool handed back.
            audit_uri = first.data.audit_uri
            assert audit_uri == f"investigation://{first.data.run_key}/audit"

            contents = await client.read_resource(audit_uri)
            audit = json.loads(contents[0].text)
            assert audit["status"] == "ok"
            assert audit["run_key"] == first.data.run_key
            assert [e["action"] for e in audit["audit_trail"]] == [
                "investigation_start",
                "context_collected",
                "investigation_complete",
            ]

            # Same key again: replayed, and the graph was never re-entered.
            second = await client.call_tool(
                "investigate_transaction",
                {
                    "transaction_id": 42,
                    "trigger": "BLOCK",
                    "idempotency_key": "round-trip",
                },
            )
            assert second.data.replayed is True
            assert second.data.run_key == first.data.run_key

    assert graph.ainvoke.await_count == 1
