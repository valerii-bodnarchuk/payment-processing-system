"""
Tests for message-content normalisation across provider shapes.

ChatOpenAI returns `message.content` as a flat string; ChatBedrockConverse
returns a list of content blocks. Every node that reads content must handle
both, and must do so by inspecting the value's shape rather than the configured
provider — the same provider returns either shape depending on the model and on
whether reasoning blocks are enabled.

No live calls to any provider.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from agent.nodes import _message_text, reason_node, synthesize_node

VERDICT = {
    "verdict": "TRUE_POSITIVE",
    "confidence": 0.9,
    "risk_level": "high",
    "summary": "Velocity spike corroborated by dispute history.",
    "key_findings": ["5 payouts in 24h"],
    "evidence": [],
    "recommended_actions": ["Hold payout"],
}

# What ChatBedrockConverse hands back: content blocks, not a flat string.
def _blocks(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def _llm_returning(content) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


def _state(**overrides) -> dict:
    state = {
        "transaction_id": 42,
        "trigger": "BLOCK",
        "messages": [SystemMessage(content="ctx")],
        "iteration": 1,
        "audit_trail": [],
    }
    state.update(overrides)
    return state


# ── The normaliser ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "content,expected",
    [
        ("plain string", "plain string"),                      # ChatOpenAI
        ([{"type": "text", "text": "block text"}], "block text"),  # Bedrock
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        (["bare", " string"], "bare string"),                  # raw string blocks
        ([{"text": "untyped"}], "untyped"),                    # type omitted
        (None, ""),
        ("", ""),
        ([], ""),
    ],
)
def test_normalises_both_shapes(content, expected):
    assert _message_text(content) == expected


def test_non_text_blocks_are_dropped_not_stringified():
    """tool_use and reasoning blocks travel separately and must not leak in."""
    content = [
        {"type": "reasoning_content", "reasoning_content": {"text": "thinking..."}},
        {"type": "text", "text": "the answer"},
        {"type": "tool_use", "id": "call_1", "name": "get_seller_risk_profile",
         "input": {"seller_id": 7}},
    ]

    result = _message_text(content)

    assert result == "the answer"
    assert "thinking" not in result
    assert "get_seller_risk_profile" not in result


def test_result_is_always_a_string():
    """Callers slice and .strip() the result — it can never be a list."""
    for content in ("s", _blocks("s"), None, [], [{"type": "tool_use"}], 42):
        assert isinstance(_message_text(content), str)


# ── Synthesis: the reported crash ────────────────────────────────

async def test_synthesis_parses_a_verdict_from_block_content():
    """Regression: 'sequence item N: expected str instance, list found'."""
    llm = _llm_returning(_blocks(json.dumps(VERDICT)))

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(_state())

    assert result["verdict"]["verdict"] == "TRUE_POSITIVE"
    assert result["verdict"]["confidence"] == 0.9
    assert result["degraded"] is False


async def test_synthesis_parses_a_verdict_from_string_content():
    """The OpenAI shape keeps working unchanged."""
    llm = _llm_returning(json.dumps(VERDICT))

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(_state())

    assert result["verdict"]["verdict"] == "TRUE_POSITIVE"
    assert result["degraded"] is False


async def test_prior_block_content_in_history_does_not_break_synthesis():
    """The crash was in joining message history, not in the final response."""
    history = _state(
        messages=[
            SystemMessage(content="ctx"),
            AIMessage(content="a flat string turn"),
            AIMessage(content=_blocks("a block-shaped turn")),
        ]
    )
    llm = _llm_returning(json.dumps(VERDICT))

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(history)

    assert result["verdict"]["verdict"] == "TRUE_POSITIVE"

    # The block-shaped turn reached the prompt as text, not as repr'd JSON.
    prompt = llm.ainvoke.await_args.args[0][-1].content
    assert "a block-shaped turn" in prompt
    assert "'type': 'text'" not in prompt


async def test_unparseable_block_content_degrades_rather_than_crashing():
    llm = _llm_returning(_blocks("I think this is fraud, honestly."))

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await synthesize_node(_state())

    assert result["degraded"] is True
    assert result["verdict"]["verdict"] == "INCONCLUSIVE"


# ── Reasoning ────────────────────────────────────────────────────

async def test_reason_node_audit_preview_is_a_string_for_block_content():
    llm = AsyncMock()
    inner = _llm_returning(_blocks("Considering the payout velocity."))
    inner.tool_calls = []
    llm.bind_tools = lambda tools: inner

    with patch("agent.nodes._get_llm", return_value=llm):
        result = await reason_node(_state())

    preview = result["audit_trail"][-1]["content_preview"]
    assert isinstance(preview, str)
    assert preview == "Considering the payout velocity."
