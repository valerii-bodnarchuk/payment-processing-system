"""
Unit tests for eval/guard.py::assert_no_contamination.

No DB, no network, no LLM — pure state inspection.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval.guard import EVAL_MARKER_PREFIX, assert_no_contamination

MARKER = f"{EVAL_MARKER_PREFIX}deadbeef1234"


def _clean_state() -> dict:
    return {
        "messages": [
            HumanMessage(content="Investigate transaction 42."),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_seller_risk_profile",
                    "args": {"seller_id": 7},
                    "id": "call_1",
                }],
            ),
            ToolMessage(
                content='{"seller": {"name": "Aurora Handmade Goods", "email": "ops@aurora-goods.example"}}',
                tool_call_id="call_1",
            ),
        ],
        "seller_profile": {"seller": {"name": "Aurora Handmade Goods"}},
        "verdict": {"verdict": "TRUE_POSITIVE"},
    }


def test_clean_state_passes():
    assert_no_contamination(_clean_state(), case_id="case-clean")


def test_marker_in_tool_message_content_raises():
    state = _clean_state()
    state["messages"][2] = ToolMessage(
        content=f'{{"seller": {{"name": "{MARKER}"}}}}',
        tool_call_id="call_1",
    )
    with pytest.raises(ValueError, match="Contamination detected"):
        assert_no_contamination(state, case_id="case-leak")


def test_case_id_appears_in_error():
    state = _clean_state()
    state["messages"].append(ToolMessage(content=MARKER, tool_call_id="call_9"))
    with pytest.raises(ValueError, match="case-xyz"):
        assert_no_contamination(state, case_id="case-xyz")


def test_marker_in_tool_call_args_raises():
    state = _clean_state()
    state["messages"][1] = AIMessage(
        content="",
        tool_calls=[{
            "name": "get_transaction_context",
            "args": {"note": MARKER},
            "id": "call_1",
        }],
    )
    with pytest.raises(ValueError):
        assert_no_contamination(state)


def test_marker_in_context_field_raises():
    state = _clean_state()
    state["seller_profile"] = {"seller": {"stripeAccountId": MARKER}}
    with pytest.raises(ValueError, match="state.seller_profile"):
        assert_no_contamination(state)


def test_marker_in_plain_dict_message_raises():
    state = _clean_state()
    state["messages"].append({"role": "tool", "content": MARKER})
    with pytest.raises(ValueError):
        assert_no_contamination(state)


def test_no_messages_key_is_safe():
    assert_no_contamination({"verdict": {"verdict": "INCONCLUSIVE"}})
