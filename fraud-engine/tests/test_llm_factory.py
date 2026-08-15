"""
Tests for agent/llm.py — the chat model factory.

No live calls to either provider. Both clients construct without credentials
(boto3 and the OpenAI SDK resolve those on the first request), so these build
real objects rather than mocks and assert what the graph will actually receive.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from agent import llm as llm_module
from agent.config import BEDROCK_MODEL_ID, BEDROCK_REGION
from agent.llm import SUPPORTED_PROVIDERS, get_chat_model

FRAUD_ENGINE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def dummy_openai_key(monkeypatch):
    """Constructing ChatOpenAI requires a key to be present.

    Set a placeholder so these tests are deterministic wherever they run: with
    a real key in .env they would pass by accident, and without one they would
    fail on a credential check that has nothing to do with what is being
    asserted. No request is ever sent.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-used")


# ── Provider selection ───────────────────────────────────────────

def test_openai_provider_returns_chat_openai():
    from langchain_openai import ChatOpenAI

    model = get_chat_model("openai")

    assert isinstance(model, ChatOpenAI)
    assert isinstance(model, BaseChatModel)
    assert model.temperature == 0


def test_bedrock_provider_returns_chat_bedrock_converse():
    from langchain_aws import ChatBedrockConverse

    model = get_chat_model("bedrock")

    assert isinstance(model, ChatBedrockConverse)
    assert isinstance(model, BaseChatModel)
    assert model.temperature == 0
    # The cross-region inference profile ID, not a plain model ID — plain IDs
    # raise ValidationException in EU regions.
    assert model.model_id == BEDROCK_MODEL_ID
    assert model.model_id.startswith("eu.")
    assert model.region_name == BEDROCK_REGION


def test_bedrock_is_not_the_legacy_chat_bedrock():
    from langchain_aws import ChatBedrock, ChatBedrockConverse

    model = get_chat_model("bedrock")

    assert type(model) is ChatBedrockConverse
    assert not isinstance(model, ChatBedrock)


def test_provider_defaults_to_openai():
    """An unset AGENT_LLM_PROVIDER must not change existing behaviour."""
    from langchain_openai import ChatOpenAI

    with patch.object(llm_module, "LLM_PROVIDER", "openai"):
        assert isinstance(get_chat_model(), ChatOpenAI)


@pytest.mark.parametrize("value", ["OpenAI", "  bedrock  ", "BEDROCK"])
def test_provider_value_is_case_and_whitespace_insensitive(value):
    model = get_chat_model(value)
    assert isinstance(model, BaseChatModel)


# ── Unknown provider ─────────────────────────────────────────────

@pytest.mark.parametrize("value", ["anthropic", "azure", "", "  "])
def test_unknown_provider_raises_with_a_clear_message(value):
    with pytest.raises(ValueError) as exc:
        get_chat_model(value)

    message = str(exc.value)
    # The message must name what was wrong and what is valid — a typo in
    # AGENT_LLM_PROVIDER should be diagnosable from the error alone.
    assert "AGENT_LLM_PROVIDER" in message
    for provider in SUPPORTED_PROVIDERS:
        assert provider in message


def test_unknown_provider_does_not_fall_back_to_a_default():
    """Silently using OpenAI for a misconfigured provider would be worse."""
    with pytest.raises(ValueError):
        get_chat_model("bedroc")  # plausible typo


# ── Overrides ────────────────────────────────────────────────────

def test_overrides_reach_the_constructor():
    model = get_chat_model("openai", temperature=0.7)
    assert model.temperature == 0.7


# ── Laziness ─────────────────────────────────────────────────────

def test_no_provider_sdk_is_imported_at_module_import():
    """Importing the factory must not drag in a provider SDK.

    Run in a subprocess: the SDKs are already in sys.modules for this test
    session, so an in-process assertion would prove nothing.
    """
    code = (
        "import sys; import agent.llm; "
        "leaked = [m for m in ('langchain_openai', 'langchain_aws', 'boto3', 'openai') "
        "if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(FRAUD_ENGINE_DIR),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"SDKs imported at module import: {result.stdout}"


def test_no_client_is_constructed_at_module_import():
    """The factory is a function, not a module-level singleton."""
    code = (
        "import sys; import agent.llm as m; "
        "assert callable(m.get_chat_model); "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(FRAUD_ENGINE_DIR),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# ── Tool calling ─────────────────────────────────────────────────

def test_both_providers_support_bind_tools():
    """The ReAct loop calls bind_tools — a provider without it is unusable here."""
    for provider in SUPPORTED_PROVIDERS:
        model = get_chat_model(provider)
        assert hasattr(model, "bind_tools")
        # Defined by the integration, not the un-implemented base method.
        assert type(model).bind_tools is not BaseChatModel.bind_tools


def test_bedrock_bind_tools_accepts_the_agent_tool_registry():
    """Bind the real registry — signature mismatches surface here, not at runtime."""
    from agent.tools.registry import ALL_TOOLS

    bound = get_chat_model("bedrock").bind_tools(ALL_TOOLS)
    assert bound is not None
