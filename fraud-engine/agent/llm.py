"""
Chat model factory — the one place that decides which LLM provider the agent uses.

There is deliberately no LLMProvider class here. LangChain's BaseChatModel is
already the provider abstraction: it is what `bind_tools` and `ainvoke` are
defined on, and it is what the graph consumes. Wrapping it again would add a
layer with nothing to do.

Two rules this module exists to keep:

1. No provider SDK is imported at module import time — each import happens inside
   its branch. `agent/mcp_server.py` must start with no credentials of any kind
   present, and an unconditional `import langchain_aws` (which pulls in boto3)
   would also make every test pay for an SDK it never calls.
2. No client is constructed at import time, for the same reason. Construction is
   cheap and lazy for both providers; credentials are resolved later, on the
   first call.
"""
from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from agent.config import (
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    LLM_PROVIDER,
    OPENAI_MODEL,
)

logger = logging.getLogger("agent.llm")

# Every provider this factory can build. Kept as data so the error message can
# name the valid options rather than hardcoding them in a string.
SUPPORTED_PROVIDERS = ("openai", "bedrock")


def get_chat_model(provider: str | None = None, **overrides) -> BaseChatModel:
    """Build the configured chat model.

    `provider` defaults to AGENT_LLM_PROVIDER. `overrides` are passed through to
    the underlying constructor, which is how a caller changes temperature or
    model without this factory growing a parameter per provider knob.

    Raises ValueError on an unknown provider — a typo in AGENT_LLM_PROVIDER
    should fail loudly at the first LLM call, not silently fall back to a
    provider the operator did not choose.
    """
    # Only None means "use the configured default". An explicit empty string is
    # a value someone passed or set, so it validates like any other — an empty
    # AGENT_LLM_PROVIDER is a misconfiguration, not a request for the default.
    selected = LLM_PROVIDER if provider is None else provider
    name = (selected or "").strip().lower()

    if name == "openai":
        # Imported here, not at module scope — see rule 1 above.
        from langchain_openai import ChatOpenAI

        params = {"model": OPENAI_MODEL, "temperature": 0}
        params.update(overrides)
        return ChatOpenAI(**params)

    if name == "bedrock":
        from langchain_aws import ChatBedrockConverse

        # ChatBedrockConverse, not the legacy ChatBedrock: the Converse API is
        # what carries tool calling across Bedrock model families.
        #
        # Credentials are intentionally absent here. boto3 resolves them from
        # its standard chain (env, ~/.aws/credentials, instance role) when the
        # first request goes out, so nothing AWS-shaped belongs in .env.
        params = {
            "model": BEDROCK_MODEL_ID,
            "region_name": BEDROCK_REGION,
            "temperature": 0,
        }
        params.update(overrides)
        return ChatBedrockConverse(**params)

    raise ValueError(
        f"Unknown AGENT_LLM_PROVIDER {name!r}. "
        f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
    )
