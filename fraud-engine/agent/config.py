"""
Agent configuration — all external URLs and LLM settings in one place.
"""
import os
from pathlib import Path

from env_bootstrap import load_env

# Must run before the os.getenv calls below, or .env values lose to the defaults.
load_env()

NESTJS_BASE_URL = os.getenv("NESTJS_BASE_URL", "http://localhost:3000")
FRAUD_ENGINE_URL = os.getenv("FRAUD_ENGINE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL")

# ── LLM ──────────────────────────────────────────────────────────
# Which provider the agent talks to. "openai" is the default so an existing
# deployment keeps its current behaviour without touching .env.
LLM_PROVIDER = os.getenv("AGENT_LLM_PROVIDER", "openai")

# Per-provider model settings, deliberately separate: switching provider should
# not require re-setting the model, and an OpenAI model name is meaningless to
# Bedrock. OPENAI_MODEL is read as a fallback so an older .env still applies.
OPENAI_MODEL = os.getenv("AGENT_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

# Cross-region inference profile ID, not a plain model ID — plain IDs fail with
# ValidationException in EU regions.
BEDROCK_MODEL_ID = os.getenv(
    "AGENT_BEDROCK_MODEL_ID",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
)
BEDROCK_REGION = os.getenv("AGENT_BEDROCK_REGION", "eu-central-1")

# Agent limits
MAX_ITERATIONS = 8

# Similar-case retrieval
SIMILAR_CASES_DEFAULT_LIMIT = 3
SIMILAR_CASES_MAX_LIMIT = 5
RAG_INDEX_PATH = os.getenv(
    "RAG_INDEX_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "indexed_cases.json"),
)
