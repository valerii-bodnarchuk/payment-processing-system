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

# LLM
OPENAI_MODEL = os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")

# Agent limits
MAX_ITERATIONS = 8

# Similar-case retrieval
SIMILAR_CASES_DEFAULT_LIMIT = 3
SIMILAR_CASES_MAX_LIMIT = 5
RAG_INDEX_PATH = os.getenv(
    "RAG_INDEX_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "indexed_cases.json"),
)
