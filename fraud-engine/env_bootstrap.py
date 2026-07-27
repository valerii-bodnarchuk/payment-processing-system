"""
env_bootstrap — make `.env` load automatically for every fraud-engine entrypoint.

The Python side (agent, eval, scripts) reads configuration straight from the
process environment via os.getenv, while the repo-root `.env` is otherwise only
consumed by NestJS/Prisma. Without this module the agent and the eval harness
silently fall back to their defaults — most visibly OPENAI_API_KEY, which has no
default and turns every LLM call into a fallback INCONCLUSIVE verdict.

Precedence (highest wins):
    1. real process environment  — explicit exports and CI secrets
    2. fraud-engine/.env         — local, gitignored, Python-only overrides
    3. <repo root>/.env          — shared with NestJS/Prisma

python-dotenv never overwrites a variable that is already set, so loading the
more specific file first is what produces that order. Nothing here can clobber
an export the caller made deliberately, which is what keeps CI reproducible.

Call load_env() at the top of a module BEFORE any module-level os.getenv().
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_FRAUD_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FRAUD_ENGINE_DIR.parent


def load_env() -> None:
    """Populate os.environ from the .env files, most specific first.

    Idempotent and safe to call from several entrypoints in one process: an
    already-set variable is left untouched on every subsequent call. Missing
    files are not an error — the defaults in agent/config.py still apply.
    """
    load_dotenv(_FRAUD_ENGINE_DIR / ".env")
    load_dotenv(_REPO_ROOT / ".env")
