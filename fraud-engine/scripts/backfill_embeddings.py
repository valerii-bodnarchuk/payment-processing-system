"""
Backfill the Case table with SEED_CASES and InvestigationRun-derived rows,
embedding each via OpenAI text-embedding-3-small.

Idempotent: rows are keyed by caseId; existing rows are skipped. To re-embed
the whole corpus, truncate the Case table first:

    psql … -c 'TRUNCATE "Case" RESTART IDENTITY;'

Both case sources flow through the same normalize_signals + _canonical_text
projection in agent.rag.embeddings, so the embedded text is identical in
shape regardless of source — preventing systematic bias between seed and
run vectors.

Usage:
    python scripts/backfill_embeddings.py [--batch-size 100] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Make `agent` importable when running this file directly.
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import asyncpg
from pgvector.asyncpg import register_vector

from env_bootstrap import load_env

# Must run before main() reads DATABASE_URL and before the OpenAI client is
# constructed for embedding.
load_env()

from agent.rag.cases import SEED_CASES
from agent.rag.synthetic_cases import SYNTHETIC_CASES
from agent.rag.embeddings import (
    DEFAULT_BATCH_SIZE,
    _canonical_text,
    embed_batch,
    normalize_signals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backfill")


# Keyword → signal map. Mirrors agent.rag.indexer.KEYWORD_SIGNALS so the
# signal vocabulary derived from InvestigationRun matches what the legacy
# JSON indexer produced for the Jaccard baseline.
KEYWORD_SIGNALS = {
    "amount_threshold": "rule:amount_threshold",
    "velocity": "rule:velocity",
    "failed_history": "rule:failed_history",
    "new_account": "rule:new_account",
    "dispute_rate": "rule:dispute_rate",
    "active_dispute": "rule:active_dispute",
    "ledger_imbalanced": "rule:ledger_imbalanced",
    "ledger_inconsistency": "rule:ledger_inconsistency",
    "insufficient_escrow": "rule:insufficient_escrow",
    "seller_blocked": "rule:seller_blocked",
}


def _risk_bucket_from_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.85:
        return "risk:critical"
    if score >= 0.7:
        return "risk:high"
    if score >= 0.3:
        return "risk:medium"
    return "risk:low"


def _safe_json(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _signals_from_run(
    verdict_payload: dict, tool_calls: list[dict], risk_level: str
) -> list[str]:
    """Project a persisted InvestigationRun into the same signal vocabulary
    that SEED_CASES use. Mirrors agent.rag.indexer.build_case_from_run logic
    so both sources land in the corpus with comparable signal strings."""
    signals: set[str] = set()

    for call in tool_calls or []:
        args = call.get("args") or {}

        fraud_decision = args.get("fraud_decision")
        if fraud_decision:
            signals.add(f"decision:{str(fraud_decision).strip().upper()}")

        risk = _risk_bucket_from_score(args.get("fraud_score"))
        if risk:
            signals.add(risk)

        for finding in args.get("findings") or []:
            n = str(finding).strip().lower().replace(" ", "_")
            if not n:
                continue
            signals.add(n if ":" in n else f"rule:{n}")

    signals.add(f"risk:{str(risk_level).lower()}")

    blob_parts: list[str] = [str(verdict_payload.get("summary") or "")]
    for k in ("key_findings", "recommended_actions"):
        for item in verdict_payload.get(k) or []:
            blob_parts.append(str(item))
    for evidence in verdict_payload.get("evidence") or []:
        if isinstance(evidence, dict):
            for k in ("source", "fact", "significance"):
                blob_parts.append(str(evidence.get(k) or ""))
    blob = " ".join(blob_parts).lower().replace(" ", "_")
    for keyword, signal in KEYWORD_SIGNALS.items():
        if keyword in blob:
            signals.add(signal)

    return normalize_signals(signals)


def _synthetic_summary(verdict_payload: dict, run_id: int) -> str:
    """Never embed an empty summary — backfill spec requires a synthetic
    fallback derived from verdictPayload when the persisted summary is null."""
    s = (verdict_payload.get("summary") or "").strip()
    if s:
        return s
    findings = verdict_payload.get("key_findings") or []
    if findings:
        return "Investigation findings: " + "; ".join(str(f) for f in findings[:3])
    return f"Persisted investigation run #{run_id} — no narrative recorded."


def _python_case_to_row(case: dict, source: str) -> dict:
    """Map a Python-dict case (SEED_CASES or SYNTHETIC_CASES) to a Case row.
    `source` is recorded verbatim — 'seed' for production seeds, 'synthetic'
    for evaluation fixtures — so the eval can stratify on it."""
    return {
        "caseId": case["case_id"],
        "source": source,
        "runId": None,
        "verdict": case["verdict"],
        "riskLevel": str(case["risk_level"]).upper(),
        "summary": case["summary"],
        "signals": normalize_signals(case.get("signals") or []),
        "recommendedActions": case.get("recommended_actions") or [],
    }


def _run_to_row(row: dict) -> dict:
    payload = _safe_json(row.get("verdictPayload")) or {}
    tool_calls = _safe_json(row.get("toolCalls")) or []
    risk_level = str(
        row.get("riskLevel") or payload.get("risk_level") or "MEDIUM"
    ).upper()
    return {
        "caseId": f"run_{row['id']}",
        "source": "run",
        "runId": row["id"],
        "verdict": row.get("verdict") or payload.get("verdict") or "INCONCLUSIVE",
        "riskLevel": risk_level,
        "summary": _synthetic_summary(payload, row["id"]),
        "signals": _signals_from_run(payload, tool_calls, risk_level),
        "recommendedActions": payload.get("recommended_actions") or [],
    }


async def _existing_case_ids(conn) -> set[str]:
    rows = await conn.fetch('SELECT "caseId" FROM "Case"')
    return {r["caseId"] for r in rows}


async def _fetch_run_rows(conn, limit: int | None) -> list[dict]:
    query = """
        SELECT id, "transactionId", trigger, verdict, confidence, "riskLevel",
               summary, "verdictPayload", "toolCalls", "completedAt"
        FROM "InvestigationRun"
        WHERE verdict IS NOT NULL
        ORDER BY id ASC
    """
    if limit is not None:
        query += f"\n        LIMIT {int(limit)}"
    return [dict(r) for r in await conn.fetch(query)]


async def main(batch_size: int, limit: int | None) -> dict:
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/payment_system",
    )

    conn = await asyncpg.connect(db_url)
    try:
        await register_vector(conn)
        existing = await _existing_case_ids(conn)
        logger.info("Existing Case rows: %d", len(existing))

        seed_candidates: list[dict] = []
        for c in SEED_CASES:
            row = _python_case_to_row(c, "seed")
            if row["caseId"] in existing:
                continue
            seed_candidates.append(row)

        synthetic_candidates: list[dict] = []
        for c in SYNTHETIC_CASES:
            row = _python_case_to_row(c, "synthetic")
            if row["caseId"] in existing:
                continue
            synthetic_candidates.append(row)

        run_rows = await _fetch_run_rows(conn, limit)
        logger.info("InvestigationRun rows fetched: %d", len(run_rows))

        run_candidates: list[dict] = []
        for r in run_rows:
            row = _run_to_row(r)
            if row["caseId"] in existing:
                continue
            run_candidates.append(row)

        candidates = seed_candidates + synthetic_candidates + run_candidates
        if not candidates:
            logger.info("Nothing to embed — all caseIds already present.")
            return {
                "embedded_seed": 0,
                "embedded_synthetic": 0,
                "embedded_run": 0,
                "skipped_existing": len(existing),
                "total_in_table": len(existing),
            }

        logger.info(
            "Embedding %d cases (%d seed, %d synthetic, %d run) in batches of %d...",
            len(candidates),
            len(seed_candidates),
            len(synthetic_candidates),
            len(run_candidates),
            batch_size,
        )

        embedded_seed = 0
        embedded_synthetic = 0
        embedded_run = 0
        n_batches = (len(candidates) + batch_size - 1) // batch_size

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            texts = [_canonical_text(c) for c in batch]
            vectors = embed_batch(texts)

            async with conn.transaction():
                for case_row, vec in zip(batch, vectors):
                    await conn.execute(
                        """
                        INSERT INTO "Case" (
                            "caseId", "source", "runId", "verdict", "riskLevel",
                            "summary", "signals", "recommendedActions", "embedding"
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                        """,
                        case_row["caseId"],
                        case_row["source"],
                        case_row["runId"],
                        case_row["verdict"],
                        case_row["riskLevel"],
                        case_row["summary"],
                        case_row["signals"],
                        json.dumps(case_row["recommendedActions"]),
                        vec,
                    )
                    if case_row["source"] == "seed":
                        embedded_seed += 1
                    elif case_row["source"] == "synthetic":
                        embedded_synthetic += 1
                    else:
                        embedded_run += 1

            logger.info(
                "  Batch %d/%d committed (cumulative seed=%d synthetic=%d run=%d)",
                i // batch_size + 1,
                n_batches,
                embedded_seed,
                embedded_synthetic,
                embedded_run,
            )

        total_now = (
            len(existing) + embedded_seed + embedded_synthetic + embedded_run
        )
        logger.info(
            "Embedded %d seed cases, %d synthetic cases, %d run cases (total in table: %d)",
            embedded_seed,
            embedded_synthetic,
            embedded_run,
            total_now,
        )
        return {
            "embedded_seed": embedded_seed,
            "embedded_synthetic": embedded_synthetic,
            "embedded_run": embedded_run,
            "skipped_existing": len(existing),
            "total_in_table": total_now,
        }
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Case embeddings.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit InvestigationRun rows considered (for testing).",
    )
    args = parser.parse_args()

    result = asyncio.run(main(args.batch_size, args.limit))
    print(json.dumps(result, indent=2))
