"""
Append-only persistence for completed investigation runs.

Persistence is best-effort: if DATABASE_URL is not configured or the database is
unreachable, the graph still returns its verdict and logs the failure.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg

from agent.config import DATABASE_URL

logger = logging.getLogger("agent.persistence.audit")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _extract_tool_calls(messages: list[Any]) -> list[dict]:
    # Imported here, not at module scope: agent.nodes imports this module for
    # persist_investigation_run, so a top-level import would be circular.
    from agent.nodes import _message_text

    calls: list[dict] = []
    for index, msg in enumerate(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                calls.append({
                    "message_index": index,
                    "type": "tool_call",
                    "name": call.get("name"),
                    "args": _jsonable(call.get("args", {})),
                    "id": call.get("id"),
                })

        name = getattr(msg, "name", None)
        if name:
            calls.append({
                "message_index": index,
                "type": "tool_result",
                "name": name,
                "content": _message_text(getattr(msg, "content", None))[:4000],
            })

    return calls


async def persist_investigation_run(
    state: dict,
    audit_trail: list[dict],
    database_url: str | None = None,
) -> dict:
    """
    Persist one completed investigation run and its ordered audit entries.

    Returns a status dict and never raises to the graph caller.
    """
    db_url = database_url if database_url is not None else DATABASE_URL
    if not db_url:
        return {"persisted": False, "reason": "DATABASE_URL not configured"}

    verdict = state.get("verdict") or {}
    completed_at = _parse_timestamp(
        audit_trail[-1].get("timestamp") if audit_trail else None,
    )
    tool_calls = _extract_tool_calls(state.get("messages", []))

    conn = None
    try:
        conn = await asyncpg.connect(db_url)
        async with conn.transaction():
            run_id = await conn.fetchval(
                """
                INSERT INTO "InvestigationRun" (
                    "transactionId", "trigger", "verdict", "confidence",
                    "riskLevel", "summary", "verdictPayload", "toolCalls", "completedAt"
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9)
                RETURNING id
                """,
                state["transaction_id"],
                state.get("trigger", "MANUAL"),
                verdict.get("verdict"),
                verdict.get("confidence"),
                verdict.get("risk_level"),
                verdict.get("summary"),
                json.dumps(_jsonable(verdict)),
                json.dumps(_jsonable(tool_calls)),
                completed_at,
            )

            for sequence, entry in enumerate(audit_trail):
                await conn.execute(
                    """
                    INSERT INTO "InvestigationAuditEntry" (
                        "runId", "sequence", "timestamp", "action", "stage", "payload"
                    ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    run_id,
                    sequence,
                    _parse_timestamp(entry.get("timestamp")),
                    str(entry.get("action", "unknown")),
                    entry.get("stage"),
                    json.dumps(_jsonable(entry)),
                )

        return {"persisted": True, "run_id": run_id}
    except Exception as e:
        logger.exception("Failed to persist investigation run")
        return {"persisted": False, "reason": str(e)}
    finally:
        if conn is not None:
            await conn.close()
