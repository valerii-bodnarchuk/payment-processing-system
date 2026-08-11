"""
FastAPI router for the investigation agent.
Mounted on the existing fraud-engine app at /investigate.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("agent.api")

router = APIRouter(prefix="/investigate", tags=["Investigation Agent"])


class InvestigateRequest(BaseModel):
    transaction_id: int
    trigger: Literal["BLOCK", "REVIEW", "MANUAL"] = "MANUAL"


class InvestigateResponse(BaseModel):
    transaction_id: int
    verdict: dict
    audit_trail: list[dict] = Field(default_factory=list)
    iterations_used: int
    # None when the run was not persisted (no DATABASE_URL, or a write failure).
    run_id: int | None = None
    # True when the verdict did not come from a complete reasoning pass. Still
    # returned with 200: a degraded verdict carries the collected context and is
    # useful to the human reading it — it just must not be mistaken for a
    # reasoned one. degradation_reason names the failure (agent.nodes
    # DEGRADATION_*).
    degraded: bool = False
    degradation_reason: str | None = None


@router.post("", response_model=InvestigateResponse)
async def investigate(req: InvestigateRequest):
    """
    Run a fraud investigation on a transaction.

    The agent collects context from NestJS endpoints, runs a ReAct reasoning
    loop with LLM + tools, and produces a structured verdict.
    """
    # Lazy import to avoid loading LangGraph at module import time
    # (allows fraud engine to start without OPENAI_API_KEY for /check endpoint)
    from agent.graph import investigation_graph

    try:
        result = await investigation_graph.ainvoke({
            "transaction_id": req.transaction_id,
            "trigger": req.trigger,
        })
    except Exception as e:
        logger.exception(f"Investigation failed for transaction {req.transaction_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Investigation failed: {str(e)}",
        )

    verdict = result.get("verdict")
    if not verdict:
        raise HTTPException(
            status_code=500,
            detail="Agent produced no verdict",
        )

    return InvestigateResponse(
        transaction_id=req.transaction_id,
        verdict=verdict,
        audit_trail=result.get("audit_trail", []),
        iterations_used=result.get("iteration", 0),
        run_id=result.get("run_id"),
        degraded=bool(result.get("degraded", False)),
        degradation_reason=result.get("degradation_reason"),
    )


@router.post("/stream")
async def investigate_stream(req: InvestigateRequest):
    """Stream the same investigation as POST /investigate, but emit one
    Server-Sent Event per LangGraph node delta plus a terminal `done` event.

    Persistence is unchanged: audit_node still writes the run to PostgreSQL at
    the end. The stream is additive observability, not a replacement for the
    durable record.
    """
    from agent.streaming import stream_investigation

    return StreamingResponse(
        stream_investigation(req.transaction_id, req.trigger),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable proxy buffering so events flush as the graph progresses
            # instead of arriving as one chunk at the end.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def agent_health():
    """Health check for the investigation agent subsystem."""
    return {"status": "healthy", "service": "investigation-agent"}
