import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.models import (
    FraudCheckRequest,
    FraudCheckResponse,
    DetailedFraudCheckResponse,
    OutcomeReport,
    OutcomeStats,
    Decision,
    RuleResult,
)
from app.rules.engine import evaluate
from app.config import CONFIG

_telemetry_log = logging.getLogger("agent.telemetry")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start tracing on boot, flush it on shutdown.

    Both halves are guarded: tracing is observability, and a broken exporter or
    a missing opentelemetry install must not take the fraud engine with it. The
    import is local so that a machine without the OTel packages still serves
    /check — same reasoning as the optional agent router below.
    """
    shutdown = None
    try:
        from agent.telemetry import setup_telemetry, shutdown_telemetry

        setup_telemetry()
        shutdown = shutdown_telemetry
    except Exception:
        _telemetry_log.warning(
            "Telemetry initialisation failed — continuing without tracing",
            exc_info=True,
        )

    yield

    if shutdown is not None:
        try:
            # Flushes the BatchSpanProcessor; without it the last batch of spans
            # dies with the process.
            shutdown()
        except Exception:
            _telemetry_log.warning("Telemetry shutdown failed", exc_info=True)


app = FastAPI(
    title="Fraud Engine",
    description="Rule-based fraud scoring for payment processing",
    version="1.0.0",
    lifespan=lifespan,
)

# In-memory outcome store — demo only, not persisted across restarts
_outcomes: list[OutcomeReport] = []

# Running decision counter (incremented on every /check call)
_total_decisions: int = 0


def score_to_decision(score: float) -> Decision:
    boundaries = CONFIG["decision_boundaries"]
    if score >= boundaries["block_above"]:
        return Decision.BLOCK
    if score >= boundaries["allow_below"]:
        return Decision.REVIEW
    return Decision.ALLOW


def build_explanation(
    decision: Decision,
    risk_score: float,
    triggered: list[RuleResult],
) -> str:
    score_str = str(round(risk_score, 2))
    if decision == Decision.ALLOW:
        return f"Transaction passed all checks (score: {score_str})"
    rule_names = ", ".join(r.rule for r in triggered)
    dominant = max(triggered, key=lambda r: r.score)
    dominant_str = f"{dominant.rule} ({dominant.reason})"
    if decision == Decision.REVIEW:
        return (
            f"Flagged for review: {rule_names}. "
            f"Dominant factor: {dominant_str}. "
            f"Score: {score_str}"
        )
    return (
        f"Blocked: {rule_names}. "
        f"Primary risk: {dominant_str}. "
        f"Combined score: {score_str}"
    )


def _build_response(req: FraudCheckRequest) -> tuple[FraudCheckResponse, list[RuleResult]]:
    """Core check logic shared by /check and /check/explain."""
    global _total_decisions
    _total_decisions += 1

    risk_score, all_results = evaluate(req)
    risk_score_rounded = round(risk_score, 2)
    decision = score_to_decision(risk_score_rounded)
    triggered = [r for r in all_results if r.triggered]

    response = FraudCheckResponse(
        transaction_id=req.transaction_id,
        risk_score=risk_score_rounded,
        decision=decision,
        rules_triggered=triggered,
        explanation=build_explanation(decision, risk_score_rounded, triggered),
    )
    return response, all_results


@app.post("/check", response_model=FraudCheckResponse)
def fraud_check(req: FraudCheckRequest):
    response, _ = _build_response(req)
    return response


@app.post("/check/explain", response_model=DetailedFraudCheckResponse)
def fraud_check_explain(req: FraudCheckRequest):
    response, all_results = _build_response(req)
    score_breakdown = {
        r.rule: round(r.score * CONFIG["rules"][r.rule]["weight"], 4)
        for r in all_results
    }
    return DetailedFraudCheckResponse(
        **response.model_dump(),
        all_rules=all_results,
        config_version=CONFIG["version"],
        score_breakdown=score_breakdown,
    )


@app.post("/outcomes", response_model=OutcomeReport)
def record_outcome(report: OutcomeReport):
    _outcomes.append(report)
    return report


@app.get("/outcomes/stats", response_model=OutcomeStats)
def outcome_stats():
    fraud_outcomes = {"fraudulent", "chargeback", "disputed"}

    true_positives = sum(
        1 for o in _outcomes
        if o.original_decision in (Decision.BLOCK, Decision.REVIEW)
        and o.actual_outcome in fraud_outcomes
    )
    false_positives = sum(
        1 for o in _outcomes
        if o.original_decision in (Decision.BLOCK, Decision.REVIEW)
        and o.actual_outcome == "legitimate"
    )
    false_negatives = sum(
        1 for o in _outcomes
        if o.original_decision == Decision.ALLOW
        and o.actual_outcome in fraud_outcomes
    )

    predicted_positive = true_positives + false_positives
    actual_positive = true_positives + false_negatives

    return OutcomeStats(
        total_decisions=_total_decisions,
        outcomes_reported=len(_outcomes),
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=round(true_positives / predicted_positive, 4) if predicted_positive else None,
        recall=round(true_positives / actual_positive, 4) if actual_positive else None,
    )


# ── Investigation Agent (optional — only loads if dependencies are installed) ──
try:
    from agent.api import router as agent_router
    app.include_router(agent_router)
    logging.getLogger("agent").info("Investigation agent router mounted at /investigate")
except ImportError:
    logging.getLogger("agent").info(
        "Investigation agent not loaded — install langgraph, langchain-core, langchain-openai, httpx"
    )


@app.get("/health")
def health():
    return {"status": "healthy", "service": "fraud-engine"}


# ── HTTP instrumentation (last: every route must already be registered) ──
#
# `/health` matches both this app's probe and the agent's /investigate/health.
# Excluding them keeps liveness polling out of the trace store, where it would
# otherwise be the overwhelming majority of spans. There is no Python /metrics
# endpoint — Prometheus metrics live on the NestJS side.
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="/health")
except Exception:
    _telemetry_log.warning(
        "FastAPI instrumentation unavailable — HTTP spans disabled",
        exc_info=True,
    )
