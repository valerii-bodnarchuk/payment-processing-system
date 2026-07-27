"""
Run eval/runner.py against real infra with a SCRIPTED judge instead of an LLM.

The LLM is replaced, so no OPENAI_API_KEY is needed. Everything else is real:
Postgres, NestJS :3000, fraud engine :8000, graph routing, tool HTTP calls,
contamination guard, scoring.

The scripted judge reads the ACTUAL tool output and decides by comparing the
score recomputed by the fraud engine against the score stored on the payout.
This measures whether the pipeline delivers enough signal to separate the two
cases — NOT whether an LLM would reason its way there.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from env_bootstrap import load_env

# Must run before main() reads DATABASE_URL out of the environment.
load_env()


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _last_tool_output(messages, tool_name):
    for msg in reversed(messages):
        if getattr(msg, "name", None) == tool_name:
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def make_scripted_judge(trace):
    """Fresh mock per case. Walks the real tools, then judges on real data."""
    state = {"step": 0}

    async def mock_ainvoke(messages, *args, **kwargs):
        try:
            return await _judge(messages, *args, **kwargs)
        except Exception:
            import traceback
            print("!!! ERROR INSIDE THE MOCK (the reason node swallows it):")
            traceback.print_exc()
            raise

    async def _judge(messages, *args, **kwargs):
        state["step"] += 1
        step = state["step"]
        tools_present = [getattr(m, "name", None) for m in messages
                         if type(m).__name__ == "ToolMessage"]
        print(f"    [step {step}] messages={len(messages)} tool_outputs={tools_present}")

        if step == 1:
            tx = _tool_call("get_transaction_context",
                            {"transaction_id": trace["transaction_id"]}, "c1")
            return tx

        # Only overwrite when the parse actually yields data: at synthesis the
        # messages carry no ToolMessages and would blank the collected trace.
        ctx = _last_tool_output(messages, "get_transaction_context")
        if ctx and ctx.get("payoutReports"):
            report = ctx["payoutReports"][0]
            trace["stored_score"] = (report.get("context") or {}).get("fraudScore")
            trace["stored_decision"] = (report.get("context") or {}).get("fraudDecision")
            trace["seller_id"] = report.get("sellerId")
            trace["amount"] = report.get("amount")
            trace["payout_id"] = report.get("payoutId")

        if step == 2:
            # The payout under investigation is excluded from the window volume:
            # the engine's daily_volume rule adds `amount` back itself, so
            # forwarding the unfiltered total would count this payout twice.
            # Every OTHER payout in the window stays in the aggregate.
            return _tool_call("get_seller_risk_profile",
                              {"seller_id": trace["seller_id"],
                               "exclude_payout_id": trace["payout_id"]}, "c2")

        prof = _last_tool_output(messages, "get_seller_risk_profile")
        if prof and prof.get("riskMetrics"):
            metrics = prof["riskMetrics"]
            trace["history"] = {
                "accountAgeDays": metrics.get("accountAgeDays"),
                "payoutVelocity24h": metrics.get("payoutVelocity24h"),
                "totalVolume24h": metrics.get("totalVolume24h"),
                "totalDisputes": metrics.get("totalDisputes"),
                "failedPayouts": metrics.get("failedPayouts"),
            }

        if step == 3:
            h = trace.get("history") or {}
            return _tool_call("get_fraud_score_explanation", {
                "transaction_id": trace["transaction_id"],
                "seller_id": trace["seller_id"],
                "amount": trace["amount"],
                "seller_payout_count_24h": h.get("payoutVelocity24h") or 0,
                "seller_total_amount_24h": h.get("totalVolume24h") or 0,
                "seller_failed_payouts_7d": h.get("failedPayouts") or 0,
                "seller_account_age_days": h.get("accountAgeDays") or 0,
                "seller_dispute_count": h.get("totalDisputes") or 0,
            }, "c3")

        expl = _last_tool_output(messages, "get_fraud_score_explanation")
        if expl and expl.get("risk_score") is not None:
            trace["recomputed_score"] = expl.get("risk_score")
            trace["recomputed_decision"] = expl.get("decision")

        if step == 4:
            return AIMessage(content="Data gathered.\n\nINVESTIGATION_COMPLETE")

        # ── Synthesis: decide from the real numbers ──
        recomputed = trace.get("recomputed_score")
        stored = trace.get("stored_score")
        h = trace.get("history") or {}
        behavioural = sum(filter(None, [
            (h.get("payoutVelocity24h") or 0) >= 5,
            (h.get("accountAgeDays") or 999) < 7,
            (h.get("totalDisputes") or 0) >= 1,
            (h.get("failedPayouts") or 0) >= 2,
        ]))
        trace["behavioural_signals"] = behavioural

        if recomputed is not None and recomputed > 0.7 and behavioural >= 2:
            verdict, reason = "TRUE_POSITIVE", "recomputed score confirms BLOCK and seller history corroborates it"
        elif recomputed is not None and stored is not None and recomputed < stored and behavioural == 0:
            verdict, reason = "FALSE_POSITIVE", "stored score exceeds what this history supports; no behavioural rule fires"
        else:
            verdict, reason = "INCONCLUSIVE", "signals do not settle the question"
        trace["reason"] = reason

        return AIMessage(content=json.dumps({
            "verdict": verdict, "confidence": 0.8, "risk_level": "medium",
            "summary": reason,
            "key_findings": [f"recomputed={recomputed} stored={stored}",
                             f"behavioural signals firing: {behavioural}"],
            "evidence": [{"source": "get_fraud_score_explanation",
                          "fact": f"engine recomputed {recomputed}",
                          "significance": reason}],
            "recommended_actions": ["Route per verdict."],
        }))

    llm = AsyncMock()
    llm.ainvoke = mock_ainvoke
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


async def main():
    import os

    import asyncpg
    from eval.data import load_golden
    from eval.guard import assert_no_contamination
    from eval.seed import seed_eval_fixtures
    from eval.runner import run_case, print_report, summarise

    # Same default as eval/runner.py — matches docker-compose.dev.yml.
    DSN = os.getenv("DATABASE_URL",
                    "postgresql://postgres:postgres@127.0.0.1:5432/payment_system")
    conn = await asyncpg.connect(DSN)
    mapping = await seed_eval_fixtures(conn)
    await conn.close()
    print("seeded:", mapping, "\n")

    cases = load_golden("eval/golden.jsonl")
    traces, results = {}, []

    from agent.graph import build_investigation_graph

    for case in cases:
        tx_id = mapping[case["scenario"]]
        trace = {"transaction_id": tx_id}
        traces[case["id"]] = trace
        judge = make_scripted_judge(trace)
        with patch("agent.nodes._get_llm", return_value=judge):
            graph = build_investigation_graph()
            results.append(await run_case(graph, case, tx_id))

    print_report(summarise(results))

    print("\n=== what the agent actually saw ===")
    for cid, t in traces.items():
        print(f"\n{cid}")
        print(f"  stored on Payout  : score={t.get('stored_score')} decision={t.get('stored_decision')}")
        print(f"  engine recomputed : score={t.get('recomputed_score')} decision={t.get('recomputed_decision')}")
        print(f"  history           : {t.get('history')}")
        print(f"  behavioural rules : {t.get('behavioural_signals')}")
        print(f"  conclusion        : {t.get('reason')}")


asyncio.run(main())
