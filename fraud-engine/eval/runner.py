"""
eval/runner.py — run the golden dataset through the investigation graph.

Flow per case:
    seed fixtures -> resolve scenario -> graph.ainvoke -> contamination guard
    -> score against the expected verdict

The guard runs BEFORE scoring, so a leaked EVAL:: marker fails the run loudly
instead of quietly inflating accuracy.

Retrieval audit
---------------
`find_similar_cases` returns historical cases WITH their `verdict` field, so a
near-duplicate in the corpus can hand the agent its answer. Every run records
which verdicts retrieval surfaced (`retrieval` in the per-case result) so an
accuracy number can always be checked against how much of it came from copying.

Usage:
    python -m eval.runner                       # seed + run everything
    python -m eval.runner --no-seed             # reuse rows already in the DB
    python -m eval.runner --golden other.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

import asyncpg

from eval.data import load_golden
from eval.guard import assert_no_contamination
from eval.seed import seed_eval_fixtures

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/payment_system",
)
DEFAULT_GOLDEN = "eval/golden.jsonl"


# ── Retrieval audit ─────────────────────────────────────────────────────────

def _audit_retrieval(state: dict) -> dict:
    """Which verdicts `find_similar_cases` put in front of the agent.

    Advisory-only per the tool's own docstring, but a retrieved verdict that
    matches the expected label means the case may have been copied rather than
    investigated — worth seeing next to every score.
    """
    surfaced: list[dict] = []
    for msg in state.get("messages", []) or []:
        name = getattr(msg, "name", None) or (
            msg.get("name") if isinstance(msg, dict) else None
        )
        if name != "find_similar_cases":
            continue
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        for case in payload.get("cases", []):
            surfaced.append({
                "case_id": case.get("case_id"),
                "verdict": case.get("verdict"),
                "similarity": case.get("similarity"),
            })
    return {"called": bool(surfaced), "surfaced": surfaced}


# ── Single case ─────────────────────────────────────────────────────────────

async def run_case(graph, case: dict, transaction_id: int) -> dict:
    """Invoke the graph for one golden case and score it.

    An unlabelled case (expected.verdict == UNLABELLED) is run and reported but
    left unscored: `passed` is None so it lands in neither the numerator nor the
    denominator of the accuracy figure.
    """
    expected = case["expected"]["verdict"]
    unlabelled = case.get("unlabelled", False)

    try:
        state = await graph.ainvoke({
            "transaction_id": transaction_id,
            "trigger": case["trigger"],
        })
    except Exception as exc:  # noqa: BLE001 — a crashed case is a failed case
        return {
            "id": case["id"], "scenario": case["scenario"],
            "expected": expected, "actual": None,
            "passed": None if unlabelled else False,
            "unlabelled": unlabelled,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # Contamination check comes before scoring: a leaked marker invalidates
    # the result no matter what verdict came out.
    assert_no_contamination(state, case_id=case["id"])

    verdict = (state.get("verdict") or {}).get("verdict")
    return {
        "id": case["id"], "scenario": case["scenario"],
        "expected": expected, "actual": verdict,
        "passed": None if unlabelled else verdict == expected,
        "unlabelled": unlabelled,
        "iterations": state.get("iteration", 0),
        "transaction_id": transaction_id,
        "retrieval": _audit_retrieval(state),
    }


# ── Full run ────────────────────────────────────────────────────────────────

async def run_eval(golden_path: str, dsn: str, do_seed: bool = True) -> dict:
    cases = load_golden(golden_path)

    conn = await asyncpg.connect(dsn)
    try:
        if do_seed:
            mapping = await seed_eval_fixtures(conn)
        else:
            mapping = await _resolve_existing(conn)
    finally:
        await conn.close()

    missing = sorted({c["scenario"] for c in cases} - mapping.keys())
    if missing:
        raise ValueError(
            f"No seeded transaction for scenario(s): {', '.join(missing)}. "
            "Register a builder in eval/seed.py::SCENARIO_BUILDERS."
        )

    # Imported late so the module can be loaded without LangGraph present.
    from agent.graph import build_investigation_graph
    graph = build_investigation_graph()

    results = [
        await run_case(graph, case, mapping[case["scenario"]]) for case in cases
    ]
    return summarise(results)


def summarise(results: list[dict]) -> dict:
    """Accuracy over SCORED cases only — unlabelled ones are excluded from both
    the numerator and the denominator, and counted separately."""
    scored = [r for r in results if r.get("passed") is not None]
    passed = sum(1 for r in scored if r["passed"])
    return {
        "total": len(scored),
        "passed": passed,
        "unlabelled": len(results) - len(scored),
        "accuracy": round(passed / len(scored), 4) if scored else 0.0,
        "results": results,
    }


async def _resolve_existing(conn: asyncpg.Connection) -> dict[str, int]:
    """Map scenarios to already-seeded rows via the marker, newest first.

    Only usable with --no-seed; the marker is opaque, so this reverses it
    through the same Python-side hash the seeder uses.
    """
    from eval.seed import SCENARIO_BUILDERS, marker_for

    mapping: dict[str, int] = {}
    for scenario_key in SCENARIO_BUILDERS:
        tx_id = await conn.fetchval(
            'SELECT id FROM "Transaction" WHERE description = $1 ORDER BY id DESC LIMIT 1',
            f"{marker_for(scenario_key)} main",
        )
        if tx_id is not None:
            mapping[scenario_key] = tx_id
    return mapping


# ── Reporting ───────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    print(f"\n{'case':36s} {'expected':16s} {'actual':16s} result")
    print("-" * 82)
    for r in report["results"]:
        mark = "—    (unlabelled)" if r["passed"] is None else (
            "PASS" if r["passed"] else "FAIL"
        )
        actual = r["actual"] or r.get("error", "—")
        print(f"{r['id']:36s} {r['expected']:16s} {str(actual):16s} {mark}")

        audit = r.get("retrieval") or {}
        for hit in audit.get("surfaced", []):
            flag = " <-- matches expected" if hit["verdict"] == r["expected"] else ""
            print(
                f"{'':36s}   retrieved {hit['case_id']} "
                f"({hit['verdict']}, sim={hit['similarity']}){flag}"
            )

    print("-" * 82)
    line = f"passed {report['passed']}/{report['total']}  accuracy {report['accuracy']}"
    if report.get("unlabelled"):
        line += f"  ({report['unlabelled']} unlabelled, not scored)"
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden dataset.")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--dsn", default=DATABASE_URL)
    parser.add_argument(
        "--no-seed", action="store_true",
        help="reuse rows already in the DB instead of re-seeding",
    )
    args = parser.parse_args()

    report = asyncio.run(run_eval(args.golden, args.dsn, do_seed=not args.no_seed))
    print_report(report)
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
