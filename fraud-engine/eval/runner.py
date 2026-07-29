"""
eval/runner.py — run the golden dataset through the investigation graph.

Flow per case:
    seed fixtures -> resolve scenario -> graph.ainvoke (xN) -> contamination
    guard -> score against the expected verdict

The guard runs BEFORE scoring, so a leaked EVAL:: marker fails the run loudly
instead of quietly inflating accuracy.

Repeated runs
-------------
The agent is not deterministic: the same fixture can produce different verdicts
across invocations, so a single pass measures one sample, not the agent. With
--runs N each case is invoked N times and scored as the FRACTION of runs that
matched the expected verdict — 0.60 rather than FAIL. Overall accuracy is the
mean of those per-case fractions, so every case weighs the same regardless of
how many runs it took.

A fraction strictly between 0 and 1 means the agent answered differently on
identical input. That is a distinct failure mode from being consistently wrong —
it says the case sits on the edge of the model's decision boundary — so those
cases are marked UNSTABLE in the report and counted separately.

Runs execute SEQUENTIALLY, never concurrently. The fixtures are shared mutable
rows in one database and the graph writes an audit trail per invocation;
overlapping runs would interleave those writes and make the trail unreadable.
Wall-clock time is not the constraint being optimised here.

Retrieval audit
---------------
`find_similar_cases` returns historical cases WITH their `verdict` field, so a
near-duplicate in the corpus can hand the agent its answer. Every run records
which verdicts retrieval surfaced (`retrieval` in the per-case result) so an
accuracy number can always be checked against how much of it came from copying.
Across repeated runs the hits are aggregated with the number of runs that
surfaced them, since the agent may consult retrieval on some passes and not
others.

Usage:
    python -m eval.runner                       # seed + run everything once
    python -m eval.runner --runs 5              # 5 passes per case, fractional scores
    python -m eval.runner --no-seed             # reuse rows already in the DB
    python -m eval.runner --golden other.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

import asyncpg

from env_bootstrap import load_env

from eval.data import load_golden
from eval.guard import assert_no_contamination
from eval.seed import seed_eval_fixtures

# Must run before the os.getenv below, or .env values lose to the defaults.
load_env()

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

    `called` counts the tool MESSAGES, not the surfaced cases. Those are
    different facts and conflating them hid a real one: a call that returns zero
    matches used to be indistinguishable from never calling the tool at all, so
    "retrieval was not consulted" and "retrieval had nothing to say" read the
    same in the report. `empty` and `unparsed` keep them apart.
    """
    calls = 0
    empty = 0
    unparsed = 0
    surfaced: list[dict] = []

    for msg in state.get("messages", []) or []:
        name = getattr(msg, "name", None) or (
            msg.get("name") if isinstance(msg, dict) else None
        )
        if name != "find_similar_cases":
            continue

        # The tool ran — true regardless of what came back out of it.
        calls += 1

        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if not isinstance(content, str):
            unparsed += 1
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            unparsed += 1
            continue

        cases = payload.get("cases") or []
        if not cases:
            empty += 1
        for case in cases:
            surfaced.append({
                "case_id": case.get("case_id"),
                "verdict": case.get("verdict"),
                "similarity": case.get("similarity"),
            })

    return {
        "called": calls > 0,
        "calls": calls,
        "empty": empty,
        "unparsed": unparsed,
        "surfaced": surfaced,
    }


# ── Single case ─────────────────────────────────────────────────────────────

def _merge_retrieval(per_run: list[dict]) -> dict:
    """Fold per-run retrieval audits into one, keeping the run count per hit.

    The agent may call `find_similar_cases` on some passes and not others, so a
    hit that appears in 1 of 5 runs is a different fact from one that appears in
    5 of 5 — `runs` preserves that distinction.
    """
    counts: dict[tuple, dict] = {}
    runs_with_retrieval = 0
    runs_empty_handed = 0

    for audit in per_run:
        if audit.get("called"):
            runs_with_retrieval += 1
            # Called but nothing came back — reported separately from not calling.
            if not audit["surfaced"]:
                runs_empty_handed += 1
        for hit in audit["surfaced"]:
            key = (hit["case_id"], hit["verdict"], hit["similarity"])
            entry = counts.setdefault(key, {**hit, "runs": 0})
            entry["runs"] += 1

    return {
        "called": runs_with_retrieval > 0,
        "runs_with_retrieval": runs_with_retrieval,
        "runs_empty_handed": runs_empty_handed,
        "surfaced": sorted(
            counts.values(), key=lambda h: (-h["runs"], str(h["case_id"]))
        ),
    }


async def _invoke_once(graph, case: dict, transaction_id: int) -> tuple[str | None, dict]:
    """One graph invocation. Returns (verdict, per-run detail).

    A crashed run yields verdict None — it counts against the case's score
    rather than aborting the whole eval, since an agent that falls over on some
    passes and not others is exactly what repeated runs exist to expose.
    """
    try:
        state = await graph.ainvoke({
            "transaction_id": transaction_id,
            "trigger": case["trigger"],
        })
    except Exception as exc:  # noqa: BLE001 — a crashed run is a wrong run
        return None, {
            "verdict": None,
            "error": f"{type(exc).__name__}: {exc}",
            "iterations": None,
            "retrieval": {
                "called": False, "calls": 0, "empty": 0,
                "unparsed": 0, "surfaced": [],
            },
        }

    # Contamination check comes before scoring: a leaked marker invalidates
    # the result no matter what verdict came out. Deliberately NOT caught —
    # contamination is a broken fixture, not a bad answer, and must stop the run.
    assert_no_contamination(state, case_id=case["id"])

    verdict = (state.get("verdict") or {}).get("verdict")
    return verdict, {
        "verdict": verdict,
        "error": None,
        "iterations": state.get("iteration", 0),
        "retrieval": _audit_retrieval(state),
    }


async def run_case(graph, case: dict, transaction_id: int, runs: int = 1) -> dict:
    """Invoke the graph for one golden case `runs` times and score it.

    `score` is the fraction of runs whose verdict matched the expected one, so a
    case is no longer PASS/FAIL but a number in [0, 1]. With runs=1 that number
    is 0.0 or 1.0 and `passed` keeps its original boolean meaning.

    An unlabelled case (expected.verdict == UNLABELLED) is run and reported but
    left unscored: `score` and `passed` are None so it lands in neither the
    numerator nor the denominator of the accuracy figure. Its verdict spread is
    still recorded — that is the whole reason to run it.

    Runs are sequential by design; see the module docstring.
    """
    expected = case["expected"]["verdict"]
    unlabelled = case.get("unlabelled", False)

    verdicts: list[str | None] = []
    details: list[dict] = []
    for _ in range(runs):
        verdict, detail = await _invoke_once(graph, case, transaction_id)
        verdicts.append(verdict)
        details.append(detail)

    correct = sum(1 for v in verdicts if v == expected)
    score = None if unlabelled else correct / runs

    distribution: dict[str, int] = {}
    for v in verdicts:
        key = v or "ERROR"
        distribution[key] = distribution.get(key, 0) + 1

    errors = [d["error"] for d in details if d["error"]]
    iterations = [d["iterations"] for d in details if d["iterations"] is not None]

    return {
        "id": case["id"], "scenario": case["scenario"],
        "expected": expected,
        # Single most frequent verdict — the headline answer. Ties break on the
        # first seen, which with runs=1 is simply the only verdict there was.
        "actual": max(distribution, key=lambda k: distribution[k]) if distribution else None,
        "verdicts": verdicts,
        "distribution": distribution,
        "runs": runs,
        "correct": correct,
        "score": score,
        "passed": None if unlabelled else score == 1.0,
        "unstable": score is not None and 0.0 < score < 1.0,
        "unlabelled": unlabelled,
        "errors": errors,
        "iterations": iterations,
        "transaction_id": transaction_id,
        "retrieval": _merge_retrieval([d["retrieval"] for d in details]),
    }


# ── Full run ────────────────────────────────────────────────────────────────

async def run_eval(
    golden_path: str, dsn: str, do_seed: bool = True, runs: int = 1
) -> dict:
    if runs < 1:
        raise ValueError(f"--runs must be at least 1, got {runs}")

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

    # Sequential on purpose — see the module docstring.
    results = [
        await run_case(graph, case, mapping[case["scenario"]], runs=runs)
        for case in cases
    ]
    return summarise(results, runs=runs)


def summarise(results: list[dict], runs: int = 1) -> dict:
    """Accuracy over SCORED cases only — unlabelled ones are excluded from both
    the numerator and the denominator, and counted separately.

    Accuracy is the MEAN of the per-case fractions, not the fraction of all
    runs: averaging per case keeps every case equally weighted, so a case that
    happened to crash on one pass cannot drag the figure more than one case's
    worth. With runs=1 the two definitions coincide.

    `passed` counts only cases that were right on EVERY run, which keeps the
    exit code honest — a case that is right 4 times out of 5 has not passed.
    """
    scored = [r for r in results if r.get("score") is not None]
    passed = sum(1 for r in scored if r["score"] == 1.0)
    unstable = sum(1 for r in scored if r["unstable"])
    accuracy = sum(r["score"] for r in scored) / len(scored) if scored else 0.0

    return {
        "total": len(scored),
        "passed": passed,
        "unstable": unstable,
        "unlabelled": len(results) - len(scored),
        "runs": runs,
        "accuracy": round(accuracy, 4),
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

_VERDICT_ABBREV = {
    "TRUE_POSITIVE": "TP",
    "FALSE_POSITIVE": "FP",
    "INCONCLUSIVE": "INC",
    "ERROR": "ERR",
}


def _format_distribution(distribution: dict[str, int]) -> str:
    """`{'TRUE_POSITIVE': 3, 'INCONCLUSIVE': 2}` -> `TP x3, INC x2`, commonest
    first, so a split decision is visible at a glance."""
    ordered = sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(
        f"{_VERDICT_ABBREV.get(verdict, verdict)} x{count}" for verdict, count in ordered
    )


def _format_score(r: dict) -> str:
    """`0.60 (3/5)` for a scored case, an em dash for an unlabelled one."""
    if r["score"] is None:
        return "—"
    return f"{r['score']:.2f} ({r['correct']}/{r['runs']})"


def print_report(report: dict) -> None:
    runs = report.get("runs", 1)

    print(f"\n{'case':36s} {'expected':16s} {'score':12s} result")
    print("-" * 88)
    for r in report["results"]:
        if r["score"] is None:
            mark = "—    (unlabelled)"
        elif r["unstable"]:
            # Right on some runs, wrong on others: not a pass, and not the same
            # thing as a consistent failure either.
            mark = "UNSTABLE"
        elif r["score"] == 1.0:
            mark = "PASS"
        else:
            mark = "FAIL"

        print(f"{r['id']:36s} {r['expected']:16s} {_format_score(r):12s} {mark}")

        # With one run the distribution says nothing the columns do not; with
        # several it is the actual finding.
        if runs > 1:
            print(f"{'':36s}   verdicts: {_format_distribution(r['distribution'])}")
        elif r["score"] is None or r["score"] != 1.0:
            print(f"{'':36s}   verdict:  {r['actual'] or '—'}")

        for err in dict.fromkeys(r.get("errors") or []):
            print(f"{'':36s}   error:    {err}")

        # Retrieval: report whether it was consulted at all before reporting what
        # it returned. Silence used to mean either, which hid a dead tool.
        audit = r.get("retrieval") or {}
        consulted = audit.get("runs_with_retrieval", 0)
        if consulted == 0:
            print(f"{'':36s}   retrieval: not consulted ({runs}/{runs} runs)")
        else:
            note = f"consulted in {consulted}/{runs} runs"
            if audit.get("runs_empty_handed"):
                note += f", {audit['runs_empty_handed']} returned no matches"
            print(f"{'':36s}   retrieval: {note}")

        for hit in audit.get("surfaced", []):
            flag = " <-- matches expected" if hit["verdict"] == r["expected"] else ""
            seen = f" [{hit['runs']}/{runs} runs]" if runs > 1 else ""
            print(
                f"{'':36s}   retrieved {hit['case_id']} "
                f"({hit['verdict']}, sim={hit['similarity']}){seen}{flag}"
            )

    print("-" * 88)
    line = (
        f"passed {report['passed']}/{report['total']}  "
        f"accuracy {report['accuracy']}  ({runs} run(s) per case)"
    )
    if report.get("unstable"):
        line += f"  [{report['unstable']} unstable]"
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
    parser.add_argument(
        "--runs", type=int, default=1,
        help="invocations per case, run sequentially; each case scores as the "
             "fraction of runs matching the expected verdict (default: 1)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error(f"--runs must be at least 1, got {args.runs}")

    report = asyncio.run(
        run_eval(args.golden, args.dsn, do_seed=not args.no_seed, runs=args.runs)
    )
    print_report(report)
    # Only a clean sweep exits 0: anything unstable is a case the agent does not
    # answer reliably, which is a failure for CI purposes even at score 0.8.
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
