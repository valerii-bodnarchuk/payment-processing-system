"""
eval/data.py — golden-dataset loader + validation.

load_golden(path) reads a JSONL file where each non-blank line is one labelled
case:

    {"id": str,
     "scenario": str,
     "trigger": "BLOCK" | "REVIEW" | "MANUAL",
     "expected": {"verdict": "TRUE_POSITIVE" | "FALSE_POSITIVE" | "INCONCLUSIVE"}}

Cases are decoupled from the database: `scenario` is a logical key resolved to a
concrete transaction_id at seed time (see eval/seed.py), never a raw id.

Validation is strict — an invalid case fails loudly with file:line context so a
malformed golden set can never silently skew an eval run.

Unlabelled cases
----------------
`expected.verdict` may be the placeholder "UNLABELLED" for a case whose fixture
and arithmetic are settled but whose label has not been decided yet. Such a case
still seeds and still runs — so its measured behaviour can be inspected — but it
is NOT scored, and the runner keeps it out of the accuracy denominator. Assigning
a real verdict is a deliberate human act; a placeholder must never silently
count as a pass or a fail.
"""
from __future__ import annotations

import json
from pathlib import Path

# Kept in sync with the agent's verdict enum and the graph's trigger Literal
# (agent/state.py: trigger, and the synthesised verdict).
VALID_TRIGGERS = frozenset({"BLOCK", "REVIEW", "MANUAL"})
VALID_VERDICTS = frozenset({"TRUE_POSITIVE", "FALSE_POSITIVE", "INCONCLUSIVE"})

# Stand-in for "fixture is ready, label is not". Accepted by the loader, never
# scored by the runner. Deliberately not a member of VALID_VERDICTS so it can
# never be mistaken for a real label anywhere downstream.
PLACEHOLDER_VERDICT = "UNLABELLED"

_REQUIRED_KEYS = ("id", "scenario", "trigger", "expected")


def load_golden(path: str | Path) -> list[dict]:
    """Load and validate golden cases from a JSONL file.

    Blank lines are skipped. Raises ValueError (with file:line context) on any
    malformed or invalid case, and on an empty dataset.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"golden dataset not found: {p}")

    cases: list[dict] = []
    seen_ids: set[str] = set()

    with p.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {exc}") from exc

            _validate_case(case, p, lineno)

            case_id = case["id"]
            if case_id in seen_ids:
                raise ValueError(f"{p}:{lineno}: duplicate case id {case_id!r}")
            seen_ids.add(case_id)
            case["unlabelled"] = case["expected"]["verdict"] == PLACEHOLDER_VERDICT
            cases.append(case)

    if not cases:
        raise ValueError(f"{p}: no golden cases found (file empty or all blank)")

    return cases


def _validate_case(case: object, p: Path, lineno: int) -> None:
    where = f"{p}:{lineno}"

    if not isinstance(case, dict):
        raise ValueError(f"{where}: case must be a JSON object, got {type(case).__name__}")

    missing = [k for k in _REQUIRED_KEYS if k not in case]
    if missing:
        raise ValueError(f"{where}: missing required key(s): {', '.join(missing)}")

    if not isinstance(case["id"], str) or not case["id"].strip():
        raise ValueError(f"{where}: 'id' must be a non-empty string")

    if not isinstance(case["scenario"], str) or not case["scenario"].strip():
        raise ValueError(f"{where}: 'scenario' must be a non-empty string")

    trigger = case["trigger"]
    if trigger not in VALID_TRIGGERS:
        raise ValueError(
            f"{where}: invalid trigger {trigger!r}; "
            f"expected one of {sorted(VALID_TRIGGERS)}"
        )

    expected = case["expected"]
    if not isinstance(expected, dict) or "verdict" not in expected:
        raise ValueError(f"{where}: 'expected' must be an object with a 'verdict' key")

    verdict = expected["verdict"]
    if verdict not in VALID_VERDICTS and verdict != PLACEHOLDER_VERDICT:
        raise ValueError(
            f"{where}: invalid expected.verdict {verdict!r}; "
            f"expected one of {sorted(VALID_VERDICTS)} "
            f"or {PLACEHOLDER_VERDICT!r} for a case awaiting its label"
        )
