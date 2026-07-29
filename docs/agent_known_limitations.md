# Investigation Agent — Known Limitations

Constraints found by measurement, each with the evidence that established it and
the condition under which it stops being a constraint. Both entries below are
about what the agent can and cannot be asked to do TODAY — neither is a bug
awaiting a patch, and both were reached by reverting or declining a change that
looked like an improvement.

## 1. Retrieval cannot be made a mandatory investigation step

`find_similar_cases` must stay OPTIONAL in `agent/prompts.py` until the
precedent corpus is balanced across verdicts.

**Why.** The live retrieval path ranks over `agent/rag/cases.py::SEED_CASES` and
nothing else — `LocalCaseStore` also loads `data/indexed_cases.json`, but that
file is not present, so the corpus the tool actually searches is 6 cases:

| verdict | count |
|---|---|
| TRUE_POSITIVE | 4 |
| FALSE_POSITIVE | 1 |
| INCONCLUSIVE | 1 |

Ranking is Jaccard overlap on signal sets (`decision:*`, `risk:*`, `rule:*`), so
any BLOCK-band query matches TRUE_POSITIVE precedents first almost by
construction. The 40 synthetic fixtures described in `retrieval_status.md` are
embedded in pgvector but are NOT wired into this tool yet; they do not mitigate
the imbalance on the live path.

**Evidence.** The prompt originally gated retrieval behind "after collecting
fraud-score context, you **may** call" while also instructing "do not call any
more tools after you have enough information, be decisive". Those two conflict
at exactly the step where retrieval becomes permitted, and the result was 0
calls in 30 eval runs. Making retrieval a required step fixed the call rate
(30/30) and regressed accuracy, measured over 5 runs per case:

| case | retrieval optional | retrieval mandatory |
|---|---|---|
| 001 fraud_confirmed (TP) | TP x4, INC x1 | TP x5 |
| 002 high_score_clean_history (FP) | TP x4, **FP x1** | TP x5 |
| 003 pending_burst (unlabelled) | TP x5 | TP x5 |
| 004 queued_backlog (INCONCLUSIVE) | **INC x5** | FP x3, INC x1, TP x1 |
| 005 thin_history (INCONCLUSIVE) | **INC x5** | INC x3, TP x2 |
| 006 clean_seller_burst (FP) | INC x3, FP x2 | FP x2, TP x2, INC x1 |

Two cases that were stable at 5/5 correct broke, and 002 lost its only
FALSE_POSITIVE. On 001, 002, 003 and 006 every precedent surfaced was a
TRUE_POSITIVE — three per case — and on 002 the agent named them in its
justification: *"significant risk factors, supported by similar historical
cases"*.

**Caveat on that comparison.** The mandatory-retrieval run also carried the
`collect_node` volume fix, so the two columns differ by two changes, not one.
Attribution to retrieval alone rests on the surfaced-verdict evidence above, not
on the table. A single-variable run has not been done.

**Retained from the attempt.** The calibration paragraph in the `## Similar
Cases` section stays: retrieved verdicts are never answers, matching AND
diverging signals must be named, and a precedent differing on the signal that
carried its verdict must be discounted. That guidance is correct whether or not
the call is mandatory.

**Exit condition.** Balance the corpus by verdict — enough FALSE_POSITIVE and
INCONCLUSIVE precedents that signal overlap no longer implies a TRUE_POSITIVE
neighbour — or wire in the synthetic corpus and confirm the retrieved-verdict
mix is no longer skewed. Then re-run `--runs 5` with retrieval mandatory and
compare against the optional baseline, one variable at a time.

## 2. Seller history older than ~30 days never reaches the model

The agent cannot weigh long-range history, so no fixture or investigation should
depend on it.

**Why.** Two independent cuts in `agent/nodes.py::collect_node`:

- `/admin/sellers/:id/payout-timeline` is called without `daysBack`, taking the
  endpoint default of 30 days. Anything older is not in the response at all.
- The timeline JSON is inlined into the first message as
  `json.dumps(...)[:2000]`. The endpoint orders newest-first, so the clip drops
  the OLDEST entries — precisely the historical baseline.

**Evidence.** Measured against the six eval fixtures:

| fixture | timeline JSON | payouts in window | visible after `[:2000]` |
|---|---|---|---|
| 001 Lumen Trade Supply | 2140 chars | 6 | 6 |
| 002 Ottokar Grienau | 496 chars | 1 | 1 |
| 003 Nordvik Outdoor | 2916 chars | 8 | 6 |
| 004 Halvorsen Marine | 2284 chars | 6 | 6 |
| 005 Cassia Botanicals | 494 chars | 1 | 1 |
| 006 Bergqvist Antikvariat | 4080 chars | 12 | 7 |

`clean_seller_volume_burst` (006) was built with a comparable 8-payout day 31
days back, as the counter-argument to a velocity/daily_volume BLOCK. It is
unreachable twice over: outside the 30-day lookback, and behind the clip even if
the lookback were widened. The case still discriminates, but on risk-profile
aggregates only (`accountAgeDays`, `totalDisputes`, `failedPayouts`,
`avgPayoutAmount`, `totalVolumeLifetime`) — those are ~875 chars and survive
their own 2000-char cut intact.

**Not fixed deliberately.** Raising the clip or passing a longer `daysBack`
changes inlined context size, and therefore token cost, on every investigation.
That is a budget decision, not a defect fix. Moving the 006 fixture inside the
lookback alone accomplishes nothing while the clip stands.

**Exit condition.** Decide the context budget. If long-range history is wanted,
the sound version is probably not a bigger clip but a summarised timeline —
per-period volume and status counts rather than raw payout rows — so the
aggregate survives truncation by being small.
