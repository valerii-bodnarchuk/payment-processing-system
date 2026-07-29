# Similar-Cases Retrieval — Status

> Agent-facing constraint: retrieval must stay OPTIONAL in the investigation
> prompt until the corpus is balanced by verdict. The live tool ranks over the 6
> `SEED_CASES` only (4 of them TRUE_POSITIVE), and making the call mandatory
> measurably regressed eval accuracy. See
> [agent_known_limitations.md](agent_known_limitations.md) for the numbers and
> the exit condition.

## What landed

**pgvector infrastructure** (`8646bce`): PostgreSQL `vector(384)` column on the
`Case` table, pgvector extension enabled in Compose and CI, asyncpg vector
registration in the backfill path.

**Local embeddings** (`8646bce`): `BAAI/bge-small-en-v1.5` (384-d) via
`sentence-transformers`, normalised vectors, single `_canonical_text` projection
(`verdict | risk | signals | summary`) shared by seed and synthetic sources to
prevent systematic embedding bias between case origins.

**Backfill script** (`fraud-engine/scripts/backfill_embeddings.py`, `fd150f3`):
idempotent, keyed on `caseId`; handles `seed`, `synthetic`, and
`investigation_run` sources in one pass.

**Corpus — 46 cases total:**
- 6 seed cases (`source=seed`) — static production anchors
- 20 Phase 1 synthetic fixtures (`source=synthetic`, SYN-001..020) — 4 cases
  across 5 intent clusters, used to establish retrieval baselines
- 20 Phase 2a boundary cases (`source=synthetic`, SYN-021..040) — 5 cases per
  cross-cluster boundary type to stress the retriever with hard negatives and
  ambiguous overlaps

**Phase 2a sanity check** (`fraud-engine/scripts/phase2a_analysis.py`):
pairwise cosine similarity analysis on all 40 synthetic cases.

## What's pending

- `similar_cases.py` rewrite: replace Jaccard signal-overlap ranking with
  pgvector cosine ANN retrieval (`<=>` operator); query-time embedding of the
  `SimilarCaseQuery` input via `embed_one`.
- Eval script: precision@k, recall@k, MRR against a labelled query set derived
  from the 40 synthetic fixtures.
- Eval report: baseline Jaccard vs. vector retrieval comparison.

## Key finding from Phase 2a sanity check

The vector space discriminates fraud patterns as expected: mean intra-cluster
cosine similarity is 0.8647 vs. mean cross-cluster 0.7619 — a clear separation.
However, it does **not** separate verdicts when signals overlap: Type C boundary
cases (geo mismatch on legitimate VIP travel, `verdict=FALSE_POSITIVE`) and Type D
(merchant collusion vs. mass friendly fraud) retrieve their opposite-verdict
neighbours at 0.88–0.96 similarity. The working hypothesis is that verdict is
under-weighted in `_canonical_text` relative to the signal list — signals are a
dense comma-joined token sequence that dominates the embedding, while the verdict
prefix is a single token. Next investigation: reweight the canonical projection
(e.g. repeat the verdict token, or move it into the summary prefix) and re-run
the Phase 2a similarity check.

## Known coverage gap

SYN-009 (account-takeover core) appears in 7 of the 10 least-similar pairs —
cluster 3 is the most semantically isolated cluster in the corpus. Future
fixture work should add cluster-3-adjacent cases (credential-stuffing on
moderate-value accounts, ATO with partial device match) before running the
full precision/recall eval.
