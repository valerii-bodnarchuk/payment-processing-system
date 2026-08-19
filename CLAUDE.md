# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit Authorship

Every commit must end with exactly this trailer — no variations, no model name suffix:

```
Co-Authored-By: Claude <noreply@anthropic.com>
```

The primary author is always the git user (`valerii.bodnarchuk <bodnarchukvalole@gmail.com>`), set by the local git config. Claude is the co-author. This ensures the contribution graph on GitHub shows commits as belonging to the primary author.

Do not copy the trailer format from `git log`: older commits carry a `Claude Sonnet 4.6` suffix and many carry no trailer at all. The rule above is the target, not the historical average.

## Commands

### NestJS Backend
```bash
npm run start:dev       # Dev server with hot reload
npm run build           # Compile TypeScript to dist/
npm run typecheck       # Type check without emit
npm test                # Run Jest tests (all suites, --runInBand --verbose)

# Targeted test runs
npx jest --testPathPatterns="ledger.service.spec" --verbose      # Ledger unit tests
npx jest --testPathPatterns="ledger.concurrency" --verbose       # Row-locking / overspend race
npx jest --testPathPatterns="payment-lifecycle" --verbose        # Integration suite
npx jest --testPathPatterns="investigation.service" --verbose    # NestJS investigation endpoints
npx jest --verbose                                               # All tests with detail
```

Ten spec files total: `ledger.service`, `ledger.concurrency`, `payout-state-machine`, `dispute-state-machine`, `reconciliation.service`, `admin.service`, `investigation.service`, `common/money`, `common/utils/money.util`, and `test/integration/payment-lifecycle`.

### Infrastructure
```bash
npm run docker:dev       # Start PostgreSQL (pgvector) + Redis only (for local dev)
npm run docker:dev:down  # Stop dev infra
npm run docker:up        # NestJS stack (app + postgres + redis + migrations)
npm run docker:down      # Stop stack
```

`docker:up` does **not** include the Python service. `Dockerfile` builds the NestJS app only; the fraud engine and the agent run locally via uvicorn.

### Database
```bash
npm run prisma:generate  # Regenerate Prisma client after schema changes
npm run prisma:migrate   # Run pending migrations
npm run prisma:seed      # Populate test data
```

### Fraud Engine + Agent (Python microservice)
```bash
cd fraud-engine
source venv/bin/activate

uvicorn app.main:app --reload --port 8000   # Rules API + agent router (if deps import)
pytest tests/                                # 12 test files; pytest.ini sets asyncio_mode=auto

python -m eval.runner                        # Golden dataset, seed + one pass per case
python -m eval.runner --runs 5               # N passes per case, fractional scores + instability
python -m eval.runner --no-seed              # Reuse rows already in the DB

python agent/mcp_server.py                   # MCP server over stdio
python scripts/index_investigations.py       # Index completed runs into the local RAG index
python scripts/backfill_embeddings.py        # Fill the Case table + pgvector embeddings
```

There is a second venv under `fraud-engine/notebooks/venv` for the ML notebook — unrelated to the service.

### Local dev setup
```bash
npm run docker:dev       # Start infra
cp .env.example .env    # Configure env — see Environment Variables, .env.example is not exhaustive
npm install && npm run prisma:generate
npm run prisma:migrate && npm run prisma:seed
npm run start:dev
```

Swagger docs: http://localhost:3000/api

## Architecture

Three parts: a **NestJS/TypeScript** payment processing backend, a **Python/FastAPI** rule-based fraud engine, and a **LangGraph** investigation agent that lives inside the same Python service and calls back into NestJS over HTTP.

### Core Design Principles
- **Double-entry ledger**: All money movements record balanced DEBIT/CREDIT entries via `LedgerService`. Every ledger operation is wrapped in a Prisma `$transaction` for atomicity.
- **Minor units**: All internal amounts are integer cents (`src/common/money.ts`). €1.00 = 100. Never float arithmetic on money; convert at the API boundary via `toMinorUnits` / `toMajorUnits`.
- **Idempotency**: Payments are deduplicated via Redis-cached `idempotencyKey` (24h TTL).
- **State machines**: Payouts (`PENDING → ELIGIBLE → PROCESSING → PAID/FAILED`, plus `REVERSED`) and Disputes (`OPEN → UNDER_REVIEW → WON/LOST/REFUNDED`) enforce transitions via `validateTransition()`.
- **Fail-open fraud**: If the fraud engine is unreachable, payouts default to `REVIEW` (not `BLOCK`) — `src/fraud/fraud.service.ts`.

### Ledger Account Types
`BUYER`, `SELLER`, `PLATFORM_FEE`, `ESCROW` — seller accounts have `allowNegative: true` to handle dispute losses.

### Payout Pipeline
1. `POST /payouts` → PENDING payout, ledger: ESCROW DEBIT → SELLER CREDIT (reserved)
2. `POST /payouts/:id/eligible` → calls fraud engine → ELIGIBLE if ALLOW
3. `POST /payouts/:id/process` → Stripe Transfer, books ESCROW DEBIT + SELLER CREDIT + PLATFORM_FEE CREDIT → PAID
4. `POST /payouts/:id/retry` → manual retry; automatic retry is max 3 attempts, exponential backoff via `withRetry()`

### Fraud Engine Integration
- `FraudService` in `src/fraud/` calls the Python service at `FRAUD_ENGINE_URL` (default: `http://localhost:8000`)
- Scores 0.0–1.0: `< 0.3` → ALLOW, `0.3–0.7` → REVIEW, `≥ 0.7` → BLOCK (`fraud-engine/config/rules.yaml`)
- Six rules: velocity, amount threshold, daily volume, failed history, new account, dispute rate
- `POST /check`, `POST /check/explain` (per-rule breakdown), `POST /outcomes`, `GET /outcomes/stats`

**Dormant ML branch**: `fraud-engine/app/ml/`, `notebooks/train_fraud_model.py` and the IEEE CSVs under `fraud-engine/data/` exist but are **not** imported by the scoring path. Live scoring is `rules.yaml` only. Do not assume the model is wired in.

### Async Queue
BullMQ + Redis (`src/queue/`): `PayoutQueue` adds jobs, `PayoutProcessor` executes them, `PayoutScheduler` triggers daily payouts via cron. `POST /queue/payout` enqueues manually. `@bull-board/*` is in `package.json` but **not mounted** — there is no admin UI route.

### Reconciliation
Hourly (24h window) and daily (all-time) reconciliation syncs internal payout/ledger state with Stripe. Detects orphaned Stripe transfers and ledger imbalances.

### Ledger Integrity
`LedgerService.verifyIntegrity()` runs three `$queryRaw` aggregate checks: global debit/credit totals, per-transaction balance (`GROUP BY / HAVING`), and orphaned entry scan. Returns `LedgerIntegrityReport`. Called by `ReconciliationService.reconcileLedger()` and exposed at `GET /ledger/integrity`. Logs `warn` on imbalance, `info` on clean pass.

### Dispute Loss Allocation
1. Payout not yet released → refund from escrow, no seller loss
2. Payout released, seller has balance → reverse payout (seller absorbs loss)
3. Payout released, seller withdrew → seller balance goes negative → `payoutsBlocked = true` automatically

## Investigation Agent (`fraud-engine/agent/`)

A LangGraph ReAct agent that investigates a flagged transaction and produces a structured verdict.

### Mounting
`app/main.py` imports `agent.api.router` inside a `try/except ImportError` and mounts it at `/investigate`. **A missing Python dependency silently disables the agent** and leaves `/check` working — if the routes 404, check the startup log line from the `agent` logger before debugging routing. There is no lifespan/startup hook; `agent.graph` is imported lazily inside the request handler.

### Graph (`agent/graph.py`)
```
start → collect → reason ⇄ tools → synthesize → audit → END
```
Six nodes: `start`, `collect`, `reason`, `tools` (prebuilt `ToolNode`), `synthesize`, `audit`. The `reason ⇄ tools` loop is ReAct; `_route_after_reason` sends flow to `synthesize` on tool-free responses, on `INVESTIGATION_COMPLETE`, or when `iteration >= MAX_ITERATIONS` (8, `agent/config.py`). Compiled once into the module-level `investigation_graph` singleton.

### Node contract
Every node is `async def node(state: InvestigationState) -> dict` and returns a **partial** delta, not the full state. Only `messages` has a reducer (`add_messages`); `audit_trail` is appended manually via `state.get("audit_trail", []) + [entry]`.

### State (`agent/state.py`)
`InvestigationState` is a `TypedDict, total=False`. Inputs: `transaction_id`, `trigger`. Collected: `transaction_data`, `seller_profile`, `payout_timeline`, `fraud_score_detail`, `ledger_check`, `similar_cases`. Loop: `messages`, `iteration`. Output: `verdict`, `audit_trail`, `degraded`, `degradation_reason`, `run_id`.

`run_id` only exists after `audit_node` (it is the persisted `InvestigationRun.id`) and is `None` when persistence is skipped or fails. `transaction_id` is the only identifier available for the whole run.

### Degradation contract
A degraded verdict is syntactically valid but was not reasoned to. Consumers must not infer this from confidence or parse it out of the summary. Three codes in `agent/nodes.py`:

| Code | Raised where | Transient |
|------|--------------|-----------|
| `LLM_UNAVAILABLE` | provider call raised in `reason_node` or `synthesize_node` | yes |
| `OUTPUT_UNPARSEABLE` | synthesis returned non-JSON | no |
| `ITERATIONS_EXHAUSTED` | cap hit, model never signalled completion | no |

Degradation is **sticky and earliest-wins**: a successful synthesis on partial context does not clear a flag set by a failed reasoning leg. The marker is written in three places — state fields, the `audit_trail` entry, and the verdict payload itself (so it survives being read straight out of `InvestigationRun.verdictPayload`). It surfaces on the API as `degraded` / `degradation_reason` with HTTP 200. When adding a new failure path, set all three.

### LLM providers (`agent/llm.py`)
`get_chat_model()` is the single factory, selected by `AGENT_LLM_PROVIDER` (`openai` | `bedrock`). Two rules the module exists to keep: no provider SDK is imported at module scope, and no client is constructed at import time — `agent/mcp_server.py` must start with no credentials present. Bedrock uses `ChatBedrockConverse` with a cross-region inference profile ID (plain model IDs fail with `ValidationException` in EU regions); boto3 resolves credentials from its own chain, so nothing AWS-shaped belongs in `.env`.

`agent/nodes.py::_get_llm()` is a thin delegator kept as the patch target for the test suite and `eval/run_mocked.py`. Patch that, not `get_chat_model`.

**Message content shapes**: ChatOpenAI returns `content` as a string, ChatBedrockConverse as a list of blocks — and the same provider returns both depending on model and reasoning settings. Always flatten via `_message_text()` before doing string work on `content`. Substring checks against a raw list silently never match; this has been the cause of two separate bug fixes.

### Tools (`agent/tools/registry.py`)
Six tools in `ALL_TOOLS`: `get_transaction_context`, `get_seller_risk_profile`, `get_payout_timeline`, `get_fraud_score_explanation`, `check_ledger_consistency`, `find_similar_cases`. Failures surface as an `{"error": True, ...}` dict rather than an exception — either from the tool's own `try/except` or nested inside the payload from `nestjs_get` (`check_ledger_consistency` returns partial data with per-call error dicts). A tool failure therefore degrades the answer instead of killing the graph, which also means the LLM can reason over an error dict without noticing it is one.

### `excludePayoutId` invariant
`collect_node` fetches the risk profile with `excludePayoutId` set to the payout under investigation. This is **correctness, not optimisation**: the engine's `daily_volume` rule computes `seller_total_amount_24h + amount`, adding the scored payout back in itself. Feeding it a window total that already contains that payout double-counts it, inflates the recomputed score above the stored one, and makes a false positive look corroborated. `get_seller_risk_profile` has the same parameter, but the agent never calls it — the profile is already in its first message, so fixing only the tool leaves the live path broken.

### Retrieval (`agent/rag/`)
The live `find_similar_cases` ranks over the **6 static `SEED_CASES`** using deterministic signal overlap (`agent/rag/store.py`) — no network, no embeddings. The pgvector path (`Case` table, `vector(384)`, `BAAI/bge-small-en-v1.5` via sentence-transformers, `scripts/backfill_embeddings.py`) is built but is not what the tool queries. Retrieval must stay **optional** in the prompt until the corpus is verdict-balanced — making it mandatory measurably regressed eval accuracy. See `docs/retrieval_status.md` and `docs/agent_known_limitations.md`.

### Persistence (`agent/persistence/audit.py`)
`audit_node` writes one `InvestigationRun` plus ordered `InvestigationAuditEntry` rows via asyncpg, in a transaction. Best-effort by design: it never raises into the graph, returns a status dict, and yields `run_id: None` when `DATABASE_URL` is unset or the write fails. Callers must tolerate a missing `run_id` rather than treat it as a broken run.

### Streaming and MCP
- `POST /investigate/stream` — SSE, one `node` event per LangGraph delta plus `start`/`done`/`error` (`agent/streaming.py`). Additive observability; persistence is unchanged. Note the wire summary currently projects the verdict down to `verdict/confidence/risk_level/summary` and does **not** carry the degradation flags.
- `agent/mcp_server.py` — FastMCP server exposing the `investigate_transaction` tool and an `investigation://{run_key}/audit` resource. Runs over stdio by default.

### Eval harness (`fraud-engine/eval/`)
`python -m eval.runner` seeds fixtures, runs the golden set through the real graph, and scores against expected verdicts. A contamination guard runs **before** scoring so a leaked `EVAL::` marker fails loudly instead of inflating accuracy. `--runs N` invokes each case N times sequentially and scores the fraction that matched; a fraction strictly between 0 and 1 marks the case UNSTABLE, which is a distinct failure mode from being consistently wrong. Every run records which verdicts retrieval surfaced, so accuracy can be checked against how much of it came from copying. Exit 0 only on a clean sweep.

### NestJS endpoints the agent depends on
| Endpoint | Used by |
|----------|---------|
| `GET/POST /investigate/transaction/:id` | `collect_node`, `get_transaction_context` |
| `GET/POST /investigate/payout/:id` | deterministic root-cause report |
| `GET /admin/sellers/:id/risk-profile?excludePayoutId=` | `collect_node`, `get_seller_risk_profile` |
| `GET /admin/sellers/:id/payout-timeline` | `collect_node`, `get_payout_timeline` |

Changing the shape of these responses breaks the agent silently — the tools return `{"error": True}` and the verdict degrades rather than failing.

## Observability
- **NestJS**: `nestjs-pino` structured logging, request IDs via `genReqId` (honours inbound `x-request-id`, else `randomUUID`). Prometheus metrics at `GET /metrics` (`prom-client`), liveness at `GET /health`, global `ThrottlerGuard`.
- **Python**: stdlib `logging` only, per-module loggers (`agent.nodes`, `agent.llm`, …). No structured logging and no log config in the app, so anything below WARNING from a non-uvicorn logger is dropped by default.
- **Python tracing** (`agent/telemetry.py`): OTel SDK is bootstrapped from the FastAPI lifespan — `TracerProvider` + `BatchSpanProcessor` + OTLP/gRPC to `OTEL_EXPORTER_OTLP_ENDPOINT`, `service.name` from `OTEL_SERVICE_NAME`, disabled wholesale by `OTEL_SDK_DISABLED`. `FastAPIInstrumentor` covers HTTP; `/health` is excluded. Jaeger runs in `docker-compose.dev.yml` (UI on 16686). Import `tracer` and `current_trace_id()` from that module — both are safe before setup runs and when tracing is off. An unreachable collector drops spans and never fails a request.
  Graph nodes and LLM calls are **not yet instrumented** — that is deliberate and comes in later changes, along with putting the trace id into log lines. Until then there is still **no correlation ID in the logs**.

### Key Modules
| Module | Path | Purpose |
|--------|------|---------|
| Ledger | `src/ledger/` | Double-entry bookkeeping engine |
| Payment | `src/payment/` | Stripe PaymentIntent + escrow entry |
| Payout | `src/payout/` | Full payout lifecycle + retry logic |
| Fraud | `src/fraud/` | HTTP client to Python fraud engine |
| Dispute | `src/dispute/` | Chargeback handling + reversal |
| Seller | `src/seller/` | Stripe Connect KYC + account management |
| Webhook | `src/webhook/` | Stripe event processing |
| Queue | `src/queue/` | BullMQ async payout processing |
| Reconciliation | `src/reconciliation/` | Stripe/ledger sync |
| Admin | `src/admin/` | Ops endpoints + seller risk profile / payout timeline (agent data sources) |
| Investigation | `src/investigation/` | Deterministic root-cause reports; the agent's context source |
| Metrics | `src/metrics/` | Prometheus business metrics |
| Health | `src/health/` | Liveness probe |
| Idempotency | `src/idempotency/` | Redis-backed idempotency keys |
| Common | `src/common/` | Money helpers, logger type, guards, decorators |
| Redis / Stripe / Prisma | `src/redis/`, `src/stripe/`, `src/prisma/` | Client wrappers |

### Prisma Models
Payments: `Account`, `Transaction`, `Entry`, `Payout`, `Seller`, `Dispute`. Agent: `InvestigationRun`, `InvestigationAuditEntry` (append-only audit rows), `Case` (RAG corpus with a `vector(384)` embedding column). PostgreSQL runs the `pgvector/pgvector:pg16` image everywhere — dev compose, full compose, and CI.

### Environment Variables
`.env.example` is incomplete — treat this table as the reference.

| Variable | Consumer | Default |
|----------|----------|---------|
| `DATABASE_URL` | NestJS/Prisma + agent persistence + eval | — |
| `REDIS_HOST` / `REDIS_PORT` (or `REDIS_URL` in prod) | NestJS | localhost:6379 |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | NestJS | — |
| `FRAUD_ENGINE_URL` | NestJS `FraudService` | `http://localhost:8000` |
| `LOG_LEVEL`, `PORT` | NestJS | `info`, `3000` |
| `NESTJS_BASE_URL` | agent tools | `http://localhost:3000` |
| `AGENT_LLM_PROVIDER` | `agent/llm.py` | `openai` |
| `AGENT_LLM_MODEL` (falls back to `OPENAI_MODEL`) | OpenAI branch | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI branch | — |
| `AGENT_BEDROCK_MODEL_ID` | Bedrock branch | `eu.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `AGENT_BEDROCK_REGION` | Bedrock branch | `eu-central-1` |
| `RAG_INDEX_PATH` | `agent/rag/store.py`, `indexer.py` | `fraud-engine/data/indexed_cases.json` (absent until generated) |

Python entrypoints load `.env` via `fraud-engine/env_bootstrap.py`. Precedence: real process env → `fraud-engine/.env` → repo-root `.env`. Call `load_env()` before any module-level `os.getenv()`.

### CI Pipeline (`.github/workflows/ci.yml`)
Four jobs:
1. `lint` — `prisma generate` + `tsc --noEmit`
2. `test` (needs lint) — Jest against real pgvector + Redis service containers; `sk_test_fake` / `whsec_fake`
3. `fraud-engine-test` (needs lint) — Python 3.13, offline subset only: `test_rules`, `test_api`, `test_agent_tools`, `test_agent_graph`, `test_agent_rag`. The other seven Python test files are **not** run in CI.
4. `docker` (needs test) — image build validation, gated to pushes on `main`, so it never runs on PRs

### Repo layout notes
Root `eval/` and `fraud-engine/evals/` are vestigial and empty — the real harness is `fraud-engine/eval/`. `.gitignore` does not cover `venv/`, `__pycache__/`, or `*.pyc`; they are currently untracked but unprotected, so check `git status` before staging in `fraud-engine/`.

## Financial Invariants (Non-Negotiable)
- NEVER run destructive SQL (DROP, TRUNCATE, DELETE without WHERE)
- NEVER run migrations without explicit confirmation
- ALL financial state transitions must be inside Prisma $transaction
- Ledger entries are IMMUTABLE — never update, only append. The same holds for `InvestigationAuditEntry`.
- Money is integer minor units end to end — no floats, no implicit major/minor conversions
- Idempotency required on all payment/payout mutations
- A degraded verdict must never be presentable as a reasoned one — carry `degraded` + `degradation_reason` through every new path
- Correctness > simplicity for money-touching code

## Portfolio Context
- Target: senior fintech backend roles, DACH/UK, €100k+
- Code quality should reflect senior-level architectural decisions
- Every design decision should be defensible in a technical interview
