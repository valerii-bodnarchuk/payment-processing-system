# Fraud Engine

Rule-based fraud scoring microservice. Scores seller payout requests 0.0–1.0 and returns ALLOW / REVIEW / BLOCK decisions.

**Decision boundaries:** `< 0.3` → ALLOW &nbsp;·&nbsp; `0.3–0.7` → REVIEW &nbsp;·&nbsp; `≥ 0.7` → BLOCK

## Stack

Python 3.13 · FastAPI · Pydantic · PyYAML

## Quick Start

```bash
cd fraud-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Swagger: http://localhost:8000/docs

## Scoring Model

Additive weighted scoring across 6 independent rules, capped at 1.0:

```
total_score = min(Σ (rule.score × rule.weight), 1.0)
```

All weights default to `1.0`. Increase a weight to amplify a rule's impact without touching code — edit `config/rules.yaml` instead.

## Rule Thresholds (`config/rules.yaml`)

| Rule | Trigger | Score |
|------|---------|-------|
| **velocity** | ≥ 5 payouts in 24h | +0.20 |
| **velocity** | ≥ 10 payouts in 24h | +0.40 |
| **amount_threshold** | single payout ≥ €5,000 | +0.20 |
| **amount_threshold** | single payout ≥ €10,000 | +0.50 |
| **daily_volume** | 24h total ≥ €20,000 | +0.15 |
| **daily_volume** | 24h total ≥ €50,000 | +0.40 |
| **failed_history** | ≥ 2 failed payouts in 7d | +0.20 |
| **failed_history** | ≥ 5 failed payouts in 7d | +0.50 |
| **new_account** | account age < 30 days | +0.05 |
| **new_account** | account age < 7 days | +0.15 |
| **dispute_rate** | ≥ 1 dispute on record | +0.15 |
| **dispute_rate** | ≥ 3 disputes on record | +0.50 |

To tune thresholds or weights, edit `config/rules.yaml` — no code changes needed. The config is loaded once at startup.

## API

### `POST /check`
Standard fraud check. Returns decision + explanation.

```json
// Request
{
  "transaction_id": 1,
  "seller_id": 42,
  "amount": 7500.0,
  "seller_payout_count_24h": 3,
  "seller_total_amount_24h": 12000.0,
  "seller_failed_payouts_7d": 1,
  "seller_account_age_days": 45,
  "seller_dispute_count": 0
}

// Response
{
  "transaction_id": 1,
  "risk_score": 0.2,
  "decision": "ALLOW",
  "rules_triggered": [],
  "explanation": "Transaction passed all checks (score: 0.2)"
}
```

### `POST /check/explain`
Same as `/check` but returns the full rule-by-rule breakdown — used by investigation agents.

Additional response fields:
- `all_rules` — all 6 rules including untriggered ones
- `score_breakdown` — `{ rule_name: weighted_score }` per rule
- `config_version` — the version string from `rules.yaml`

### `POST /outcomes`
Record whether a fraud decision was correct (feedback loop).

```json
{
  "transaction_id": 1,
  "original_decision": "REVIEW",
  "actual_outcome": "legitimate"  // or "fraudulent", "disputed", "chargeback"
}
```

### `GET /outcomes/stats`
Returns precision/recall stats across all reported outcomes.

```json
{
  "total_decisions": 150,
  "outcomes_reported": 42,
  "false_positives": 5,
  "false_negatives": 2,
  "precision": 0.8571,
  "recall": 0.9375
}
```

`precision` and `recall` are `null` until enough outcomes are reported to compute them.
Outcomes are stored in memory — not persisted across restarts.

## MCP server

The investigation agent is also exposed over MCP, so other agents (Cursor, Claude
Desktop, any MCP host) can call it as a tool.

```bash
cd fraud-engine
source venv/bin/activate
python -m agent.mcp_server          # stdio
```

Client config — `~/.cursor/mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fraud-investigation": {
      "command": "/absolute/path/to/fraud-engine/venv/bin/python",
      "args": ["-m", "agent.mcp_server"],
      "cwd": "/absolute/path/to/fraud-engine",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "NESTJS_BASE_URL": "http://localhost:3000",
        "DATABASE_URL": "postgresql://..."
      }
    }
  }
}
```

The server starts without `OPENAI_API_KEY` — the graph is imported lazily on the
first tool call — but an investigation will fail without one.

### Why the tool contract is not 1:1 with `POST /investigate`

The REST endpoint serves a human and returns the full audit trail inline; an MCP
host pays context tokens for every byte a tool returns, so the tool returns a
compact verdict and parks the trail behind the `investigation://{run_key}/audit`
resource, which the caller fetches only when it needs the evidence. And because
agents retry on their own initiative — where one investigation costs several LLM
round trips — calls are idempotent by `idempotency_key`: a repeat returns the
first result with `replayed: true`, and a call arriving mid-run joins the
in-flight one rather than starting a second.

| Surface | Returns | Idempotent |
|---------|---------|------------|
| `POST /investigate` | verdict + full audit trail | no |
| `investigate_transaction` (MCP) | compact verdict + `audit_uri` | yes, by `run_key` |

Idempotency state is process-local. The durable version keys off `InvestigationRun`
with a unique index on `(transactionId, idempotencyKey)` — not implemented yet.

## Tests

```bash
cd fraud-engine
source venv/bin/activate
pytest tests/ -v
```
