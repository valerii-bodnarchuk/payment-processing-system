"""
eval — golden-dataset infrastructure for the fraud-investigation agent.

Pieces (scoring/runner intentionally excluded):
  - golden.jsonl : labelled cases, decoupled from DB ids via a logical scenario key
  - data.py      : load_golden() + validation
  - seed.py      : idempotent asyncpg seeding, returns {scenario_key: transaction_id}
  - guard.py     : assert_no_contamination() — the eval marker must never reach the LLM
"""
