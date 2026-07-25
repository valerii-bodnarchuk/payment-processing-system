"""
eval/seed.py — idempotent seeding of golden-dataset fixtures.

Mirrors the raw-asyncpg INSERT style of tests/conftest.py::seed_blocked_payout,
but decoupled from transaction_id: golden.jsonl references scenarios by a logical
key, and this module maps {scenario_key -> transaction_id} freshly on every run.

────────────────────────────────────────────────────────────────────────────
Contamination invariant (non-negotiable)
────────────────────────────────────────────────────────────────────────────
The investigation agent exposes some DB columns to the LLM through its tools.
The eval marker MUST NEVER land in an exposed column, or the agent would read a
hint straight out of the data and any measured accuracy would be fake.

Marker lives ONLY in columns that are NOT serialised to the LLM:
    - Transaction.description = "EVAL::<hash>"
    - Account.name            = "EVAL::<hash>"   (both BUYER and SELLER accounts)

<hash> is an opaque digest of the scenario key with ZERO semantics. The
scenario_key <-> hash mapping is kept here in Python — never written to the DB.

LLM-VISIBLE columns (verified via the agent tools + NestJS endpoints) that each
builder MUST fill with domain-plausible, marker-free values:
    - Seller.name              (get_seller_risk_profile)
    - Seller.email             (get_seller_risk_profile)
    - Seller.stripeAccountId   (get_seller_risk_profile + collectContext)
    - Payout.failureReason     (get_payout_timeline + findings)
guard.assert_no_contamination() enforces this at runtime; keeping the marker out
of these columns keeps it enforceable.

Platform ESCROW / PLATFORM_FEE accounts (from `npm run prisma:seed`) are looked
up and reused read-only, exactly as the e2e fixture does — never created or
deleted here.
"""
from __future__ import annotations

import hashlib
from typing import Awaitable, Callable

import asyncpg

# Prefix shared with eval/guard.py — keep the two in sync.
EVAL_MARKER_PREFIX = "EVAL::"

# A builder inserts one scenario and returns its transaction_id.
# Signature: (conn, scenario_key, marker) -> transaction_id
Builder = Callable[[asyncpg.Connection, str, str], Awaitable[int]]


# ── Marker helpers ──────────────────────────────────────────────────────────

def scenario_hash(scenario_key: str) -> str:
    """Opaque, stable digest of a scenario key. No semantics, not reversible
    to the key by the LLM — purely a DB-side handle for teardown."""
    return hashlib.sha1(scenario_key.encode("utf-8")).hexdigest()[:12]


def marker_for(scenario_key: str) -> str:
    """The EVAL::<hash> string to write into non-exposed marker columns."""
    return f"{EVAL_MARKER_PREFIX}{scenario_hash(scenario_key)}"


# ── Idempotent teardown ─────────────────────────────────────────────────────

async def _truncate_eval_fixtures(conn: asyncpg.Connection) -> None:
    """Delete only eval rows, making a re-seed idempotent without touching e2e
    data or platform accounts.

    Anchored on the non-exposed markers (Transaction.description and
    Account.name LIKE 'EVAL::%'), then FK-traversed. Deletes in FK order:
    Dispute -> Payout -> Entry -> Transaction -> Seller -> Account.

    Sellers have no contamination-safe marker column of their own (name/email/
    stripeAccountId are all LLM-visible), so eval sellers are found by FK from
    their marked SELLER account.
    """
    like = EVAL_MARKER_PREFIX + "%"

    async with conn.transaction():
        tx_ids = [
            r["id"]
            for r in await conn.fetch(
                'SELECT id FROM "Transaction" WHERE description LIKE $1', like
            )
        ]
        acct_ids = [
            r["id"]
            for r in await conn.fetch(
                'SELECT id FROM "Account" WHERE name LIKE $1', like
            )
        ]
        seller_ids = (
            [
                r["id"]
                for r in await conn.fetch(
                    'SELECT id FROM "Seller" WHERE "accountId" = ANY($1::int[])',
                    acct_ids,
                )
            ]
            if acct_ids
            else []
        )

        if tx_ids:
            await conn.execute(
                'DELETE FROM "Dispute" WHERE "transactionId" = ANY($1::int[])', tx_ids
            )
            await conn.execute(
                'DELETE FROM "Payout" WHERE "transactionId" = ANY($1::int[])', tx_ids
            )
            await conn.execute(
                'DELETE FROM "Entry" WHERE "transactionId" = ANY($1::int[])', tx_ids
            )
            await conn.execute(
                'DELETE FROM "Transaction" WHERE id = ANY($1::int[])', tx_ids
            )
        if seller_ids:
            await conn.execute(
                'DELETE FROM "Seller" WHERE id = ANY($1::int[])', seller_ids
            )
        if acct_ids:
            await conn.execute(
                'DELETE FROM "Account" WHERE id = ANY($1::int[])', acct_ids
            )


# ── Public API ──────────────────────────────────────────────────────────────

async def seed_eval_fixtures(conn: asyncpg.Connection) -> dict[str, int]:
    """Truncate prior eval rows, then insert every registered scenario.

    Each builder runs in its own transaction so a failure in one scenario rolls
    back cleanly and never leaves orphan Seller/Account rows behind.

    Returns {scenario_key: transaction_id} for the runner's resolver to map
    golden cases onto freshly seeded rows.
    """
    await _truncate_eval_fixtures(conn)

    result: dict[str, int] = {}
    for scenario_key, builder in SCENARIO_BUILDERS.items():
        async with conn.transaction():
            tx_id = await builder(conn, scenario_key, marker_for(scenario_key))
        result[scenario_key] = tx_id
    return result


# ── Insert helpers ──────────────────────────────────────────────────────────
#
# Mirror tests/conftest.py::seed_blocked_payout: platform ESCROW / PLATFORM_FEE
# accounts are looked up read-only, everything else is inserted per scenario.


async def _lookup_platform_accounts(conn: asyncpg.Connection) -> tuple[int, int]:
    """(escrow_account_id, platform_fee_account_id) from `npm run prisma:seed`."""
    escrow = await conn.fetchrow(
        'SELECT id FROM "Account" WHERE type = $1 ORDER BY id LIMIT 1', "ESCROW"
    )
    fee = await conn.fetchrow(
        'SELECT id FROM "Account" WHERE type = $1 ORDER BY id LIMIT 1', "PLATFORM_FEE"
    )
    if not escrow or not fee:
        raise RuntimeError(
            "Platform ESCROW / PLATFORM_FEE accounts not found. "
            "Run `npm run prisma:seed` before seeding eval fixtures."
        )
    return escrow["id"], fee["id"]


async def _insert_account(
    conn: asyncpg.Connection, marker: str, account_type: str
) -> int:
    """Account.name carries the marker — never serialised to the LLM."""
    row = await conn.fetchrow(
        """
        INSERT INTO "Account" (name, type, "allowNegative", "createdAt")
        VALUES ($1, $2, TRUE, NOW())
        RETURNING id
        """,
        marker,
        account_type,
    )
    return row["id"]


async def _insert_settled_transaction(
    conn: asyncpg.Connection,
    marker: str,
    label: str,
    buyer_account_id: int,
    escrow_account_id: int,
    amount: int,
    age_interval: str = "0 days",
) -> int:
    """COMPLETED transaction with balanced buyer DEBIT / escrow CREDIT entries.

    Entries MUST balance: LedgerService.verifyIntegrity() runs a per-transaction
    GROUP BY/HAVING check, and any imbalance would fire the unrelated
    `ledger_imbalanced` critical finding and pollute the scenario.

    Balance convention (ledger.service.ts:284): CREDIT adds, DEBIT subtracts —
    so the escrow CREDIT is what funds the payout.
    """
    tx = await conn.fetchrow(
        f"""
        INSERT INTO "Transaction" (description, status, "createdAt")
        VALUES ($1, 'COMPLETED', NOW() - INTERVAL '{age_interval}')
        RETURNING id
        """,
        f"{marker} {label}",
    )
    tx_id = tx["id"]

    for account_id, entry_type in ((buyer_account_id, "DEBIT"), (escrow_account_id, "CREDIT")):
        await conn.execute(
            f"""
            INSERT INTO "Entry" ("accountId", "transactionId", amount, type, "createdAt")
            VALUES ($1, $2, $3, '{entry_type}', NOW() - INTERVAL '{age_interval}')
            """,
            account_id,
            tx_id,
            amount,
        )
    return tx_id


async def _insert_payout(
    conn: asyncpg.Connection,
    *,
    transaction_id: int,
    seller_id: int,
    escrow_account_id: int,
    platform_fee_account_id: int,
    amount: int,
    status: str,
    fraud_decision: str,
    fraud_score: float,
    age_interval: str = "0 days",
    attempts: int = 0,
    failure_reason: str | None = None,
) -> int:
    """Payout row. `fraud_score` / `fraud_decision` are stored LITERALS — the
    rules engine does not recompute them at seed time, which is exactly the
    lever that lets a scenario agree or disagree with its own history."""
    platform_fee = round(amount * 0.05)
    row = await conn.fetchrow(
        f"""
        INSERT INTO "Payout" (
            status, amount, "platformFee", "sellerAmount",
            "transactionId", "sellerId",
            "escrowAccountId", "platformFeeAccountId",
            attempts, "maxAttempts",
            "fraudDecision", "fraudScore", "failureReason",
            "createdAt", "updatedAt"
        ) VALUES (
            $1, $2, $3, $4,
            $5, $6,
            $7, $8,
            $9, 3,
            $10, $11, $12,
            NOW() - INTERVAL '{age_interval}', NOW() - INTERVAL '{age_interval}'
        ) RETURNING id
        """,
        status, amount, platform_fee, amount - platform_fee,
        transaction_id, seller_id,
        escrow_account_id, platform_fee_account_id,
        attempts,
        fraud_decision, fraud_score, failure_reason,
    )
    return row["id"]


# ── Scenario builders ───────────────────────────────────────────────────────
#
# One builder per row in eval/golden.jsonl, registered in SCENARIO_BUILDERS
# under the SAME string used in that row's "scenario" field.
#
# Every builder must keep the marker out of the four LLM-VISIBLE columns:
#   Seller.name, Seller.email, Seller.stripeAccountId, Payout.failureReason
# eval/guard.py enforces this at runtime.


async def _build_fraud_confirmed_by_history(
    conn: asyncpg.Connection, scenario_key: str, marker: str
) -> int:
    """Fraud signal corroborated by seller behaviour.

    Score arithmetic (fraud-engine/config/rules.yaml, weights all 1.0,
    total = min(sum, 1.0), BLOCK above 0.7):

        amount_threshold  amount 45000 >= 10000        +0.50
        velocity          6 payouts/24h >= 5           +0.20
        new_account       age 3d < 7d                  +0.15
        dispute_rate      1 dispute >= 1               +0.15
        daily_volume      60000+45000 >= 50000         +0.40
        ------------------------------------------------------
        raw sum 1.40 -> capped 1.00 -> BLOCK

    The user-specified combination reaches 1.0 without daily_volume (0.5+0.2+
    0.15+0.15); daily_volume also fires here but the cap makes it moot, so the
    stored score stays 1.0 either way.

    Every behavioural rule fires, so the stored score is fully corroborated by
    the seller's own history — nothing for the agent to argue with.

    Placement note (adjustable): the dispute sits on a HISTORY transaction, not
    on the transaction under investigation. dispute_rate counts it either way
    (it matches on transaction.payouts.some(sellerId)), but keeping it off the
    main transaction avoids firing the unrelated `active_dispute` critical
    finding, which would recommend "wait for dispute resolution" instead of a
    fraud judgement. Move it to the main transaction if that is the intent.
    """
    escrow_id, fee_id = await _lookup_platform_accounts(conn)

    buyer_account_id = await _insert_account(conn, marker, "BUYER")
    seller_account_id = await _insert_account(conn, marker, "SELLER")

    # LLM-VISIBLE: plausible merchant identity, zero eval/test wording.
    seller = await conn.fetchrow(
        """
        INSERT INTO "Seller" (
            name, email, status, "accountId",
            "stripeAccountId", "chargesEnabled", "payoutsEnabled",
            "payoutsBlocked", "negativeBalance",
            "createdAt", "updatedAt"
        ) VALUES (
            $1, $2, 'ACTIVE', $3,
            $4, TRUE, TRUE,
            FALSE, 0,
            NOW() - INTERVAL '3 days', NOW() - INTERVAL '3 days'
        ) RETURNING id
        """,
        "Lumen Trade Supply",
        "payouts@lumen-trade-supply.com",
        seller_account_id,
        "acct_1QfLumenTradeSupply",
    )
    seller_id = seller["id"]

    # Velocity: five settled payouts inside the 24h window, plus the payout
    # under investigation = 6 >= the rule's min_count of 5.
    history_tx_ids: list[int] = []
    for i in range(5):
        hist_tx = await _insert_settled_transaction(
            conn, marker, f"history-{i + 1}", buyer_account_id, escrow_id,
            amount=12_000, age_interval=f"{4 + i * 3} hours",
        )
        await _insert_payout(
            conn,
            transaction_id=hist_tx, seller_id=seller_id,
            escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
            amount=12_000, status="PAID",
            fraud_decision="ALLOW", fraud_score=0.15,
            age_interval=f"{4 + i * 3} hours",
        )
        history_tx_ids.append(hist_tx)

    # dispute_rate: one dispute against the seller's earlier trading.
    await conn.execute(
        """
        INSERT INTO "Dispute" (status, reason, amount, "transactionId", "createdAt", "updatedAt")
        VALUES ('OPEN', 'FRAUDULENT', $1, $2, NOW() - INTERVAL '6 hours', NOW() - INTERVAL '6 hours')
        """,
        12_000,
        history_tx_ids[0],
    )

    # Transaction under investigation.
    main_tx = await _insert_settled_transaction(
        conn, marker, "main", buyer_account_id, escrow_id, amount=45_000,
    )
    await _insert_payout(
        conn,
        transaction_id=main_tx, seller_id=seller_id,
        escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
        amount=45_000, status="PENDING",
        fraud_decision="BLOCK", fraud_score=1.0,
    )
    return main_tx


async def _build_high_score_clean_history(
    conn: asyncpg.Connection, scenario_key: str, marker: str
) -> int:
    """High stored fraud score with nothing in the history to support it.

    Same payout amount (45000) and the same BLOCK band as
    `fraud_confirmed_by_history`, so the ONLY variable between the two cases is
    the seller's history — the agent has to reach a different conclusion from
    behaviour alone rather than from the score.

    What rules.yaml computes for THIS history:

        amount_threshold  amount 45000 >= 10000        +0.50
        daily_volume      see below                    +0.15 or +0.40
        velocity          1 payout/24h  < 5             0.00
        new_account       age 4y       >= 30d           0.00
        failed_history    0 failures    < 2             0.00
        dispute_rate      0 disputes    < 1             0.00

    MEASURED, not assumed — daily_volume decides the outcome, and it depends on
    what the caller passes (engine.py: total = seller_total_amount_24h + amount,
    i.e. the engine expects PRIOR volume and adds the current amount itself):

        seller_total_amount_24h=0      -> total  45000 -> +0.15 -> 0.65 REVIEW
        seller_total_amount_24h=45000  -> total  90000 -> +0.40 -> 0.90 BLOCK

    `get_seller_risk_profile` reports totalVolume24h over ALL payouts in the
    window INCLUDING the one under investigation, so an agent that forwards it
    verbatim double-counts this payout and the engine returns 0.90 BLOCK —
    ABOVE the stored 0.85. The intended contradiction (recomputed < stored)
    then disappears and the score looks corroborated instead.

    So this scenario only discriminates if the agent passes prior-window volume
    (excluding the current payout). Confirmed against the live pipeline; both
    numbers above came from POST /check/explain, not from arithmetic on paper.
    Resolving this is a domain call: either treat "does the agent avoid the
    double-count" as part of what is being measured, or move the case onto a
    signal that does not route through daily_volume.

    Matches the existing labelled precedent in the corpus: SYN-017
    (agent/rag/synthetic_cases.py, cluster `legitimate_high_risk`) is a
    high-value purchase that trips amount_threshold on a long-tenured,
    dispute-free customer and resolves to FALSE_POSITIVE.
    """
    escrow_id, fee_id = await _lookup_platform_accounts(conn)

    buyer_account_id = await _insert_account(conn, marker, "BUYER")
    seller_account_id = await _insert_account(conn, marker, "SELLER")

    # LLM-VISIBLE: plausible long-established merchant, zero eval/test wording.
    seller = await conn.fetchrow(
        """
        INSERT INTO "Seller" (
            name, email, status, "accountId",
            "stripeAccountId", "chargesEnabled", "payoutsEnabled",
            "payoutsBlocked", "negativeBalance",
            "createdAt", "updatedAt"
        ) VALUES (
            $1, $2, 'ACTIVE', $3,
            $4, TRUE, TRUE,
            FALSE, 0,
            NOW() - INTERVAL '4 years', NOW() - INTERVAL '30 days'
        ) RETURNING id
        """,
        "Hartmann Uhren Manufaktur",
        "buchhaltung@hartmann-uhren.de",
        seller_account_id,
        "acct_1QfHartmannUhren",
    )
    seller_id = seller["id"]

    # Exactly one payout, no history, no disputes, no failures.
    main_tx = await _insert_settled_transaction(
        conn, marker, "main", buyer_account_id, escrow_id, amount=45_000,
    )
    await _insert_payout(
        conn,
        transaction_id=main_tx, seller_id=seller_id,
        escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
        amount=45_000, status="PENDING",
        fraud_decision="BLOCK", fraud_score=0.85,
    )
    return main_tx


async def _build_pending_burst_volume_spike(
    conn: asyncpg.Connection, scenario_key: str, marker: str
) -> int:
    """A burst of UNSETTLED payouts inside the window — daily_volume MUST fire.

    Regression guard for the risk-profile aggregate itself, not for the LLM.
    The window volume here lives entirely in payouts that have NOT settled yet
    (all PENDING), which is exactly the shape a payout-draining seller has: the
    money is queued, not paid out. Any implementation of `totalVolume24h` that
    drops unsettled payouts wholesale reports ~0 for this seller and silently
    turns the spike invisible.

    Score arithmetic (fraud-engine/config/rules.yaml, weights all 1.0,
    total = min(sum, 1.0), BLOCK above 0.7). The engine computes daily_volume
    as `seller_total_amount_24h + amount`, i.e. it expects PRIOR window volume
    and adds the payout under investigation itself:

        amount_threshold  amount 9000 >= 5000          +0.20
        velocity          6 payouts/24h >= 5           +0.20
        failed_history    2 failures/7d >= 2           +0.20
        new_account       age 210d     >= 30d           0.00
        dispute_rate      0 disputes    < 1             0.00
        daily_volume      depends on the aggregate — see below

    daily_volume is the ONLY rule that moves, and it decides the band:

        prior 45000 (the five other PENDINGs)  -> total 54000 -> +0.40 -> 1.00 BLOCK
        prior 0     (unsettled dropped)        -> total  9000 ->  0.00 -> 0.60 REVIEW

    So an aggregate that excludes unsettled payouts costs this case its BLOCK
    and, downstream, its verdict. Contrast with `high_score_clean_history`,
    which pins the opposite error (counting the investigated payout twice);
    together the two bracket `totalVolume24h` from both sides.

    Placement note (adjustable): the two FAILED payouts sit 3d and 4d back —
    inside failed_history's 7d lookback but OUTSIDE the 24h window, so they
    contribute a behavioural signal without touching the volume arithmetic
    above. Move them inside 24h only if the volume numbers are recomputed.
    """
    escrow_id, fee_id = await _lookup_platform_accounts(conn)

    buyer_account_id = await _insert_account(conn, marker, "BUYER")
    seller_account_id = await _insert_account(conn, marker, "SELLER")

    # LLM-VISIBLE: plausible merchant identity, zero eval/test wording.
    seller = await conn.fetchrow(
        """
        INSERT INTO "Seller" (
            name, email, status, "accountId",
            "stripeAccountId", "chargesEnabled", "payoutsEnabled",
            "payoutsBlocked", "negativeBalance",
            "createdAt", "updatedAt"
        ) VALUES (
            $1, $2, 'ACTIVE', $3,
            $4, TRUE, TRUE,
            FALSE, 0,
            NOW() - INTERVAL '210 days', NOW() - INTERVAL '2 days'
        ) RETURNING id
        """,
        "Nordvik Outdoor Equipment",
        "finance@nordvik-outdoor.no",
        seller_account_id,
        "acct_1QfNordvikOutdoor",
    )
    seller_id = seller["id"]

    # failed_history: two FAILED payouts inside the 7d lookback but outside the
    # 24h window, so they never enter the volume arithmetic.
    for i, (days_ago, reason) in enumerate(
        ((3, "Stripe transfer failed: insufficient_funds in platform balance"),
         (4, "Stripe transfer failed: account_frozen on destination account")),
    ):
        failed_tx = await _insert_settled_transaction(
            conn, marker, f"failed-{i + 1}", buyer_account_id, escrow_id,
            amount=8_000, age_interval=f"{days_ago} days",
        )
        await _insert_payout(
            conn,
            transaction_id=failed_tx, seller_id=seller_id,
            escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
            amount=8_000, status="FAILED",
            fraud_decision="ALLOW", fraud_score=0.20,
            age_interval=f"{days_ago} days",
            attempts=3, failure_reason=reason,
        )

    # The burst: five PENDING payouts stacked up inside the window, 45000 total.
    # None of them settled — this is the volume the daily_volume rule exists for.
    for i, hours_ago in enumerate((2, 5, 8, 12, 18)):
        burst_tx = await _insert_settled_transaction(
            conn, marker, f"burst-{i + 1}", buyer_account_id, escrow_id,
            amount=9_000, age_interval=f"{hours_ago} hours",
        )
        await _insert_payout(
            conn,
            transaction_id=burst_tx, seller_id=seller_id,
            escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
            amount=9_000, status="PENDING",
            fraud_decision="ALLOW", fraud_score=0.25,
            age_interval=f"{hours_ago} hours",
        )

    # Transaction under investigation — the sixth payout of the burst.
    main_tx = await _insert_settled_transaction(
        conn, marker, "main", buyer_account_id, escrow_id, amount=9_000,
    )
    await _insert_payout(
        conn,
        transaction_id=main_tx, seller_id=seller_id,
        escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
        amount=9_000, status="PENDING",
        fraud_decision="BLOCK", fraud_score=0.95,
    )
    return main_tx


# scenario_key -> builder. Keys MUST match the "scenario" field in golden.jsonl.
SCENARIO_BUILDERS: dict[str, Builder] = {
    "fraud_confirmed_by_history": _build_fraud_confirmed_by_history,
    "high_score_clean_history": _build_high_score_clean_history,
    "pending_burst_volume_spike": _build_pending_burst_volume_spike,
}
