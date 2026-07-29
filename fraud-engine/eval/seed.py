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

    Only the first of those preserves the contradiction this case exists to
    measure (recomputed 0.65 < stored 0.85). Feed the window total unfiltered
    and the payout is counted twice, the engine returns 0.90 — ABOVE the stored
    0.85 — and a false positive reads as corroborated instead.

    This case is what exposed that the fix had been applied in the wrong place.
    `get_seller_risk_profile` gained an `exclude_payout_id` parameter, but the
    agent never calls that tool: collect_node hands it the profile up front, so
    there is nothing left to fetch. The live path went through collect_node,
    which was still requesting the profile unfiltered, so the agent forwarded
    45000 and concluded TRUE_POSITIVE — while the scripted judge in
    eval/run_mocked.py, which calls the tool with exclude_payout_id itself, kept
    passing. Same fixture, opposite results, and only repeated runs made the
    split visible. collect_node now passes excludePayoutId (agent/nodes.py), so
    the seller's prior-window volume here is 0 and the contradiction stands.

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
        "Ottokar Grienau Uhren Manufaktur",
        "buchhaltung@grienau-uhren.de",
        seller_account_id,
        "acct_1QfGrienauUhren",
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


async def _build_queued_backlog_after_transfer_failures(
    conn: asyncpg.Connection, scenario_key: str, marker: str
) -> int:
    """Queue backlog that reads as draining OR as fallout from a platform outage.

    The grey-zone case for the REVIEW trigger: the window volume is real and the
    rule fires on it, but the failureReason text on the two dead payouts points
    at the platform's own transfer path (api_connection_error, settlement-batch
    lock timeout) rather than at anything the seller did. Both readings survive
    the evidence, which is the point.

    Score arithmetic (fraud-engine/config/rules.yaml, weights all 1.0,
    total = min(sum, 1.0); REVIEW band is 0.3 <= score < 0.7). MEASURED against
    the live engine via POST /check/explain, not derived on paper:

        daily_volume      prior 48000 + amount 4800 = 52800 >= 50000   +0.40
        failed_history    2 failures >= 2                              +0.20
        amount_threshold  amount 4800  < 5000                           0.00
        velocity          4 payouts/24h < 5                             0.00
        new_account       age 400d     >= 30d                           0.00
        dispute_rate      0 disputes    < 1                             0.00
        --------------------------------------------------------------------
        total 0.60 -> REVIEW

    Deliberately insensitive to the totalVolume24h double-count that
    `high_score_clean_history` and `pending_burst_volume_spike` bracket: the
    unfiltered window total (52800) plus the amount lands at 57600, still over
    the same 50000 tier, so the case scores 0.60 REVIEW under either convention
    and stays grey no matter how the agent calls get_seller_risk_profile.

    Two constraints hold the band and must be honoured by any edit:
      - amount stays under 5000, or amount_threshold adds 0.20 (-> 0.80, BLOCK)
        and the case turns into a second `high_score_clean_history`;
      - at most 4 payouts inside 24h, or velocity adds 0.20 (-> 0.80, BLOCK).
    The queued payouts are therefore FEW and LARGE (3 x 16000) while the payout
    under investigation is small — volume without count, which is also what
    separates this shape from `pending_burst_volume_spike`.

    Placement note (adjustable): the two FAILED payouts sit 3d and 4d back —
    inside the 7d failed_history lookback, outside the 24h volume window. Moving
    them inside 24h adds 32000 to the volume (harmless, same tier) but pushes
    velocity to 6 and the score to 0.80 BLOCK.
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
            NOW() - INTERVAL '400 days', NOW() - INTERVAL '3 days'
        ) RETURNING id
        """,
        "Halvorsen Marine Parts",
        "settlements@halvorsen-marine.dk",
        seller_account_id,
        "acct_1QfHalvorsenMarine",
    )
    seller_id = seller["id"]

    # failed_history: two payouts that exhausted all 3 attempts. The reasons are
    # LLM-VISIBLE and carry the whole ambiguity — both name the platform's own
    # transfer path, neither describes seller behaviour.
    for i, (days_ago, reason) in enumerate(
        ((3, "Stripe transfer failed: api_connection_error while creating transfer"),
         (4, "Stripe transfer failed: lock_timeout on platform settlement batch")),
    ):
        failed_tx = await _insert_settled_transaction(
            conn, marker, f"failed-{i + 1}", buyer_account_id, escrow_id,
            amount=16_000, age_interval=f"{days_ago} days",
        )
        await _insert_payout(
            conn,
            transaction_id=failed_tx, seller_id=seller_id,
            escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
            amount=16_000, status="FAILED",
            fraud_decision="ALLOW", fraud_score=0.20,
            age_interval=f"{days_ago} days",
            attempts=3, failure_reason=reason,
        )

    # The backlog: three large payouts sitting PENDING since releases were
    # paused. 48000 of window volume in only three rows — the stored scores ramp
    # as the volume accumulated under them, which the timeline tool surfaces.
    for i, (hours_ago, stored_score, stored_decision) in enumerate(
        ((19, 0.20, "ALLOW"), (11, 0.45, "REVIEW"), (6, 0.55, "REVIEW")),
    ):
        queued_tx = await _insert_settled_transaction(
            conn, marker, f"queued-{i + 1}", buyer_account_id, escrow_id,
            amount=16_000, age_interval=f"{hours_ago} hours",
        )
        await _insert_payout(
            conn,
            transaction_id=queued_tx, seller_id=seller_id,
            escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
            amount=16_000, status="PENDING",
            fraud_decision=stored_decision, fraud_score=stored_score,
            age_interval=f"{hours_ago} hours",
        )

    # Transaction under investigation: a small new order that inherits the
    # backlog's volume. Stored score agrees with the recomputation (0.60) — the
    # question here is what the volume MEANS, not whether the score is right.
    main_tx = await _insert_settled_transaction(
        conn, marker, "main", buyer_account_id, escrow_id, amount=4_800,
    )
    await _insert_payout(
        conn,
        transaction_id=main_tx, seller_id=seller_id,
        escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
        amount=4_800, status="PENDING",
        fraud_decision="REVIEW", fraud_score=0.60,
    )
    return main_tx


async def _build_thin_history_single_signal(
    conn: asyncpg.Connection, scenario_key: str, marker: str
) -> int:
    """One rule fires and there is no history to confirm or refute it.

    The grey-zone case for the MANUAL trigger: an operator escalated this by
    hand precisely because the automation had nothing to work with. The seller
    is five days old with EXACTLY ONE payout — the one under investigation — so
    every behavioural rule reads zero not because the seller is clean but
    because the seller has no record at all. Absence of evidence, arriving in
    the same shape as evidence of absence.

    Score arithmetic (fraud-engine/config/rules.yaml, weights all 1.0,
    total = min(sum, 1.0); REVIEW band is 0.3 <= score < 0.7). MEASURED against
    the live engine via POST /check/explain, not derived on paper:

        amount_threshold  amount 9600 >= 5000                          +0.20
        new_account       age 5d       < 7d                            +0.15
        velocity          1 payout/24h < 5                              0.00
        daily_volume      prior 0 + amount 9600 = 9600  < 20000         0.00
        failed_history    0 failures   < 2                              0.00
        dispute_rate      0 disputes   < 1                              0.00
        --------------------------------------------------------------------
        total 0.35 -> REVIEW

    0.35 sits low in the band ON PURPOSE: barely over the 0.3 ALLOW line is what
    an escalation-by-hand looks like — not a score anyone would auto-block on.
    The margin is thin, so the amount is held UNDER 10000 to keep the case grey
    under either totalVolume24h convention: the unfiltered window total (9600)
    plus the amount is 19200, still short of the 20000 daily_volume tier, so
    the double-count cannot add 0.15 here. At amount >= 10000 it could, and the
    case would flip to 0.80 BLOCK on a caller detail rather than on evidence.

    Nearest precedent, and why this is not its twin: SEED_CASES
    `case_new_seller_high_amount_true_positive` (agent/rag/cases.py) is also
    new_account + amount_threshold on a seller with no history, and retrieval
    WILL surface it — but it is a BLOCK-band case whose TRUE_POSITIVE rests on
    a high-value payout, and 9600 sits in the engine's LOW amount tier (+0.20,
    not +0.50). The overlap is the signal names; the magnitude that carried
    that precedent's verdict is absent here.
    """
    escrow_id, fee_id = await _lookup_platform_accounts(conn)

    buyer_account_id = await _insert_account(conn, marker, "BUYER")
    seller_account_id = await _insert_account(conn, marker, "SELLER")

    # LLM-VISIBLE: plausible newly-onboarded merchant, zero eval/test wording.
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
            NOW() - INTERVAL '5 days', NOW() - INTERVAL '5 days'
        ) RETURNING id
        """,
        "Cassia Botanicals Studio",
        "accounts@cassia-botanicals.co.uk",
        seller_account_id,
        "acct_1QfCassiaBotanicals",
    )
    seller_id = seller["id"]

    # No history at all — this single payout IS the seller's entire record.
    # Nothing else may be inserted here: one prior payout would give
    # daily_volume something to work with and destroy the thin-data premise.
    main_tx = await _insert_settled_transaction(
        conn, marker, "main", buyer_account_id, escrow_id, amount=9_600,
    )
    await _insert_payout(
        conn,
        transaction_id=main_tx, seller_id=seller_id,
        escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
        amount=9_600, status="PENDING",
        fraud_decision="REVIEW", fraud_score=0.35,
    )
    return main_tx


async def _build_clean_seller_volume_burst(
    conn: asyncpg.Connection, scenario_key: str, marker: str
) -> int:
    """BLOCK earned by throughput alone on a seller with nothing against them.

    The second false-alarm shape, built so it CANNOT lean on the mechanism that
    carries `high_score_clean_history`: the amount is held under 5000, so
    amount_threshold contributes nothing and the entire BLOCK comes from
    velocity + daily_volume. Different rules, different reason to doubt.

    Score arithmetic (fraud-engine/config/rules.yaml, weights all 1.0,
    total = min(sum, 1.0), BLOCK at >= 0.7). MEASURED against the live engine
    via POST /check/explain, not derived on paper:

        velocity          12 payouts/24h >= 10                         +0.40
        daily_volume      prior 49500 + amount 4500 = 54000 >= 50000    +0.40
        amount_threshold  amount 4500   < 5000                           0.00
        failed_history    0 failures    < 2                              0.00
        new_account       age 3y       >= 30d                            0.00
        dispute_rate      0 disputes    < 1                              0.00
        --------------------------------------------------------------------
        total 0.80 -> BLOCK

    That combination is FORCED, not chosen. With amount_threshold excluded by
    design and failed_history / dispute_rate / new_account all required to read
    zero (a clean seller is the premise), the only path over 0.7 is the top tier
    of both remaining rules: >= 10 payouts and >= 50000 in the window. Any edit
    that drops either one drops the case out of BLOCK entirely.

    Insensitive to the totalVolume24h double-count: unfiltered, the window total
    is 54000 and the engine adds the amount again for 58500 — same 50000 tier,
    same 0.80. Velocity is never reduced by exclude_payout_id, so it is stable
    by construction.

    What makes the burst explainable rather than suspicious is history the 24h
    window cannot see: three years of trading, zero disputes, zero failures,
    every payout in today's burst ALREADY PAID, an average payout amount that
    today's 4500 sits right on top of, and a comparable 8-payout day 31 days
    back that settled without incident.

    HOW MUCH OF THAT THE AGENT CAN ACTUALLY SEE — measured, not assumed:
      - The risk-profile aggregates are fully visible and carry most of the
        argument: accountAgeDays 1096, totalDisputes 0, failedPayouts 0,
        totalVolumeLifetime, avgPayoutAmount, firstPayoutDate.
      - The 31-day burst and the 60/90/120/150-day steady payouts are NOT
        visible. get_payout_timeline defaults to daysBack=30 and collect_node
        does not override it, so anything older than 30 days is outside the
        lookback entirely.
      - Even inside 30 days the timeline is clipped: this seller's is 4080
        chars against collect_node's [:2000] cut, and the endpoint orders newest
        first, so 7 of 12 payouts survive and the 5 oldest are dropped.
    So the "comparable burst a month ago" reads as designed evidence in this
    fixture but never reaches the model. The case still discriminates on the
    aggregates above; moving the prior burst inside the lookback only helps if
    the truncation limit moves with it.

    Nearest precedent, and why this is not its twin: SEED_CASES
    `case_velocity_spike_true_positive` (agent/rag/cases.py) is the closest
    match by signal overlap (decision:BLOCK + rule:velocity + risk:high) and
    retrieval will surface it with a TRUE_POSITIVE verdict attached — pointing
    the WRONG way. It is defined by two things this case deliberately lacks:
    `rule:failed_history` (zero failures here) and, in its own words, "no
    matching historical volume" — where this seller's lifetime aggregates show
    steady trading at exactly today's amounts. An agent that copies the
    retrieved verdict instead of reading those differences gets this case wrong,
    which is what makes it worth scoring.

    Placement note (adjustable): the historical burst sits at 31 days and the
    steady payouts at 60/90/120/150 days, all far outside the 24h window, so
    they add narrative without touching the arithmetic above. See the visibility
    note above before treating them as evidence the agent weighs — currently
    they are not.
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
            NOW() - INTERVAL '3 years', NOW() - INTERVAL '1 day'
        ) RETURNING id
        """,
        "Bergqvist Antikvariat",
        "ekonomi@bergqvist-antikvariat.se",
        seller_account_id,
        "acct_1QfBergqvistAntik",
    )
    seller_id = seller["id"]

    # Steady baseline, months back: this seller has always traded.
    for i, days_ago in enumerate((150, 120, 90, 60)):
        steady_tx = await _insert_settled_transaction(
            conn, marker, f"steady-{i + 1}", buyer_account_id, escrow_id,
            amount=4_200, age_interval=f"{days_ago} days",
        )
        await _insert_payout(
            conn,
            transaction_id=steady_tx, seller_id=seller_id,
            escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
            amount=4_200, status="PAID",
            fraud_decision="ALLOW", fraud_score=0.10,
            age_interval=f"{days_ago} days",
        )

    # The precedent for today: an eight-payout day a month ago that settled
    # cleanly. Outside the 24h window, so it is narrative only.
    for i in range(8):
        prior_burst_tx = await _insert_settled_transaction(
            conn, marker, f"prior-burst-{i + 1}", buyer_account_id, escrow_id,
            amount=4_400, age_interval=f"31 days {i * 2} hours",
        )
        await _insert_payout(
            conn,
            transaction_id=prior_burst_tx, seller_id=seller_id,
            escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
            amount=4_400, status="PAID",
            fraud_decision="ALLOW", fraud_score=0.15,
            age_interval=f"31 days {i * 2} hours",
        )

    # Today's burst: eleven payouts, ALL settled, 49500 of prior window volume.
    # Stored scores ramp as the day's volume accumulated under them.
    for i, hours_ago in enumerate((22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2)):
        burst_tx = await _insert_settled_transaction(
            conn, marker, f"burst-{i + 1}", buyer_account_id, escrow_id,
            amount=4_500, age_interval=f"{hours_ago} hours",
        )
        await _insert_payout(
            conn,
            transaction_id=burst_tx, seller_id=seller_id,
            escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
            amount=4_500, status="PAID",
            fraud_decision="ALLOW" if i < 6 else "REVIEW",
            fraud_score=0.15 if i < 6 else 0.55,
            age_interval=f"{hours_ago} hours",
        )

    # Transaction under investigation — the twelfth payout of the day, the one
    # that tips velocity into its top tier.
    main_tx = await _insert_settled_transaction(
        conn, marker, "main", buyer_account_id, escrow_id, amount=4_500,
    )
    await _insert_payout(
        conn,
        transaction_id=main_tx, seller_id=seller_id,
        escrow_account_id=escrow_id, platform_fee_account_id=fee_id,
        amount=4_500, status="PENDING",
        fraud_decision="BLOCK", fraud_score=0.80,
    )
    return main_tx


# scenario_key -> builder. Keys MUST match the "scenario" field in golden.jsonl.
SCENARIO_BUILDERS: dict[str, Builder] = {
    "fraud_confirmed_by_history": _build_fraud_confirmed_by_history,
    "high_score_clean_history": _build_high_score_clean_history,
    "pending_burst_volume_spike": _build_pending_burst_volume_spike,
    "queued_backlog_after_transfer_failures": _build_queued_backlog_after_transfer_failures,
    "thin_history_single_signal": _build_thin_history_single_signal,
    "clean_seller_volume_burst": _build_clean_seller_volume_burst,
}
