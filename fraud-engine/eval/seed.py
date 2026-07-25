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
import uuid
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


# ── Scenario builders ───────────────────────────────────────────────────────
#
# One builder per row in eval/golden.jsonl. Register it in SCENARIO_BUILDERS
# under the SAME string used in that row's "scenario" field.
#
# Insertion pattern (mirror tests/conftest.py::seed_blocked_payout):
#   1. Look up (do NOT create) the platform ESCROW / PLATFORM_FEE accounts.
#   2. Insert BUYER + SELLER Account   -> Account.name = marker
#   3. Insert Seller                   -> name/email/stripeAccountId: DOMAIN data
#   4. Insert Transaction              -> Transaction.description = marker
#   5. Insert balanced Entry rows (DEBIT/CREDIT)
#   6. Insert Payout (+ optional Dispute) -> failureReason: DOMAIN data
#   7. return transaction_id
#
# Bodies are intentionally empty — the data pattern is a domain decision.


async def _build_placeholder_scenario(
    conn: asyncpg.Connection, scenario_key: str, marker: str
) -> int:
    """Clear-cut fraud case: high fraud score, BLOCK decision the agent should
    confirm as TRUE_POSITIVE.

    Marker placement (contamination-safe — NOT seen by the LLM):
        Transaction.description = marker
        Account.name            = marker   (BUYER and SELLER accounts)

    LLM-VISIBLE — domain-plausible, zero 'eval'/'test'/marker:
        Seller.name, Seller.email, Seller.stripeAccountId, Payout.failureReason
    """
    suffix = uuid.uuid4().hex[:8]

    # ── Lookup pre-existing platform accounts ──────────────────────────
    escrow_row = await conn.fetchrow(
        'SELECT id FROM "Account" WHERE type = $1 ORDER BY id LIMIT 1',
        "ESCROW",
    )
    if not escrow_row:
        raise RuntimeError(
            'Platform ESCROW account not found. Run `npm run prisma:seed`.'
        )
    escrow_account_id: int = escrow_row["id"]

    fee_row = await conn.fetchrow(
        'SELECT id FROM "Account" WHERE type = $1 ORDER BY id LIMIT 1',
        "PLATFORM_FEE",
    )
    if not fee_row:
        raise RuntimeError(
            'Platform FEE account not found. Run `npm run prisma:seed`.'
        )
    platform_fee_account_id: int = fee_row["id"]

    # Buyer account (DEBIT source)
    buyer_row = await conn.fetchrow(
        """
        INSERT INTO "Account" (name, type, "allowNegative", "createdAt")
        VALUES ($1, $2, TRUE, NOW())
        RETURNING id
        """,
        marker,
        "BUYER",
    )
    buyer_account_id: int = buyer_row["id"]

    # Seller ledger account
    seller_account_row = await conn.fetchrow(
        """
        INSERT INTO "Account" (name, type, "allowNegative", "createdAt")
        VALUES ($1, $2, TRUE, NOW())
        RETURNING id
        """,
        marker,
        "SELLER",
    )
    seller_account_id: int = seller_account_row["id"]

    # Seller entity (ACTIVE so risk-profile and timeline endpoints work)
    seller_row = await conn.fetchrow(
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
            NOW(), NOW()
        ) RETURNING id
        """,
        f"Marcus Ferreira {suffix}",
        f"marcus.ferreira.{suffix}@sellerhub.io",
        seller_account_id,
        f"acct_{suffix}",
    )
    seller_id: int = seller_row["id"]

    # COMPLETED transaction
    tx_row = await conn.fetchrow(
        """
        INSERT INTO "Transaction" (description, status, "createdAt")
        VALUES ($1, 'COMPLETED', NOW())
        RETURNING id
        """,
        marker,
    )
    transaction_id: int = tx_row["id"]

    # Balanced ledger entries: buyer DEBIT <-> escrow CREDIT
    amount = 100_000  # EUR 1 000 in cents
    await conn.execute(
        """
        INSERT INTO "Entry" ("accountId", "transactionId", amount, type, "createdAt")
        VALUES ($1, $2, $3, 'DEBIT', NOW())
        """,
        buyer_account_id,
        transaction_id,
        amount,
    )
    await conn.execute(
        """
        INSERT INTO "Entry" ("accountId", "transactionId", amount, type, "createdAt")
        VALUES ($1, $2, $3, 'CREDIT', NOW())
        """,
        escrow_account_id,
        transaction_id,
        amount,
    )

    # PENDING payout with a high fraud score — clear-cut BLOCK
    payout_amount = 50_000   # EUR 500
    platform_fee = 2_500     # 5 %
    seller_amount = payout_amount - platform_fee

    await conn.execute(
        """
        INSERT INTO "Payout" (
            status, amount, "platformFee", "sellerAmount",
            "transactionId", "sellerId",
            "escrowAccountId", "platformFeeAccountId",
            attempts, "maxAttempts",
            "fraudDecision", "fraudScore",
            "createdAt", "updatedAt"
        ) VALUES (
            'PENDING', $1, $2, $3,
            $4, $5,
            $6, $7,
            0, 3,
            'BLOCK', 0.85,
            NOW(), NOW()
        )
        """,
        payout_amount, platform_fee, seller_amount,
        transaction_id, seller_id,
        escrow_account_id, platform_fee_account_id,
    )

    return transaction_id


# scenario_key -> builder. Keys MUST match the "scenario" field in golden.jsonl.
SCENARIO_BUILDERS: dict[str, Builder] = {
    "placeholder_scenario": _build_placeholder_scenario,
    # "your_scenario_key": _build_your_scenario,
}
