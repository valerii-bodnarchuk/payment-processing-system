"""
Tool: get_seller_risk_profile

Calls the new /admin/sellers/:id/risk-profile endpoint.
Returns seller record + ledger balance + computed risk metrics in one shot.
"""
from langchain_core.tools import tool

from agent.tools.nestjs_client import nestjs_get


@tool
async def get_seller_risk_profile(
    seller_id: int, exclude_payout_id: int | None = None
) -> dict:
    """Fetch aggregated seller risk profile: account info, ledger balance,
    payout velocity, dispute rate, volume trends, account age.
    Use this to understand the seller's overall risk posture.

    ALWAYS pass exclude_payout_id when investigating a specific payout and
    intending to feed totalVolume24h to get_fraud_score_explanation. The fraud
    engine's daily_volume rule computes `seller_total_amount_24h + amount`, so
    it adds the payout under investigation back in itself — forward the
    unfiltered total and that payout gets counted twice, inflating the score.

    exclude_payout_id removes that ONE payout and nothing else. Every other
    payout in the 24h window stays in totalVolume24h, including ones that have
    not settled yet: a seller with several payouts queued up is precisely the
    volume spike the rule exists to catch. payoutVelocity24h is never reduced
    by this parameter — forward it to the engine as-is."""

    path = f"/admin/sellers/{seller_id}/risk-profile"
    if exclude_payout_id is not None:
        path += f"?excludePayoutId={exclude_payout_id}"

    result = await nestjs_get(path)

    if isinstance(result, dict) and result.get("error"):
        return {
            "error": True,
            "tool": "get_seller_risk_profile",
            "status_code": result.get("status_code"),
            "detail": result.get("detail", "Unknown error"),
        }

    return result
