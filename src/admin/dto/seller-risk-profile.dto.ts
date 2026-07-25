import { ApiProperty } from '@nestjs/swagger';

export class SellerRiskMetricsDto {
  @ApiProperty({ description: 'Total payouts ever created for this seller' })
  totalPayouts!: number;

  @ApiProperty({ description: 'Payouts that reached PAID status' })
  paidPayouts!: number;

  @ApiProperty({ description: 'Payouts that reached FAILED status' })
  failedPayouts!: number;

  @ApiProperty({ description: 'Payouts that were REVERSED' })
  reversedPayouts!: number;

  @ApiProperty({ description: 'Total disputes filed against this seller' })
  totalDisputes!: number;

  @ApiProperty({ description: 'Disputes resolved as LOST (buyer won)' })
  lostDisputes!: number;

  @ApiProperty({
    description:
      'Payouts created in the last 24 hours. Always the full window count — ' +
      'never reduced by `excludePayoutId`, because the fraud engine\'s velocity ' +
      'rule compares this number directly without adding the payout under ' +
      'investigation back in.',
  })
  payoutVelocity24h!: number;

  @ApiProperty({
    description:
      'Total payout volume in cents in the last 24 hours, across ALL statuses — ' +
      'unsettled payouts included, since a stack of them is exactly the spike ' +
      'the daily_volume rule looks for. Reduced by exactly one payout when ' +
      '`excludePayoutId` is passed (see `volume24hExcludesPayoutId`).',
  })
  totalVolume24h!: number;

  @ApiProperty({
    description:
      'The payout id excluded from `totalVolume24h`, or null when the total ' +
      'covers the whole window. Makes the figure unambiguous to its consumers.',
    nullable: true,
  })
  volume24hExcludesPayoutId!: number | null;

  @ApiProperty({ description: 'Total lifetime payout volume in cents' })
  totalVolumeLifetime!: number;

  @ApiProperty({ description: 'Average payout amount in cents (0 if no payouts)' })
  avgPayoutAmount!: number;

  @ApiProperty({ description: 'Seller account age in days' })
  accountAgeDays!: number;

  @ApiProperty({ description: 'Date of first payout, null if none', nullable: true })
  firstPayoutDate!: string | null;

  @ApiProperty({ description: 'ISO duration since last FAILED payout, null if none', nullable: true })
  timeSinceLastFailure!: string | null;
}

export class SellerRiskProfileDto {
  @ApiProperty()
  seller!: {
    id: number;
    name: string;
    email: string;
    status: string;
    stripeAccountId: string | null;
    chargesEnabled: boolean;
    payoutsEnabled: boolean;
    payoutsBlocked: boolean;
    negativeBalance: number;
    accountAgeDays: number;
    createdAt: string;
  };

  @ApiProperty()
  ledger!: {
    accountId: number;
    balance: number;
  };

  @ApiProperty({ type: SellerRiskMetricsDto })
  riskMetrics!: SellerRiskMetricsDto;
}
