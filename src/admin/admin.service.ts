import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { LedgerService } from '../ledger/ledger.service';

@Injectable()
export class AdminService {
  constructor(
    private prisma: PrismaService,
    private ledger: LedgerService,
  ) {}

  async seedScenario(scenario: 'healthy' | 'problematic') {
    if (process.env.NODE_ENV === 'production') {
      throw new ForbiddenException('Dev seed is disabled in production');
    }

    // Always use the seeded test seller (id=1). Re-seed if missing.
    let seller = await this.prisma.seller.findFirst();
    if (!seller) {
      throw new NotFoundException('No seller found — run npm run prisma:seed first');
    }

    if (scenario === 'healthy') {
      const tx = await this.prisma.transaction.create({
        data: { status: 'COMPLETED', description: 'Dev seed: healthy payout' },
      });
      const payout = await this.prisma.payout.create({
        data: {
          status: 'ELIGIBLE',
          amount: 4500,
          platformFee: 225,
          sellerAmount: 4275,
          attempts: 0,
          maxAttempts: 3,
          transactionId: tx.id,
          sellerId: seller.id,
          escrowAccountId: 1,
          platformFeeAccountId: 2,
          fraudScore: 0.12,
          fraudDecision: 'ALLOW',
        },
      });
      return {
        scenario: 'healthy',
        payoutId: payout.id,
        investigateUrl: `/investigate/payout/${payout.id}`,
        summary: 'ELIGIBLE payout, fraud ALLOW (score 0.12), no disputes',
      };
    }

    // Problematic: fraud BLOCK + max retries + FAILED + active dispute + seller blocked
    const tx = await this.prisma.transaction.create({
      data: { status: 'COMPLETED', description: 'Dev seed: problematic payout' },
    });

    const payout = await this.prisma.payout.create({
      data: {
        status: 'FAILED',
        amount: 25000,
        platformFee: 1250,
        sellerAmount: 23750,
        attempts: 3,
        maxAttempts: 3,
        failureReason: 'stripe_transfer_declined: account restricted',
        lastAttemptAt: new Date(),
        transactionId: tx.id,
        sellerId: seller.id,
        escrowAccountId: 1,
        platformFeeAccountId: 2,
        fraudScore: 0.87,
        fraudDecision: 'BLOCK',
      },
    });

    await this.prisma.dispute.create({
      data: {
        status: 'OPEN',
        reason: 'FRAUDULENT',
        amount: 25000,
        description: 'Buyer claims unauthorized transaction',
        transactionId: tx.id,
        payoutId: payout.id,
      },
    });

    await this.prisma.seller.update({
      where: { id: seller.id },
      data: { payoutsBlocked: true },
    });

    return {
      scenario: 'problematic',
      payoutId: payout.id,
      investigateUrl: `/investigate/payout/${payout.id}`,
      summary: 'FAILED payout, fraud BLOCK (score 0.87), 3/3 retries, OPEN dispute, seller blocked',
    };
  }

  /**
   * Aggregate all risk-relevant data for a seller into a single response.
   * Designed for fraud investigation — combines seller record, ledger balance,
   * and computed risk metrics in one round-trip.
   *
   * `excludePayoutId` omits exactly one payout — the one under investigation —
   * from `totalVolume24h`, and nothing else. Callers that feed the fraud engine
   * need this because `check_daily_volume` computes
   * `seller_total_amount_24h + amount`, i.e. it adds the payout being scored
   * back in itself; forwarding the unfiltered total counts that payout twice.
   * Mirrors the production call site (payout.service.ts::markEligible), which
   * passes prior settled volume for the same reason.
   *
   * The exclusion is deliberately narrow. It does NOT filter by status: a
   * seller with a stack of unsettled payouts is precisely the spike the
   * daily_volume rule exists to catch, so every other payout in the window —
   * PENDING included — stays in the total. It also leaves `payoutVelocity24h`
   * untouched, because `check_velocity` does NOT add the current payout back
   * (it compares `seller_payout_count_24h` directly), so the count must keep
   * including the payout under investigation to stay engine-accurate.
   */
  async getSellerRiskProfile(sellerId: number, excludePayoutId?: number) {
    const seller = await this.prisma.seller.findUnique({
      where: { id: sellerId },
    });

    if (!seller) {
      throw new NotFoundException(`Seller ${sellerId} not found`);
    }

    const now = Date.now();
    const twentyFourHoursAgo = new Date(now - 24 * 60 * 60 * 1000);

    // Parallel queries — all independent, no reason to serialize
    const [
      allPayouts,
      payouts24h,
      allDisputes,
      lostDisputes,
      { balance },
    ] = await Promise.all([
      this.prisma.payout.findMany({
        where: { sellerId },
        orderBy: { createdAt: 'asc' },
      }),
      this.prisma.payout.findMany({
        where: {
          sellerId,
          createdAt: { gte: twentyFourHoursAgo },
        },
      }),
      this.prisma.dispute.count({
        where: {
          transaction: { payouts: { some: { sellerId } } },
        },
      }),
      this.prisma.dispute.count({
        where: {
          status: 'LOST',
          transaction: { payouts: { some: { sellerId } } },
        },
      }),
      this.ledger.getAccountBalance(seller.accountId),
    ]);

    const statusCounts = { paid: 0, failed: 0, reversed: 0 };
    let totalVolumeLifetime = 0;
    let lastFailureDate: Date | null = null;

    for (const p of allPayouts) {
      if (p.status === 'PAID') statusCounts.paid++;
      if (p.status === 'FAILED') {
        statusCounts.failed++;
        if (!lastFailureDate || p.createdAt > lastFailureDate) {
          lastFailureDate = p.createdAt;
        }
      }
      if (p.status === 'REVERSED') statusCounts.reversed++;
      totalVolumeLifetime += p.amount;
    }

    // Filtered here rather than in the query: velocity must still count every
    // payout in the window, including the excluded one. See the method doc.
    const totalVolume24h = payouts24h
      .filter((p) => p.id !== excludePayoutId)
      .reduce((sum, p) => sum + p.amount, 0);
    const accountAgeDays = Math.floor(
      (now - seller.createdAt.getTime()) / (24 * 60 * 60 * 1000),
    );

    const firstPayout = allPayouts.length > 0 ? allPayouts[0] : null;

    let timeSinceLastFailure: string | null = null;
    if (lastFailureDate) {
      const diffMs = now - lastFailureDate.getTime();
      const diffHours = Math.floor(diffMs / (60 * 60 * 1000));
      const diffDays = Math.floor(diffHours / 24);
      timeSinceLastFailure =
        diffDays > 0 ? `${diffDays}d ${diffHours % 24}h` : `${diffHours}h`;
    }

    return {
      seller: {
        id: seller.id,
        name: seller.name,
        email: seller.email,
        status: seller.status,
        stripeAccountId: seller.stripeAccountId,
        chargesEnabled: seller.chargesEnabled,
        payoutsEnabled: seller.payoutsEnabled,
        payoutsBlocked: seller.payoutsBlocked,
        negativeBalance: seller.negativeBalance,
        accountAgeDays,
        createdAt: seller.createdAt.toISOString(),
      },
      ledger: {
        accountId: seller.accountId,
        balance,
      },
      riskMetrics: {
        totalPayouts: allPayouts.length,
        paidPayouts: statusCounts.paid,
        failedPayouts: statusCounts.failed,
        reversedPayouts: statusCounts.reversed,
        totalDisputes: allDisputes,
        lostDisputes,
        payoutVelocity24h: payouts24h.length,
        totalVolume24h,
        volume24hExcludesPayoutId: excludePayoutId ?? null,
        totalVolumeLifetime,
        avgPayoutAmount:
          allPayouts.length > 0
            ? Math.round(totalVolumeLifetime / allPayouts.length)
            : 0,
        accountAgeDays,
        firstPayoutDate: firstPayout ? firstPayout.createdAt.toISOString() : null,
        timeSinceLastFailure,
      },
    };
  }

  /**
   * Chronological timeline of seller's payouts with computed metadata.
   * Designed for pattern detection — trend analysis, velocity spikes,
   * failure clustering.
   */
  async getPayoutTimeline(sellerId: number, daysBack = 30) {
    const seller = await this.prisma.seller.findUnique({
      where: { id: sellerId },
    });

    if (!seller) {
      throw new NotFoundException(`Seller ${sellerId} not found`);
    }

    const since = new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000);

    const payouts = await this.prisma.payout.findMany({
      where: {
        sellerId,
        createdAt: { gte: since },
      },
      orderBy: { createdAt: 'desc' },
    });

    const terminalStatuses = new Set(['PAID', 'FAILED', 'REVERSED']);

    const timeline = payouts.map((p) => {
      let timeToCompletion: string | null = null;
      if (terminalStatuses.has(p.status)) {
        const endTime = p.paidAt ?? p.updatedAt;
        const diffMs = endTime.getTime() - p.createdAt.getTime();
        const diffMinutes = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMinutes / 60);
        timeToCompletion =
          diffHours > 0
            ? `${diffHours}h ${diffMinutes % 60}m`
            : `${diffMinutes}m`;
      }

      return {
        payoutId: p.id,
        createdAt: p.createdAt.toISOString(),
        amount: p.amount,
        sellerAmount: p.sellerAmount,
        status: p.status,
        fraudDecision: p.fraudDecision,
        fraudScore: p.fraudScore,
        failureReason: p.failureReason,
        timeToCompletion,
        transactionId: p.transactionId,
        attempts: p.attempts,
      };
    });

    // Status distribution
    const statusDistribution: Record<string, number> = {};
    for (const p of payouts) {
      statusDistribution[p.status] = (statusDistribution[p.status] || 0) + 1;
    }

    // Trend: compare volume in the older half vs the recent half of the window
    const midpoint = new Date(since.getTime() + (Date.now() - since.getTime()) / 2);
    let olderVolume = 0;
    let recentVolume = 0;
    for (const p of payouts) {
      if (p.createdAt < midpoint) olderVolume += p.amount;
      else recentVolume += p.amount;
    }

    let trend: 'increasing' | 'stable' | 'decreasing';
    if (olderVolume === 0 && recentVolume === 0) {
      trend = 'stable';
    } else if (olderVolume === 0 && recentVolume === 0) {
      trend = 'increasing';
    } else {
      const ratio = recentVolume / olderVolume;
      if (ratio > 1.3) trend = 'increasing';
      else if (ratio < 0.7) trend = 'decreasing';
      else trend = 'stable';
    }

    const totalAmount = payouts.reduce((sum, p) => sum + p.amount, 0);

    return {
      timeline,
      summary: {
        totalCount: payouts.length,
        statusDistribution,
        avgAmount: payouts.length > 0 ? Math.round(totalAmount / payouts.length) : 0,
        trend,
      },
    };
  }
}
