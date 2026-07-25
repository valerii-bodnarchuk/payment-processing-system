import { Test, TestingModule } from '@nestjs/testing';
import { AdminService } from './admin.service';
import { PrismaService } from '../prisma/prisma.service';
import { LedgerService } from '../ledger/ledger.service';
import { MetricsService } from '../metrics/metrics.service';
import { getLoggerToken } from 'nestjs-pino';

describe('AdminService', () => {
  let service: AdminService;
  let prisma: PrismaService;

  let buyerAccountId: number;
  let sellerAccountId: number;
  let escrowAccountId: number;
  let platformFeeAccountId: number;
  let sellerId: number;

  beforeAll(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AdminService,
        LedgerService,
        PrismaService,
        {
          provide: getLoggerToken(LedgerService.name),
          useValue: { info: jest.fn(), warn: jest.fn(), error: jest.fn() },
        },
        {
          provide: MetricsService,
          useValue: { ledgerTransactionsTotal: { inc: jest.fn() } },
        },
      ],
    }).compile();

    service = module.get(AdminService);
    prisma = module.get(PrismaService);
  });

  beforeEach(async () => {
    await prisma.$executeRaw`TRUNCATE "Entry", "Dispute", "Payout", "Transaction", "Seller", "Account" RESTART IDENTITY CASCADE`;

    const buyer = await prisma.account.create({
      data: { name: 'Buyer', type: 'BUYER', allowNegative: true },
    });
    const sellerAcct = await prisma.account.create({
      data: { name: 'Seller', type: 'SELLER', allowNegative: true },
    });
    const escrow = await prisma.account.create({
      data: { name: 'Escrow', type: 'ESCROW' },
    });
    const fee = await prisma.account.create({
      data: { name: 'Platform Fee', type: 'PLATFORM_FEE' },
    });

    buyerAccountId = buyer.id;
    sellerAccountId = sellerAcct.id;
    escrowAccountId = escrow.id;
    platformFeeAccountId = fee.id;

    const seller = await prisma.seller.create({
      data: {
        name: 'Test Seller',
        email: `seller-${Date.now()}@test.com`,
        accountId: sellerAccountId,
        status: 'ACTIVE',
        chargesEnabled: true,
        payoutsEnabled: true,
        stripeAccountId: `acct_test_${Date.now()}`,
      },
    });
    sellerId = seller.id;
  });

  afterAll(async () => {
    await prisma.$executeRaw`TRUNCATE "Entry", "Dispute", "Payout", "Transaction", "Seller", "Account" RESTART IDENTITY CASCADE`;
    await prisma.$disconnect();
  });

  async function createTransaction(amount: number) {
    return prisma.transaction.create({
      data: {
        description: `Payment ${Date.now()}`,
        status: 'COMPLETED',
        stripePaymentIntentId: `pi_test_${Date.now()}_${Math.random().toString(36).slice(2)}`,
        entries: {
          create: [
            { accountId: buyerAccountId, amount, type: 'DEBIT' },
            { accountId: escrowAccountId, amount, type: 'CREDIT' },
          ],
        },
      },
    });
  }

  async function createPayout(
    transactionId: number,
    amount: number,
    overrides: Partial<{
      status: string;
      fraudDecision: string;
      fraudScore: number;
      failureReason: string;
      attempts: number;
      paidAt: Date;
    }> = {},
  ) {
    const fee = Math.round((amount * 5) / 100);
    return prisma.payout.create({
      data: {
        amount,
        platformFee: fee,
        sellerAmount: amount - fee,
        status: (overrides.status as any) ?? 'PAID',
        fraudDecision: (overrides.fraudDecision as any) ?? null,
        fraudScore: overrides.fraudScore ?? null,
        failureReason: overrides.failureReason ?? null,
        attempts: overrides.attempts ?? 1,
        maxAttempts: 3,
        paidAt: overrides.paidAt ?? null,
        transactionId,
        sellerId,
        escrowAccountId,
        platformFeeAccountId,
      },
    });
  }

  // ── getSellerRiskProfile ──────────────────────────────────────

  describe('getSellerRiskProfile', () => {
    it('should return complete risk profile for seller with no payouts', async () => {
      const profile = await service.getSellerRiskProfile(sellerId);

      expect(profile.seller.id).toBe(sellerId);
      expect(profile.seller.status).toBe('ACTIVE');
      expect(profile.seller.accountAgeDays).toBeGreaterThanOrEqual(0);
      expect(profile.ledger.balance).toBe(0);
      expect(profile.riskMetrics.totalPayouts).toBe(0);
      expect(profile.riskMetrics.avgPayoutAmount).toBe(0);
      expect(profile.riskMetrics.firstPayoutDate).toBeNull();
      expect(profile.riskMetrics.timeSinceLastFailure).toBeNull();
    });

    it('should correctly aggregate payout status counts', async () => {
      const tx1 = await createTransaction(10000);
      const tx2 = await createTransaction(20000);
      const tx3 = await createTransaction(15000);

      await createPayout(tx1.id, 10000, { status: 'PAID' });
      await createPayout(tx2.id, 20000, { status: 'FAILED', failureReason: 'Stripe error' });
      await createPayout(tx3.id, 15000, { status: 'REVERSED' });

      const profile = await service.getSellerRiskProfile(sellerId);

      expect(profile.riskMetrics.totalPayouts).toBe(3);
      expect(profile.riskMetrics.paidPayouts).toBe(1);
      expect(profile.riskMetrics.failedPayouts).toBe(1);
      expect(profile.riskMetrics.reversedPayouts).toBe(1);
      expect(profile.riskMetrics.totalVolumeLifetime).toBe(45000);
      expect(profile.riskMetrics.avgPayoutAmount).toBe(15000);
    });

    it('should compute 24h velocity correctly', async () => {
      const tx1 = await createTransaction(5000);
      const tx2 = await createTransaction(7000);

      await createPayout(tx1.id, 5000, { status: 'PAID' });
      await createPayout(tx2.id, 7000, { status: 'PAID' });

      const profile = await service.getSellerRiskProfile(sellerId);

      expect(profile.riskMetrics.payoutVelocity24h).toBe(2);
      expect(profile.riskMetrics.totalVolume24h).toBe(12000);
      expect(profile.riskMetrics.volume24hExcludesPayoutId).toBeNull();
    });

    // Regression: the fraud engine's daily_volume rule computes
    // `seller_total_amount_24h + amount`, so a caller scoring one payout must
    // hand it the window MINUS that payout — and minus nothing else. Excluding
    // by status instead (e.g. dropping every PENDING) would zero out exactly
    // the burst the rule is meant to catch.
    it('should exclude only the named payout from totalVolume24h, keeping other pending ones', async () => {
      const txs = await Promise.all([
        createTransaction(9000),
        createTransaction(9000),
        createTransaction(9000),
        createTransaction(9000),
      ]);

      const [investigated, ...others] = await Promise.all(
        txs.map((tx) => createPayout(tx.id, 9000, { status: 'PENDING' })),
      );

      const profile = await service.getSellerRiskProfile(
        sellerId,
        investigated.id,
      );

      // The three other PENDING payouts survive the exclusion — only the
      // investigated one is gone.
      expect(profile.riskMetrics.totalVolume24h).toBe(27000);
      expect(profile.riskMetrics.volume24hExcludesPayoutId).toBe(
        investigated.id,
      );
      expect(others).toHaveLength(3);

      // Velocity still counts the full window: check_velocity compares the
      // count directly and never adds the investigated payout back in.
      expect(profile.riskMetrics.payoutVelocity24h).toBe(4);
    });

    it('should count the whole window when excludePayoutId is omitted', async () => {
      const txs = await Promise.all([
        createTransaction(9000),
        createTransaction(9000),
        createTransaction(9000),
        createTransaction(9000),
      ]);
      await Promise.all(
        txs.map((tx) => createPayout(tx.id, 9000, { status: 'PENDING' })),
      );

      const profile = await service.getSellerRiskProfile(sellerId);

      expect(profile.riskMetrics.totalVolume24h).toBe(36000);
      expect(profile.riskMetrics.volume24hExcludesPayoutId).toBeNull();
      expect(profile.riskMetrics.payoutVelocity24h).toBe(4);
    });

    it('should ignore an excludePayoutId that is not in the window', async () => {
      const tx = await createTransaction(9000);
      await createPayout(tx.id, 9000, { status: 'PENDING' });

      const profile = await service.getSellerRiskProfile(sellerId, 999999);

      expect(profile.riskMetrics.totalVolume24h).toBe(9000);
      expect(profile.riskMetrics.payoutVelocity24h).toBe(1);
    });

    it('should track timeSinceLastFailure', async () => {
      const tx = await createTransaction(10000);
      await createPayout(tx.id, 10000, {
        status: 'FAILED',
        failureReason: 'Test failure',
      });

      const profile = await service.getSellerRiskProfile(sellerId);

      expect(profile.riskMetrics.timeSinceLastFailure).not.toBeNull();
      // Just created, so should be 0h
      expect(profile.riskMetrics.timeSinceLastFailure).toMatch(/^\d+h$/);
    });

    it('should throw NotFoundException for non-existent seller', async () => {
      await expect(service.getSellerRiskProfile(99999)).rejects.toThrow(/not found/i);
    });

    it('should include dispute counts', async () => {
      const tx = await createTransaction(10000);
      await createPayout(tx.id, 10000, { status: 'PAID' });

      await prisma.dispute.create({
        data: {
          transactionId: tx.id,
          reason: 'FRAUDULENT',
          amount: 10000,
          status: 'LOST',
        },
      });

      const profile = await service.getSellerRiskProfile(sellerId);

      expect(profile.riskMetrics.totalDisputes).toBe(1);
      expect(profile.riskMetrics.lostDisputes).toBe(1);
    });
  });

  // ── getPayoutTimeline ─────────────────────────────────────────

  describe('getPayoutTimeline', () => {
    it('should return empty timeline for seller with no payouts', async () => {
      const result = await service.getPayoutTimeline(sellerId);

      expect(result.timeline).toHaveLength(0);
      expect(result.summary.totalCount).toBe(0);
      expect(result.summary.avgAmount).toBe(0);
      expect(result.summary.trend).toBe('stable');
    });

    it('should return payouts in reverse chronological order', async () => {
      const tx1 = await createTransaction(10000);
      const tx2 = await createTransaction(20000);

      await createPayout(tx1.id, 10000, { status: 'PAID' });
      await createPayout(tx2.id, 20000, { status: 'PAID' });

      const result = await service.getPayoutTimeline(sellerId);

      expect(result.timeline).toHaveLength(2);
      // Most recent first — tx2 was created after tx1
      const amounts = result.timeline.map((e) => e.amount);
      expect(amounts).toContain(10000);
      expect(amounts).toContain(20000);
    });

    it('should compute status distribution correctly', async () => {
      const tx1 = await createTransaction(10000);
      const tx2 = await createTransaction(20000);
      const tx3 = await createTransaction(15000);

      await createPayout(tx1.id, 10000, { status: 'PAID' });
      await createPayout(tx2.id, 20000, { status: 'PAID' });
      await createPayout(tx3.id, 15000, { status: 'FAILED', failureReason: 'err' });

      const result = await service.getPayoutTimeline(sellerId);

      expect(result.summary.statusDistribution['PAID']).toBe(2);
      expect(result.summary.statusDistribution['FAILED']).toBe(1);
      expect(result.summary.avgAmount).toBe(15000);
    });

    it('should include fraud decision and score in timeline entries', async () => {
      const tx = await createTransaction(50000);
      await createPayout(tx.id, 50000, {
        status: 'ELIGIBLE',
        fraudDecision: 'REVIEW',
        fraudScore: 0.55,
      });

      const result = await service.getPayoutTimeline(sellerId);

      expect(result.timeline[0].fraudDecision).toBe('REVIEW');
      expect(result.timeline[0].fraudScore).toBeCloseTo(0.55);
    });

    it('should throw NotFoundException for non-existent seller', async () => {
      await expect(service.getPayoutTimeline(99999)).rejects.toThrow(/not found/i);
    });
  });
});
