import { Controller, Post, Body, Headers } from '@nestjs/common';
import { ApiTags, ApiHeader } from '@nestjs/swagger';
import { Throttle } from '@nestjs/throttler'; // use the built-in throttler
import { PaymentService } from './payment.service';
import { CreatePaymentDto } from './dto/create-payment.dto';

@ApiTags('Payment')
@Controller('payments')
export class PaymentController {
  constructor(private paymentService: PaymentService) {}

  @Post()
  @Throttle({ default: { limit: 10, ttl: 60000 } })
  @ApiHeader({ name: 'idempotency-key', required: false })
  async createPayment(
    @Body() body: CreatePaymentDto,
    @Headers('idempotency-key') idempotencyKey?: string,
  ) {
    return this.paymentService.createPayment(body, idempotencyKey);
  }
}