"""
Smart Limit Order Executor
===========================

Advanced limit order execution strategy that minimizes fees by:
1. Posting limit orders at best bid/ask (Maker orders = lower fees)
2. Chasing the price if it moves away
3. Converting to market order only if necessary (timeout)

This approach saves significant fees compared to market orders:
- Market order (Taker): ~0.1% fee
- Limit order (Maker): ~0.02% fee or even rebates

For a $10,000 trade:
- Market: $10 fee
- Limit: $2 fee (or $0 with rebate)
- Savings: $8 per trade

Reference: HFT best practices for fee optimization
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from enum import Enum
import time
import logging

from src.order_manager.order_types import Order, OrderType, OrderSide, OrderStatus, LimitOrder

logger = logging.getLogger(__name__)


class ChasingStrategy(Enum):
    """Limit order chasing strategy."""
    PASSIVE = "passive"          # Place at bid/ask, don't chase
    MODERATE = "moderate"        # Chase slowly (every 5 seconds)
    AGGRESSIVE = "aggressive"    # Chase quickly (every 1 second)
    ADAPTIVE = "adaptive"        # Adjust based on volatility


@dataclass
class SmartLimitConfig:
    """Configuration for smart limit order execution."""

    # Chasing behavior
    chasing_strategy: ChasingStrategy = ChasingStrategy.MODERATE

    # Timeouts
    max_wait_seconds: float = 30.0     # Max time to wait for fill
    chase_interval_seconds: float = 3.0  # How often to update price

    # Price offset from best bid/ask (in bps)
    initial_offset_bps: float = 0.0    # 0 = exactly at best bid/ask
    max_offset_bps: float = 5.0        # Max offset before converting to market

    # Convert to market order if not filled within threshold
    convert_to_market: bool = True
    market_conversion_threshold: float = 0.8  # Convert after 80% of max_wait

    # Partial fill handling
    allow_partial_fills: bool = True
    min_fill_ratio: float = 0.5        # Minimum 50% fill to consider success

    # Fee assumptions (for cost calculation)
    maker_fee_bps: float = 2.0         # 0.02% maker fee
    taker_fee_bps: float = 10.0        # 0.10% taker fee

    # Spread threshold - don't post if spread is too wide
    max_spread_bps: float = 50.0       # 0.5% max spread


@dataclass
class SmartExecutionResult:
    """Result of smart limit order execution."""

    success: bool
    order: Order
    execution_type: str  # "maker", "taker", "partial", "failed"
    average_price: Optional[float] = None
    filled_quantity: float = 0.0
    total_fee: float = 0.0
    fee_saved_vs_market: float = 0.0
    chase_count: int = 0
    total_time_seconds: float = 0.0
    error_message: Optional[str] = None


class SmartLimitExecutor:
    """
    Executes orders using smart limit order strategy.

    Goal: Minimize trading fees by being a Maker (provider of liquidity)
    rather than a Taker (consumer of liquidity).

    Usage:
        executor = SmartLimitExecutor(config)

        # Get current order book
        orderbook = exchange.fetch_order_book("BTC/USDT")

        # Execute with smart limits
        result = executor.execute(
            order=buy_order,
            exchange_api=exchange,
            orderbook=orderbook
        )

        print(f"Saved ${result.fee_saved_vs_market} in fees!")
    """

    def __init__(self, config: Optional[SmartLimitConfig] = None):
        """Initialize smart limit executor."""
        self.config = config or SmartLimitConfig()
        self.execution_history: list[SmartExecutionResult] = []
        self.total_fees_saved = 0.0

        logger.info(
            f"SmartLimitExecutor initialized with {self.config.chasing_strategy.value} strategy"
        )

    def execute(
        self,
        order: Order,
        exchange_api: Any,
        orderbook: Dict[str, Any],
        on_update: Optional[Callable[[str, float], None]] = None
    ) -> SmartExecutionResult:
        """
        Execute order using smart limit strategy.

        Args:
            order: Order to execute
            exchange_api: Exchange API instance
            orderbook: Current order book {bids: [[price, size], ...], asks: [...]}
            on_update: Optional callback for price updates (status, price)

        Returns:
            SmartExecutionResult with execution details
        """
        start_time = time.time()

        # Validate inputs
        if not orderbook or not orderbook.get("bids") or not orderbook.get("asks"):
            return self._create_failure_result(
                order, "Invalid order book data", start_time
            )

        # Check spread
        best_bid = orderbook["bids"][0][0] if orderbook["bids"] else None
        best_ask = orderbook["asks"][0][0] if orderbook["asks"] else None

        if not best_bid or not best_ask:
            return self._create_failure_result(
                order, "No bid/ask in order book", start_time
            )

        spread_bps = ((best_ask - best_bid) / best_bid) * 10000

        if spread_bps > self.config.max_spread_bps:
            logger.warning(
                f"Spread too wide: {spread_bps:.1f} bps > {self.config.max_spread_bps} bps"
            )
            # Fall back to market order
            return self._execute_as_market(order, exchange_api, best_ask if order.is_buy else best_bid, start_time)

        # Calculate initial limit price
        limit_price = self._calculate_limit_price(order, best_bid, best_ask)

        logger.info(
            f"Smart limit execution: {order.symbol} {order.side.value} "
            f"@ {limit_price:.2f} (spread: {spread_bps:.1f} bps)"
        )

        # Execute limit order with chasing
        return self._execute_with_chasing(
            order=order,
            exchange_api=exchange_api,
            initial_price=limit_price,
            start_time=start_time,
            on_update=on_update
        )

    def _calculate_limit_price(
        self,
        order: Order,
        best_bid: float,
        best_ask: float
    ) -> float:
        """
        Calculate initial limit price.

        For BUY: Place at best bid + offset (try to be filled first)
        For SELL: Place at best ask - offset
        """
        offset_ratio = self.config.initial_offset_bps / 10000

        if order.is_buy:
            # Place slightly above best bid (but below ask)
            price = best_bid * (1 + offset_ratio)
            # Don't exceed best ask
            price = min(price, best_ask * 0.9999)
        else:
            # Place slightly below best ask (but above bid)
            price = best_ask * (1 - offset_ratio)
            # Don't go below best bid
            price = max(price, best_bid * 1.0001)

        return round(price, 8)

    def _execute_with_chasing(
        self,
        order: Order,
        exchange_api: Any,
        initial_price: float,
        start_time: float,
        on_update: Optional[Callable] = None
    ) -> SmartExecutionResult:
        """Execute limit order and chase if needed."""
        current_price = initial_price
        chase_count = 0
        filled_quantity = 0.0
        total_fill_value = 0.0
        exchange_order_id = None

        # Place initial limit order
        try:
            result = self._place_limit_order(order, current_price, exchange_api)
            if not result["success"]:
                return self._create_failure_result(
                    order, result.get("error", "Failed to place limit order"), start_time
                )
            exchange_order_id = result.get("order_id")

            if on_update:
                on_update("placed", current_price)

        except Exception as e:
            return self._create_failure_result(order, str(e), start_time)

        # Chasing loop
        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed >= self.config.max_wait_seconds:
                logger.info(f"Timeout reached after {elapsed:.1f}s")
                break

            # Check if should convert to market
            if self.config.convert_to_market:
                convert_threshold = self.config.max_wait_seconds * self.config.market_conversion_threshold
                if elapsed >= convert_threshold:
                    logger.info(f"Converting to market order after {elapsed:.1f}s")
                    # Cancel limit and place market
                    self._cancel_order(exchange_order_id, order.symbol, exchange_api)
                    return self._execute_as_market(
                        order, exchange_api, current_price, start_time,
                        already_filled=filled_quantity
                    )

            # Check order status
            try:
                status = self._check_order_status(exchange_order_id, order.symbol, exchange_api)

                if status["status"] == "closed":
                    # Fully filled as maker!
                    filled_quantity = status.get("filled", order.quantity)
                    avg_price = status.get("average", current_price)
                    fee = self._calculate_maker_fee(filled_quantity * avg_price)

                    # Calculate savings
                    market_fee = self._calculate_taker_fee(filled_quantity * avg_price)
                    fee_saved = market_fee - fee

                    self.total_fees_saved += fee_saved

                    order.update_fill(filled_quantity, avg_price, fee)

                    return SmartExecutionResult(
                        success=True,
                        order=order,
                        execution_type="maker",
                        average_price=avg_price,
                        filled_quantity=filled_quantity,
                        total_fee=fee,
                        fee_saved_vs_market=fee_saved,
                        chase_count=chase_count,
                        total_time_seconds=time.time() - start_time
                    )

                elif status["status"] == "partially_filled":
                    # Track partial fill
                    filled_quantity = status.get("filled", 0)
                    if on_update:
                        on_update("partial", filled_quantity)

            except Exception as e:
                logger.warning(f"Error checking order status: {e}")

            # Wait before chasing
            time.sleep(self.config.chase_interval_seconds)

            # Chase: update price if needed
            try:
                orderbook = exchange_api.fetch_order_book(order.symbol)
                new_price = self._calculate_chase_price(
                    order, current_price, orderbook, chase_count
                )

                if new_price != current_price:
                    # Cancel and replace
                    self._cancel_order(exchange_order_id, order.symbol, exchange_api)

                    result = self._place_limit_order(order, new_price, exchange_api)
                    if result["success"]:
                        exchange_order_id = result.get("order_id")
                        current_price = new_price
                        chase_count += 1

                        logger.info(
                            f"Chase #{chase_count}: Updated price to {new_price:.2f}"
                        )

                        if on_update:
                            on_update("chased", new_price)

            except Exception as e:
                logger.warning(f"Error during chase: {e}")

        # Timeout - cancel remaining order
        if exchange_order_id:
            self._cancel_order(exchange_order_id, order.symbol, exchange_api)

        # Return result based on what was filled
        if filled_quantity > 0:
            if filled_quantity / order.quantity >= self.config.min_fill_ratio:
                # Partial success
                return SmartExecutionResult(
                    success=True,
                    order=order,
                    execution_type="partial",
                    average_price=current_price,
                    filled_quantity=filled_quantity,
                    total_fee=self._calculate_maker_fee(filled_quantity * current_price),
                    chase_count=chase_count,
                    total_time_seconds=time.time() - start_time
                )

        return self._create_failure_result(
            order, f"Timeout after {chase_count} chases", start_time, chase_count
        )

    def _calculate_chase_price(
        self,
        order: Order,
        current_price: float,
        orderbook: Dict,
        chase_count: int
    ) -> float:
        """Calculate new price for chasing."""
        best_bid = orderbook["bids"][0][0] if orderbook.get("bids") else current_price
        best_ask = orderbook["asks"][0][0] if orderbook.get("asks") else current_price

        # Increase offset with each chase
        offset_bps = min(
            self.config.initial_offset_bps + (chase_count * 1.0),
            self.config.max_offset_bps
        )
        offset_ratio = offset_bps / 10000

        if order.is_buy:
            new_price = best_bid * (1 + offset_ratio)
            # Check if we need to chase
            if new_price > current_price * 1.0001:  # Significant change
                return round(min(new_price, best_ask * 0.9999), 8)
        else:
            new_price = best_ask * (1 - offset_ratio)
            if new_price < current_price * 0.9999:
                return round(max(new_price, best_bid * 1.0001), 8)

        return current_price

    def _place_limit_order(
        self,
        order: Order,
        price: float,
        exchange_api: Any
    ) -> Dict:
        """Place limit order on exchange."""
        try:
            result = exchange_api.create_limit_order(
                symbol=order.symbol,
                side=order.side.value,
                amount=order.quantity,
                price=price
            )
            return {
                "success": True,
                "order_id": result.get("id"),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _cancel_order(self, order_id: str, symbol: str, exchange_api: Any) -> bool:
        """Cancel order on exchange."""
        try:
            exchange_api.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    def _check_order_status(
        self,
        order_id: str,
        symbol: str,
        exchange_api: Any
    ) -> Dict:
        """Check order status on exchange."""
        result = exchange_api.fetch_order(order_id, symbol)
        return {
            "status": result.get("status"),
            "filled": result.get("filled", 0),
            "average": result.get("average", 0),
        }

    def _execute_as_market(
        self,
        order: Order,
        exchange_api: Any,
        reference_price: float,
        start_time: float,
        already_filled: float = 0.0
    ) -> SmartExecutionResult:
        """Fall back to market order execution."""
        remaining_quantity = order.quantity - already_filled

        if remaining_quantity <= 0:
            return SmartExecutionResult(
                success=True,
                order=order,
                execution_type="maker",
                average_price=reference_price,
                filled_quantity=already_filled,
                total_fee=self._calculate_maker_fee(already_filled * reference_price),
                total_time_seconds=time.time() - start_time
            )

        try:
            result = exchange_api.create_market_order(
                symbol=order.symbol,
                side=order.side.value,
                amount=remaining_quantity
            )

            avg_price = result.get("average", reference_price)
            fee = self._calculate_taker_fee(remaining_quantity * avg_price)

            total_filled = already_filled + remaining_quantity
            total_fee = fee + (
                self._calculate_maker_fee(already_filled * reference_price) if already_filled > 0 else 0
            )

            order.update_fill(remaining_quantity, avg_price, fee)

            return SmartExecutionResult(
                success=True,
                order=order,
                execution_type="taker" if already_filled == 0 else "partial",
                average_price=avg_price,
                filled_quantity=total_filled,
                total_fee=total_fee,
                fee_saved_vs_market=0.0,  # No savings when using market order
                total_time_seconds=time.time() - start_time
            )

        except Exception as e:
            return self._create_failure_result(order, str(e), start_time)

    def _calculate_maker_fee(self, value: float) -> float:
        """Calculate maker fee."""
        return value * (self.config.maker_fee_bps / 10000)

    def _calculate_taker_fee(self, value: float) -> float:
        """Calculate taker fee."""
        return value * (self.config.taker_fee_bps / 10000)

    def _create_failure_result(
        self,
        order: Order,
        error: str,
        start_time: float,
        chase_count: int = 0
    ) -> SmartExecutionResult:
        """Create failure result."""
        return SmartExecutionResult(
            success=False,
            order=order,
            execution_type="failed",
            error_message=error,
            chase_count=chase_count,
            total_time_seconds=time.time() - start_time
        )

    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        total_executions = len(self.execution_history)
        maker_count = sum(1 for r in self.execution_history if r.execution_type == "maker")
        taker_count = sum(1 for r in self.execution_history if r.execution_type == "taker")

        return {
            "total_executions": total_executions,
            "maker_fills": maker_count,
            "taker_fills": taker_count,
            "maker_ratio": (maker_count / max(1, total_executions)) * 100,
            "total_fees_saved": self.total_fees_saved,
            "average_chase_count": (
                sum(r.chase_count for r in self.execution_history) / max(1, total_executions)
            )
        }
