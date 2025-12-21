"""
Order Types and State Machine
==============================

Defines all order types and their lifecycle management.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status in its lifecycle."""

    PENDING = "pending"  # Created but not submitted
    SUBMITTED = "submitted"  # Submitted to exchange
    OPEN = "open"  # Accepted by exchange
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"  # Fully executed
    CANCELLED = "cancelled"  # Cancelled by user/system
    REJECTED = "rejected"  # Rejected by exchange
    EXPIRED = "expired"  # Expired (time in force)
    FAILED = "failed"  # Technical failure


@dataclass
class Order:
    """
    Base order class representing a trading order.

    Lifecycle:
        PENDING → SUBMITTED → OPEN → [PARTIALLY_FILLED] → FILLED
                           ↓
                    CANCELLED/REJECTED/EXPIRED/FAILED
    """

    # Identity
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None

    # Basic order info
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY

    # Quantities
    quantity: float = 0.0  # Requested quantity
    filled_quantity: float = 0.0  # Filled so far
    remaining_quantity: float = 0.0  # Remaining to fill

    # Prices
    price: Optional[float] = None  # Limit price (for limit orders)
    stop_price: Optional[float] = None  # Stop trigger price
    average_fill_price: Optional[float] = None

    # Status
    status: OrderStatus = OrderStatus.PENDING

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)

    # Metadata
    strategy_name: Optional[str] = None
    position_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    # Fees and costs
    commission: float = 0.0
    commission_asset: str = "USDT"

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Timeout management
    timeout_seconds: int = 300  # 5 minutes default timeout
    timeout_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize calculated fields."""
        self.remaining_quantity = self.quantity
        if self.client_order_id is None:
            self.client_order_id = f"stoic_{self.order_id[:8]}"

    @property
    def is_buy(self) -> bool:
        """Check if order is a buy."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Check if order is a sell."""
        return self.side == OrderSide.SELL

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is active (can be filled)."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        ]

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        ]

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage (0-100)."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    @property
    def is_timed_out(self) -> bool:
        """
        Check if order has exceeded timeout.

        Returns:
            True if order is timed out and should be cancelled
        """
        if not self.is_active:
            return False

        if self.timeout_at is None:
            # Set timeout on first check
            self._set_timeout()
            return False

        return datetime.now() >= self.timeout_at

    def _set_timeout(self) -> None:
        """Calculate and set timeout timestamp."""
        from datetime import timedelta

        base_time = self.submitted_at or self.created_at
        self.timeout_at = base_time + timedelta(seconds=self.timeout_seconds)

        logger.debug(
            f"Order {self.order_id} timeout set to {self.timeout_at} "
            f"({self.timeout_seconds}s from {base_time})"
        )

    def check_timeout(self) -> bool:
        """
        Check if order has timed out and handle accordingly.

        Returns:
            True if order timed out (and status updated to EXPIRED)
        """
        if self.is_timed_out:
            logger.warning(
                f"Order {self.order_id} timed out after {self.timeout_seconds}s. "
                f"Status: {self.status.value}, Fill: {self.fill_percentage:.1f}%"
            )

            # Update status to EXPIRED
            self.update_status(
                OrderStatus.EXPIRED, error=f"Order timed out after {self.timeout_seconds} seconds"
            )

            return True

        return False

    def extend_timeout(self, additional_seconds: int = 60) -> None:
        """
        Extend order timeout by additional seconds.

        Useful for orders that are partially filled and need more time.

        Args:
            additional_seconds: Seconds to add to timeout
        """
        from datetime import timedelta

        if self.timeout_at is None:
            self._set_timeout()

        old_timeout = self.timeout_at
        self.timeout_at += timedelta(seconds=additional_seconds)

        logger.info(
            f"Order {self.order_id} timeout extended from {old_timeout} "
            f"to {self.timeout_at} (+{additional_seconds}s)"
        )

    def update_status(self, new_status: OrderStatus, error: Optional[str] = None):
        """
        Update order status with validation.

        Args:
            new_status: New status to set
            error: Optional error message
        """
        old_status = self.status

        # Validate state transition
        if not self._is_valid_transition(old_status, new_status):
            logger.warning(
                f"Invalid status transition for order {self.order_id}: "
                f"{old_status.value} → {new_status.value}"
            )
            return

        self.status = new_status
        self.updated_at = datetime.now()

        if error:
            self.error_message = error

        # Set timestamps
        if new_status == OrderStatus.SUBMITTED and self.submitted_at is None:
            self.submitted_at = datetime.now()
        elif new_status == OrderStatus.FILLED and self.filled_at is None:
            self.filled_at = datetime.now()

        logger.info(f"Order {self.order_id} status: {old_status.value} → {new_status.value}")

    def _is_valid_transition(self, from_status: OrderStatus, to_status: OrderStatus) -> bool:
        """Validate if status transition is allowed."""
        # Terminal states cannot transition
        if from_status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        ]:
            return False

        # Allow any transition from PENDING
        if from_status == OrderStatus.PENDING:
            return True

        # Common valid transitions
        valid_transitions = {
            OrderStatus.SUBMITTED: [OrderStatus.OPEN, OrderStatus.REJECTED, OrderStatus.FAILED],
            OrderStatus.OPEN: [
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.EXPIRED,
            ],
            OrderStatus.PARTIALLY_FILLED: [OrderStatus.FILLED, OrderStatus.CANCELLED],
        }

        return to_status in valid_transitions.get(from_status, [])

    def update_fill(self, filled_qty: float, fill_price: float, commission: float = 0.0):
        """
        Update order with partial or full fill.

        Args:
            filled_qty: Quantity filled in this update
            fill_price: Price at which filled
            commission: Commission charged
        """
        self.filled_quantity += filled_qty
        self.remaining_quantity = max(0, self.quantity - self.filled_quantity)
        self.commission += commission

        # Update average fill price
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            # Weighted average
            total_filled = self.filled_quantity
            self.average_fill_price = (
                self.average_fill_price * (total_filled - filled_qty) + fill_price * filled_qty
            ) / total_filled

        # Update status
        if self.remaining_quantity <= 0.000001:  # Account for floating point
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)

        self.updated_at = datetime.now()

        logger.info(
            f"Order {self.order_id} filled: {filled_qty} @ {fill_price} "
            f"({self.fill_percentage:.1f}% complete)"
        )

    def can_retry(self) -> bool:
        """Check if order can be retried."""
        return self.retry_count < self.max_retries and not self.is_terminal

    def increment_retry(self):
        """Increment retry counter."""
        self.retry_count += 1
        logger.info(f"Order {self.order_id} retry {self.retry_count}/{self.max_retries}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "side": self.side.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "average_fill_price": self.average_fill_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "commission": self.commission,
            "commission_asset": self.commission_asset,
            "error_message": self.error_message,
            "strategy_name": self.strategy_name,
            "tags": self.tags,
        }


@dataclass
class MarketOrder(Order):
    """Market order - executes immediately at best available price."""

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.MARKET


@dataclass
class LimitOrder(Order):
    """Limit order - executes at specified price or better."""

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.LIMIT

        if self.price is None:
            raise ValueError("Limit order requires price")


@dataclass
class StopLossOrder(Order):
    """Stop-loss order - triggers when price crosses stop price."""

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.STOP_LOSS

        if self.stop_price is None:
            raise ValueError("Stop-loss order requires stop_price")


@dataclass
class TakeProfitOrder(Order):
    """Take-profit order - triggers when price reaches profit target."""

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.TAKE_PROFIT

        if self.stop_price is None:
            raise ValueError("Take-profit order requires stop_price")


@dataclass
class TrailingStopOrder(Order):
    """Trailing stop order - stop price trails market price."""

    trailing_distance: float = 0.0  # Distance to trail (in price or percentage)
    trailing_percent: bool = False  # Use percentage instead of absolute
    highest_price: Optional[float] = None  # Highest price seen (for buy trailing)
    lowest_price: Optional[float] = None  # Lowest price seen (for sell trailing)

    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.TRAILING_STOP

        if self.trailing_distance <= 0:
            raise ValueError("Trailing stop requires positive trailing_distance")

    def update_trailing_stop(self, current_price: float):
        """
        Update trailing stop price based on current market price.

        Args:
            current_price: Current market price
        """
        if self.is_buy:
            # For buy orders, trail below lowest price
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price

                if self.trailing_percent:
                    self.stop_price = self.lowest_price * (1 + self.trailing_distance / 100)
                else:
                    self.stop_price = self.lowest_price + self.trailing_distance
        else:
            # For sell orders, trail above highest price
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price

                if self.trailing_percent:
                    self.stop_price = self.highest_price * (1 - self.trailing_distance / 100)
                else:
                    self.stop_price = self.highest_price - self.trailing_distance

        logger.debug(
            f"Trailing stop updated for {self.order_id}: "
            f"stop_price={self.stop_price:.2f}, current={current_price:.2f}"
        )
