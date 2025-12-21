"""
Circuit Breaker
===============

Emergency stop mechanism to protect against catastrophic losses.

Monitors:
- Daily loss limits
- Maximum drawdown
- Consecutive losses
- Order rate limiting
- System errors

Author: Stoic Citadel Team
License: MIT
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading halted
    HALF_OPEN = "half_open"  # Testing if can resume


class TripReason(Enum):
    """Reason for circuit breaker trip."""

    DAILY_LOSS = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    ORDER_RATE = "order_rate_limit"
    SYSTEM_ERROR = "system_error"
    MANUAL = "manual_override"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Loss limits
    max_daily_loss_pct: float = 5.0  # 5% max daily loss
    max_drawdown_pct: float = 15.0  # 15% max drawdown from peak

    # Consecutive losses
    max_consecutive_losses: int = 5
    consecutive_loss_amount: float = 0.0  # Total loss from consecutive trades

    # Order rate limiting
    max_orders_per_minute: int = 10
    max_orders_per_hour: int = 100

    # Error limits
    max_consecutive_errors: int = 3

    # Recovery settings
    auto_reset_after_minutes: int = 60  # Auto-reset after 1 hour
    require_manual_reset: bool = False  # Require manual intervention

    # Cool-down period after trip
    cooldown_minutes: int = 15


@dataclass
class TripEvent:
    """Record of circuit breaker trip."""

    timestamp: datetime
    reason: TripReason
    details: str
    metrics: Dict
    reset_at: Optional[datetime] = None


class CircuitBreaker:
    """
    Circuit breaker to halt trading on dangerous conditions.

    Protects against:
    1. Excessive daily losses
    2. Large drawdowns
    3. Consecutive losing trades (strategy failure)
    4. Excessive order rates (potential bug/loop)
    5. System errors

    Usage:
        breaker = CircuitBreaker()

        # Before each trade
        if breaker.is_tripped:
            logger.error("Circuit breaker is OPEN - trading halted")
            return

        # After trade execution
        breaker.record_trade(pnl=trade_pnl)

        # Check if should trip
        breaker.check_and_trip(current_pnl, max_drawdown)
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED

        # Tracking
        self.daily_pnl: float = 0.0
        self.peak_balance: float = 0.0
        self.current_balance: float = 0.0
        self.consecutive_losses: int = 0
        self.consecutive_errors: int = 0

        # Order rate tracking
        self.order_timestamps: List[datetime] = []

        # Trip history
        self.trip_history: List[TripEvent] = []
        self.last_trip: Optional[TripEvent] = None
        self.trip_count_today: int = 0

        # Reset tracking
        self.last_reset: datetime = datetime.now()
        self.day_start: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        logger.info(
            f"Circuit Breaker initialized: "
            f"Daily loss limit: {self.config.max_daily_loss_pct}%, "
            f"Max drawdown: {self.config.max_drawdown_pct}%, "
            f"Max consecutive losses: {self.config.max_consecutive_losses}"
        )

    @property
    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped (trading halted)."""
        return self.state == CircuitBreakerState.OPEN

    @property
    def is_operational(self) -> bool:
        """Check if trading is allowed."""
        # Check if new day (auto-reset daily limits)
        self._check_daily_reset()

        # Check if auto-reset timer expired
        if self.state == CircuitBreakerState.OPEN:
            self._check_auto_reset()

        return self.state == CircuitBreakerState.CLOSED

    def check_and_trip(
        self, current_pnl: float, current_drawdown: float, force: bool = False
    ) -> bool:
        """
        Check conditions and trip if necessary.

        Args:
            current_pnl: Current daily PnL percentage
            current_drawdown: Current drawdown percentage
            force: Force trip (manual override)

        Returns:
            True if circuit breaker tripped
        """
        if force:
            return self._trip(TripReason.MANUAL, "Manual override", {})

        # Check daily loss limit
        if abs(current_pnl) > self.config.max_daily_loss_pct:
            return self._trip(
                TripReason.DAILY_LOSS,
                f"Daily loss {current_pnl:.2f}% exceeds limit {self.config.max_daily_loss_pct}%",
                {"daily_pnl": current_pnl},
            )

        # Check maximum drawdown
        if abs(current_drawdown) > self.config.max_drawdown_pct:
            return self._trip(
                TripReason.MAX_DRAWDOWN,
                f"Drawdown {current_drawdown:.2f}% exceeds limit {self.config.max_drawdown_pct}%",
                {"drawdown": current_drawdown},
            )

        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return self._trip(
                TripReason.CONSECUTIVE_LOSSES,
                f"{self.consecutive_losses} consecutive losses detected",
                {"consecutive_losses": self.consecutive_losses},
            )

        # Check order rate
        if not self._check_order_rate():
            return self._trip(
                TripReason.ORDER_RATE,
                "Order rate limit exceeded",
                {"orders_last_minute": self._count_recent_orders(1)},
            )

        return False

    def record_trade(self, pnl: float):
        """
        Record trade result and update consecutive loss counter.

        Args:
            pnl: Trade PnL
        """
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on winning trade

        logger.debug(
            f"Trade recorded: PnL={pnl:.2f}, "
            f"Daily PnL={self.daily_pnl:.2f}, "
            f"Consecutive losses={self.consecutive_losses}"
        )

    def record_order(self):
        """Record order submission for rate limiting."""
        now = datetime.now()
        self.order_timestamps.append(now)

        # Clean old timestamps (keep last hour)
        cutoff = now - timedelta(hours=1)
        self.order_timestamps = [ts for ts in self.order_timestamps if ts > cutoff]

    def record_error(self, error_msg: str):
        """
        Record system error.

        Args:
            error_msg: Error message
        """
        self.consecutive_errors += 1

        logger.error(f"Error recorded: {error_msg} (consecutive: {self.consecutive_errors})")

        # Trip on too many errors
        if self.consecutive_errors >= self.config.max_consecutive_errors:
            self._trip(
                TripReason.SYSTEM_ERROR,
                f"{self.consecutive_errors} consecutive errors",
                {"error": error_msg},
            )

    def clear_errors(self):
        """Clear error counter (after successful operation)."""
        if self.consecutive_errors > 0:
            logger.info("Clearing error counter")
            self.consecutive_errors = 0

    def update_balance(self, current: float, peak: Optional[float] = None):
        """
        Update balance for drawdown calculation.

        Args:
            current: Current account balance
            peak: Peak balance (optional)
        """
        self.current_balance = current

        if peak is not None:
            self.peak_balance = peak
        elif current > self.peak_balance:
            self.peak_balance = current

    def calculate_drawdown(self) -> float:
        """
        Calculate current drawdown percentage.

        Returns:
            Drawdown percentage (negative value)
        """
        if self.peak_balance == 0:
            return 0.0

        drawdown = ((self.current_balance - self.peak_balance) / self.peak_balance) * 100
        return drawdown

    def reset(self, manual: bool = True):
        """
        Reset circuit breaker to operational state.

        Args:
            manual: Whether this is a manual reset
        """
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED

        if self.last_trip:
            self.last_trip.reset_at = datetime.now()

        self.last_reset = datetime.now()

        logger.warning(
            f"Circuit Breaker RESET: {old_state.value} → {self.state.value} "
            f"({'manual' if manual else 'automatic'})"
        )

    def _trip(self, reason: TripReason, details: str, metrics: Dict) -> bool:
        """
        Trip the circuit breaker (halt trading).

        Args:
            reason: Reason for tripping
            details: Detailed description
            metrics: Related metrics

        Returns:
            True (always trips)
        """
        self.state = CircuitBreakerState.OPEN

        trip_event = TripEvent(
            timestamp=datetime.now(), reason=reason, details=details, metrics=metrics
        )

        self.trip_history.append(trip_event)
        self.last_trip = trip_event
        self.trip_count_today += 1

        logger.critical(
            f"🚨 CIRCUIT BREAKER TRIPPED 🚨\n"
            f"Reason: {reason.value}\n"
            f"Details: {details}\n"
            f"Metrics: {metrics}\n"
            f"Trading HALTED"
        )

        # Send alert (implement notification system)
        self._send_alert(trip_event)

        return True

    def _check_order_rate(self) -> bool:
        """Check if order rate is within limits."""
        now = datetime.now()

        # Count orders in last minute
        orders_last_minute = self._count_recent_orders(1)
        if orders_last_minute > self.config.max_orders_per_minute:
            logger.warning(
                f"Order rate limit exceeded: {orders_last_minute} orders in last minute "
                f"(limit: {self.config.max_orders_per_minute})"
            )
            return False

        # Count orders in last hour
        orders_last_hour = self._count_recent_orders(60)
        if orders_last_hour > self.config.max_orders_per_hour:
            logger.warning(
                f"Order rate limit exceeded: {orders_last_hour} orders in last hour "
                f"(limit: {self.config.max_orders_per_hour})"
            )
            return False

        return True

    def _count_recent_orders(self, minutes: int) -> int:
        """Count orders in last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return sum(1 for ts in self.order_timestamps if ts > cutoff)

    def _check_auto_reset(self):
        """Check if circuit breaker should auto-reset."""
        if self.config.require_manual_reset:
            return

        if not self.last_trip:
            return

        elapsed = (datetime.now() - self.last_trip.timestamp).total_seconds() / 60

        if elapsed > self.config.auto_reset_after_minutes:
            logger.info(f"Auto-resetting circuit breaker after {elapsed:.0f} minutes")
            self.reset(manual=False)

    def _check_daily_reset(self):
        """Check if new trading day started (reset daily limits)."""
        now = datetime.now()
        current_day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if current_day_start > self.day_start:
            logger.info("New trading day - resetting daily limits")
            self.day_start = current_day_start
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.trip_count_today = 0

    def _send_alert(self, trip_event: TripEvent):
        """
        Send alert notification.

        TODO: Implement actual notification system (Slack, email, etc.)
        """
        # Placeholder for notification system
        logger.critical(f"ALERT: Circuit breaker tripped - {trip_event.reason.value}")

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "is_tripped": self.is_tripped,
            "is_operational": self.is_operational,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_errors": self.consecutive_errors,
            "current_drawdown": self.calculate_drawdown(),
            "orders_last_minute": self._count_recent_orders(1),
            "orders_last_hour": self._count_recent_orders(60),
            "trip_count_today": self.trip_count_today,
            "last_trip": self.last_trip.details if self.last_trip else None,
            "last_reset": self.last_reset.isoformat(),
        }
