"""
Stoic Citadel - Circuit Breaker System
======================================

Emergency stop mechanism for trading bot:
- Daily loss limit protection
- Consecutive losses protection
- Drawdown protection
- Volatility circuit breaker
- Manual override capability
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

# Try to import metrics exporter
try:
    from src.monitoring.metrics_exporter import get_exporter

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading halted
    HALF_OPEN = "half_open"  # Testing recovery


class TripReason(Enum):
    """Reasons for circuit breaker trip."""

    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    MAX_DRAWDOWN = "max_drawdown"
    HIGH_VOLATILITY = "high_volatility"
    API_ERRORS = "api_errors"
    MANUAL_STOP = "manual_stop"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Loss limits
    daily_loss_limit_pct: float = 0.05  # 5% daily loss limit
    consecutive_loss_limit: int = 5  # Max consecutive losses
    max_drawdown_pct: float = 0.15  # 15% max drawdown

    # Volatility limits
    max_volatility_threshold: float = 0.08  # 8% volatility threshold
    volatility_lookback_hours: int = 24

    # API error limits
    max_api_errors: int = 10
    api_error_window_minutes: int = 5

    # Recovery settings
    cooldown_minutes: int = 30
    recovery_trade_size_pct: float = 0.25  # 25% of normal size
    recovery_trades_required: int = 3

    # Auto-reset
    auto_reset_after_hours: int = 24

    # Adaptive thresholds
    enable_adaptive_thresholds: bool = True
    baseline_volatility: Optional[float] = None  # Will be learned from data
    max_volatility_multiplier: float = 2.0  # Max adaptive increase


@dataclass
class TradingSession:
    """Track current trading session metrics."""

    start_time: datetime = field(default_factory=datetime.utcnow)
    initial_balance: float = 0.0
    current_balance: float = 0.0
    peak_balance: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    consecutive_losses: int = 0
    api_errors: List[datetime] = field(default_factory=list)

    @property
    def daily_pnl_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return (self.current_balance - self.initial_balance) / self.initial_balance

    @property
    def drawdown_pct(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance


class CircuitBreaker:
    """
    Circuit breaker for trading risk management.

    Automatically halts trading when risk limits are breached,
    with gradual recovery mechanism.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.trip_reason: Optional[TripReason] = None
        self.trip_time: Optional[datetime] = None
        self.session = TradingSession()
        self.recovery_trades: int = 0
        self._lock = threading.RLock()
        self._callbacks: List[callable] = []

    def initialize_session(self, initial_balance: float) -> None:
        """Initialize or reset trading session."""
        with self._lock:
            self.session = TradingSession(
                start_time=datetime.utcnow(),
                initial_balance=initial_balance,
                current_balance=initial_balance,
                peak_balance=initial_balance,
            )
            logger.info(f"Session initialized with balance: ${initial_balance:,.2f}")

    def update_balance(self, new_balance: float) -> None:
        """Update current balance and track peak."""
        with self._lock:
            self.session.current_balance = new_balance
            if new_balance > self.session.peak_balance:
                self.session.peak_balance = new_balance

    def record_trade(self, trade: Dict, profit_pct: float) -> None:
        """
        Record trade result and check for circuit breaker conditions.

        Args:
            trade: Trade details dictionary
            profit_pct: Trade profit/loss percentage
        """
        with self._lock:
            self.session.trades.append(trade)

            # Track consecutive losses
            if profit_pct < 0:
                self.session.consecutive_losses += 1
            else:
                self.session.consecutive_losses = 0

            # Update balance
            new_balance = self.session.current_balance * (1 + profit_pct)
            self.update_balance(new_balance)

            # Check circuit breaker conditions
            self._check_conditions()

    def record_api_error(self) -> None:
        """Record API error and check error rate."""
        with self._lock:
            now = datetime.utcnow()
            self.session.api_errors.append(now)

            # Clean old errors
            cutoff = now - timedelta(minutes=self.config.api_error_window_minutes)
            self.session.api_errors = [t for t in self.session.api_errors if t > cutoff]

            # Check error rate
            if len(self.session.api_errors) >= self.config.max_api_errors:
                self._trip(TripReason.API_ERRORS)

    def check_volatility(self, current_volatility: float) -> None:
        """
        Check if volatility exceeds threshold.

        Uses adaptive thresholds based on market regime.
        """
        threshold = self._get_adaptive_volatility_threshold(current_volatility)

        if current_volatility > threshold:
            logger.warning(
                f"High volatility detected: {current_volatility:.2%} "
                f"(threshold: {threshold:.2%})"
            )
            self._trip(TripReason.HIGH_VOLATILITY)

    def _get_adaptive_volatility_threshold(self, current_volatility: float) -> float:
        """
        Calculate adaptive volatility threshold based on market regime.

        In high volatility periods, threshold increases to avoid
        unnecessary circuit breaker trips during normal market stress.

        Args:
            current_volatility: Current market volatility

        Returns:
            Adjusted volatility threshold
        """
        base_threshold = self.config.max_volatility_threshold

        if not self.config.enable_adaptive_thresholds:
            return base_threshold

        # Initialize baseline if not set
        if self.config.baseline_volatility is None:
            self.config.baseline_volatility = current_volatility
            logger.info(f"Baseline volatility initialized: {current_volatility:.2%}")
            return base_threshold

        # Calculate volatility multiplier
        vol_ratio = current_volatility / self.config.baseline_volatility

        # Increase threshold in high volatility, but cap the increase
        multiplier = min(vol_ratio, self.config.max_volatility_multiplier)

        adjusted_threshold = base_threshold * multiplier

        # Don't go below base threshold
        adjusted_threshold = max(adjusted_threshold, base_threshold)

        logger.debug(
            f"Adaptive threshold: {adjusted_threshold:.2%} "
            f"(base: {base_threshold:.2%}, multiplier: {multiplier:.2f})"
        )

        return adjusted_threshold

    def update_baseline_volatility(self, returns: List[float], window_size: int = 100) -> None:
        """
        Update baseline volatility from recent returns.

        Should be called periodically (e.g., daily) to adapt to
        changing market conditions.

        Args:
            returns: List of recent returns
            window_size: Number of recent samples to use
        """
        if not returns or len(returns) < 10:
            return

        import numpy as np

        # Use recent window
        recent_returns = returns[-window_size:]
        new_baseline = float(np.std(recent_returns))

        # Exponential moving average for smooth adaptation
        if self.config.baseline_volatility is None:
            self.config.baseline_volatility = new_baseline
        else:
            alpha = 0.1  # Smoothing factor
            self.config.baseline_volatility = (
                alpha * new_baseline + (1 - alpha) * self.config.baseline_volatility
            )

        logger.info(f"Baseline volatility updated: {self.config.baseline_volatility:.2%}")

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.HALF_OPEN:
                return True  # Allow reduced trading

            # Try to transition from OPEN to HALF_OPEN if cooldown has passed
            if self.state == CircuitState.OPEN:
                if self._attempt_half_open_transition():
                    return True  # Now in HALF_OPEN state, trading allowed with reduced size

            # Check for full auto-reset
            if self._should_auto_reset():
                self._reset()
                return True

            return False

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on circuit state.

        Returns:
            1.0 for normal, reduced for half-open, 0.0 for open
        """
        if self.state == CircuitState.CLOSED:
            return 1.0
        elif self.state == CircuitState.HALF_OPEN:
            return self.config.recovery_trade_size_pct
        else:
            return 0.0

    def manual_stop(self) -> None:
        """Manually trigger circuit breaker."""
        self._trip(TripReason.MANUAL_STOP)
        logger.warning("Circuit breaker manually triggered")

    def manual_reset(self) -> None:
        """Manually reset circuit breaker."""
        with self._lock:
            self._reset()
            logger.info("Circuit breaker manually reset")

    def attempt_recovery(self, trade_successful: bool) -> None:
        """Attempt recovery from half-open state."""
        if self.state != CircuitState.HALF_OPEN:
            return

        with self._lock:
            if trade_successful:
                self.recovery_trades += 1
                if self.recovery_trades >= self.config.recovery_trades_required:
                    self._reset()
                    logger.info("Circuit breaker fully recovered")
            else:
                # Failed recovery - back to open
                self.state = CircuitState.OPEN
                self.recovery_trades = 0
                logger.warning("Recovery failed - circuit breaker re-opened")

    def register_callback(self, callback: callable) -> None:
        """Register callback for state changes."""
        self._callbacks.append(callback)

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "trip_reason": self.trip_reason.value if self.trip_reason else None,
            "trip_time": self.trip_time.isoformat() if self.trip_time else None,
            "daily_pnl_pct": self.session.daily_pnl_pct,
            "drawdown_pct": self.session.drawdown_pct,
            "consecutive_losses": self.session.consecutive_losses,
            "recent_api_errors": len(self.session.api_errors),
            "recovery_trades": self.recovery_trades,
            "can_trade": self.can_trade(),
            "position_multiplier": self.get_position_multiplier(),
        }

    def _check_conditions(self) -> None:
        """Check all circuit breaker conditions."""
        # Daily loss limit
        if self.session.daily_pnl_pct <= -self.config.daily_loss_limit_pct:
            self._trip(TripReason.DAILY_LOSS_LIMIT)
            return

        # Consecutive losses
        if self.session.consecutive_losses >= self.config.consecutive_loss_limit:
            self._trip(TripReason.CONSECUTIVE_LOSSES)
            return

        # Max drawdown
        if self.session.drawdown_pct >= self.config.max_drawdown_pct:
            self._trip(TripReason.MAX_DRAWDOWN)
            return

    def _trip(self, reason: TripReason) -> None:
        """Trip the circuit breaker."""
        if self.state == CircuitState.OPEN:
            return  # Already tripped

        self.state = CircuitState.OPEN
        self.trip_reason = reason
        self.trip_time = datetime.utcnow()
        self.recovery_trades = 0

        logger.error(
            f"🚨 CIRCUIT BREAKER TRIPPED: {reason.value}\n"
            f"   Daily PnL: {self.session.daily_pnl_pct:.2%}\n"
            f"   Drawdown: {self.session.drawdown_pct:.2%}\n"
            f"   Consecutive Losses: {self.session.consecutive_losses}"
        )

        # Update metrics if available
        if METRICS_AVAILABLE:
            try:
                exporter = get_exporter()
                exporter.set_circuit_breaker_status(1)  # 1 = on (tripped)
            except Exception as e:
                logger.warning(f"Failed to update circuit breaker metrics: {e}")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(self.get_status())
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.trip_reason = None
        self.trip_time = None
        self.recovery_trades = 0
        self.session.consecutive_losses = 0

        logger.info("Circuit breaker reset to CLOSED state")

        # Update metrics if available
        if METRICS_AVAILABLE:
            try:
                exporter = get_exporter()
                exporter.set_circuit_breaker_status(0)  # 0 = off (closed)
            except Exception as e:
                logger.warning(f"Failed to update circuit breaker metrics: {e}")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(self.get_status())
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _should_auto_reset(self) -> bool:
        """
        Check if auto-reset conditions are met.
        
        This is a pure check with no side effects. State transitions
        should be handled by the caller with proper locking.
        """
        if not self.trip_time:
            return False

        # Check cooldown first
        cooldown_elapsed = datetime.utcnow() - self.trip_time
        if cooldown_elapsed < timedelta(minutes=self.config.cooldown_minutes):
            return False

        # Check auto-reset time
        if cooldown_elapsed >= timedelta(hours=self.config.auto_reset_after_hours):
            return True

        return False

    def _attempt_half_open_transition(self) -> bool:
        """
        Attempt to transition from OPEN to HALF_OPEN state.
        
        Returns True if transition was performed, False otherwise.
        Must be called with lock held.
        """
        if self.state != CircuitState.OPEN:
            return False
            
        if not self.trip_time:
            return False
            
        # Check cooldown
        cooldown_elapsed = datetime.utcnow() - self.trip_time
        if cooldown_elapsed < timedelta(minutes=self.config.cooldown_minutes):
            return False
            
        # Transition to half-open
        self.state = CircuitState.HALF_OPEN
        logger.info("Circuit breaker transitioning to HALF_OPEN")
        
        # Update metrics for half-open state
        if METRICS_AVAILABLE:
            try:
                exporter = get_exporter()
                exporter.set_circuit_breaker_status(0)  # 0 for closed/half-open
            except Exception as e:
                logger.warning(f"Failed to update circuit breaker metrics: {e}")
                
        return True
