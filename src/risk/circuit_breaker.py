"""
Stoic Citadel - Circuit Breaker System
======================================

Emergency stop mechanism for trading bot:
- Daily loss limit protection
- Consecutive losses protection
- Drawdown protection
- Volatility circuit breaker
- Market contagion protection
- Manual override capability
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# Try to import metrics exporter
try:
    from src.monitoring.metrics_exporter import get_exporter
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

from src.utils.notifications import get_notifier
from src.config import config

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
    FLASH_CRASH = "flash_crash"
    MARKET_CONTAGION = "market_contagion"
    API_ERRORS = "api_errors"
    MANUAL_STOP = "manual_stop"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Market protections
    flash_crash_threshold_pct: float = 0.20
    flash_crash_symbol: str = "BTC/USDT"
    contagion_threshold: float = 0.80
    min_assets_for_contagion: int = 3

    # Loss limits
    daily_loss_limit_pct: float = 0.05
    consecutive_loss_limit: int = 5
    max_drawdown_pct: float = 0.15

    # Volatility limits
    max_volatility_threshold: float = 0.08
    volatility_lookback_hours: int = 24

    # API error limits
    max_api_errors: int = 10
    api_error_window_minutes: int = 5

    # Recovery settings
    cooldown_minutes: int = 30
    recovery_trade_size_pct: float = 0.25
    recovery_trades_required: int = 3

    # Auto-reset
    auto_reset_after_hours: int = 24

    # Adaptive thresholds
    enable_adaptive_thresholds: bool = True
    baseline_volatility: float | None = None
    max_volatility_multiplier: float = 2.0

    # Persistence
    state_file_path: Path | None = None


@dataclass
class TradingSession:
    """Track current trading session metrics."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    initial_balance: float = 0.0
    current_balance: float = 0.0
    peak_balance: float = 0.0
    trades: list[dict] = field(default_factory=list)
    consecutive_losses: int = 0
    api_errors: list[datetime] = field(default_factory=list)

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
    """

    def __init__(self, config_obj: CircuitBreakerConfig | None = None):
        self.config = config_obj or CircuitBreakerConfig()
        if self.config.state_file_path is None:
            self.config.state_file_path = config().paths.user_data_dir / "circuit_breaker_state.json"
        self.state = CircuitState.CLOSED
        self.trip_reason: TripReason | None = None
        self.trip_time: datetime | None = None
        self.session = TradingSession()
        self.recovery_trades: int = 0
        self._lock = threading.RLock()
        self._callbacks: list[callable] = []
        self.load_state()

    def initialize_session(self, initial_balance: float) -> None:
        with self._lock:
            if (self.session.start_time.date() != datetime.utcnow().date()
                or self.session.initial_balance <= 0):
                self.session = TradingSession(
                    start_time=datetime.utcnow(),
                    initial_balance=initial_balance,
                    current_balance=initial_balance,
                    peak_balance=initial_balance,
                )
                logger.info(f"New session initialized: ${initial_balance:,.2f}")
            self.save_state()

    def update_balance(self, new_balance: float) -> None:
        with self._lock:
            self.session.current_balance = new_balance
            if new_balance > self.session.peak_balance:
                self.session.peak_balance = new_balance
            self.save_state()

    def record_trade(self, trade: dict, profit_pct: float) -> None:
        with self._lock:
            self.session.trades.append(trade)
            if profit_pct < 0:
                self.session.consecutive_losses += 1
            else:
                self.session.consecutive_losses = 0
            new_balance = self.session.current_balance * (1 + profit_pct)
            self.update_balance(new_balance)
            self._check_conditions()

    def check_volatility(self, current_volatility: float) -> None:
        threshold = self._get_adaptive_volatility_threshold(current_volatility)
        if current_volatility > threshold:
            logger.warning(f"High volatility: {current_volatility:.2%} (threshold: {threshold:.2%})")
            self._trip(TripReason.HIGH_VOLATILITY)

    def check_market_crash(self, symbol: str, price_change_pct: float) -> None:
        if (symbol == self.config.flash_crash_symbol 
            and price_change_pct <= -self.config.flash_crash_threshold_pct):
            logger.critical(f"🚨 FLASH CRASH on {symbol}: {price_change_pct:.2%} drop!")
            self._trip(TripReason.FLASH_CRASH)

    def check_contagion(self, asset_changes: dict[str, float]) -> None:
        if len(asset_changes) < self.config.min_assets_for_contagion:
            return
        dropping = [a for a, c in asset_changes.items() if c < -0.05]
        ratio = len(dropping) / len(asset_changes)
        if ratio >= 0.7:
             logger.critical(f"🚨 MARKET CONTAGION: {ratio:.0%} assets dropping!")
             self._trip(TripReason.MARKET_CONTAGION)

    def _check_conditions(self) -> None:
        if self.session.daily_pnl_pct <= -self.config.daily_loss_limit_pct:
            self._trip(TripReason.DAILY_LOSS_LIMIT)
        elif self.session.consecutive_losses >= self.config.consecutive_loss_limit:
            self._trip(TripReason.CONSECUTIVE_LOSSES)
        elif self.session.drawdown_pct >= self.config.max_drawdown_pct:
            self._trip(TripReason.MAX_DRAWDOWN)

    def _trip(self, reason: TripReason) -> None:
        if self.state == CircuitState.OPEN:
            return
        self.state = CircuitState.OPEN
        self.trip_reason = reason
        self.trip_time = datetime.utcnow()
        self.save_state()
        msg = f"CIRCUIT BREAKER TRIPPED: {reason.value}"
        logger.error(f"🚨 {msg}")
        get_notifier().send_notification(msg, level="critical")
        for cb in self._callbacks:
            try: cb(self.get_status())
            except Exception: pass

    def can_trade(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED: return True
            if self.state == CircuitState.HALF_OPEN: return True
            if self.state == CircuitState.OPEN and self._attempt_half_open_transition(): return True
            if self._should_auto_reset():
                self._reset()
                return True
            return False

    def _reset(self) -> None:
        self.state = CircuitState.CLOSED
        self.trip_reason = None
        self.trip_time = None
        self.save_state()
        msg = "Circuit breaker reset to CLOSED."
        logger.info(f"✅ {msg}")
        get_notifier().send_notification(msg, level="info")

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "trip_reason": self.trip_reason.value if self.trip_reason else None,
            "can_trade": self.can_trade(),
        }

    def _get_adaptive_volatility_threshold(self, current_volatility: float) -> float:
        base = self.config.max_volatility_threshold
        if not self.config.enable_adaptive_thresholds or self.config.baseline_volatility is None:
            return base
        multiplier = min(current_volatility / self.config.baseline_volatility, self.config.max_volatility_multiplier)
        return max(base * multiplier, base)

    def _should_auto_reset(self) -> bool:
        if not self.trip_time: return False
        elapsed = datetime.utcnow() - self.trip_time
        return elapsed >= timedelta(hours=self.config.auto_reset_after_hours)

    def _attempt_half_open_transition(self) -> bool:
        if self.state != CircuitState.OPEN or not self.trip_time: return False
        if (datetime.utcnow() - self.trip_time) < timedelta(minutes=self.config.cooldown_minutes):
            return False
        self.state = CircuitState.HALF_OPEN
        return True

    def manual_stop(self) -> None: self._trip(TripReason.MANUAL_STOP)
    def manual_reset(self) -> None: self._reset()
    def register_callback(self, cb: callable) -> None: self._callbacks.append(cb)

    def save_state(self) -> None:
        if not self.config.state_file_path: return
        try:
            data = {"state": self.state.value, "trip_reason": self.trip_reason.value if self.trip_reason else None}
            self.config.state_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.state_file_path, "w") as f: json.dump(data, f)
        except Exception: pass

    def load_state(self) -> None:
        if not self.config.state_file_path or not self.config.state_file_path.exists(): return
        try:
            with open(self.config.state_file_path) as f:
                data = json.load(f)
            self.state = CircuitState(data["state"])
            if data["trip_reason"]: self.trip_reason = TripReason(data["trip_reason"])
        except Exception: pass
