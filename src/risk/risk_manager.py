"""
Stoic Citadel - Integrated Risk Manager
========================================

Central risk management coordination:
- Circuit breaker integration
- Position sizing
- Portfolio risk
- Real-time risk monitoring
- Derivatives Greeks monitoring
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .correlation import CorrelationAnalyzer
from .liquidation import LiquidationConfig, LiquidationGuard
from .position_sizing import PositionSizer, PositionSizingConfig

# Try to import metrics exporter
try:
    from src.monitoring.metrics_exporter import get_exporter
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_exposure: Decimal = Decimal("0.0")
    exposure_pct: Decimal = Decimal("0.0")
    open_positions: int = 0
    daily_pnl: Decimal = Decimal("0.0")
    daily_pnl_pct: Decimal = Decimal("0.0")
    unrealized_pnl: Decimal = Decimal("0.0")
    current_drawdown_pct: Decimal = Decimal("0.0")
    var_95: Decimal = Decimal("0.0")
    sharpe_estimate: Decimal = Decimal("0.0")
    total_delta: Decimal = Decimal("0.0")
    total_gamma: Decimal = Decimal("0.0")
    circuit_state: str = "closed"
    can_trade: bool = True
    position_multiplier: Decimal = Decimal("1.0")


class RiskManager:
    """
    Central risk management system.
    """

    def __getstate__(self):
        """Custom pickling to avoid unpicklable objects like locks."""
        state = self.__dict__.copy()
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state):
        """Restore state and recreate the lock."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __init__(
        self,
        circuit_config: CircuitBreakerConfig | None = None,
        sizing_config: PositionSizingConfig | None = None,
        liquidation_config: LiquidationConfig | None = None,
        enable_notifications: bool = True,
    ):
        if circuit_config is None or sizing_config is None or liquidation_config is None:
            try:
                from src.config.manager import config
                cfg = config()
                if circuit_config is None:
                    circuit_config = CircuitBreakerConfig(
                        max_drawdown_pct=cfg.risk.max_drawdown_pct,
                        daily_loss_limit_pct=cfg.risk.max_daily_loss_pct,
                    )
                if sizing_config is None:
                    sizing_config = PositionSizingConfig(
                        max_position_pct=cfg.risk.max_position_pct,
                        max_portfolio_risk_pct=cfg.risk.max_portfolio_risk,
                    )
                if liquidation_config is None:
                    liquidation_config = LiquidationConfig(safety_buffer=cfg.risk.safety_buffer)
            except Exception:
                pass

        self.circuit_breaker = CircuitBreaker(circuit_config)
        self.position_sizer = PositionSizer(sizing_config)
        self.liquidation_guard = LiquidationGuard(liquidation_config)
        self.correlation_analyzer = CorrelationAnalyzer()

        self._account_balance: Decimal = Decimal("0.0")
        self._exchange_balances: dict[str, Decimal] = {}
        self._positions: dict[str, dict] = {}
        self._exchange_positions: dict[str, dict[str, dict]] = {}
        self._trade_history: list[dict] = []
        self._metrics: RiskMetrics = RiskMetrics()
        self._lock = threading.Lock()
        self._enable_notifications = enable_notifications
        self._notification_handlers: list[callable] = []
        self.emergency_exit = False
        self.circuit_breaker.register_callback(self._on_circuit_state_change)
        logger.info("Risk Manager initialized")

    def initialize(
        self,
        account_balance: float | Decimal,
        existing_positions: dict[str, dict] | None = None,
        exchange: str = "default"
    ) -> None:
        """
        Initialize the risk manager session with account details.
        
        Args:
            account_balance: Total available balance in base currency.
            existing_positions: Dictionary of current open positions.
            exchange: Exchange identifier for multi-exchange support.
        """
        with self._lock:
            d_balance = Decimal(str(account_balance))
            self._exchange_balances[exchange] = d_balance
            self._account_balance = sum(self._exchange_balances.values())
            self._exchange_positions[exchange] = existing_positions or {}
            
            # Rebuild flattened position map
            self._positions = {}
            for exch_pos in self._exchange_positions.values():
                self._positions.update(exch_pos)
            
            self.circuit_breaker.initialize_session(float(self._account_balance))
            self._update_metrics()
        logger.info(f"Risk Manager initialized on {exchange} with balance: {self._account_balance}")

    def evaluate_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        side: str = "long",
        **kwargs
    ) -> dict[str, Any]:
        """
        Evaluate if a new trade complies with risk rules.
        
        Args:
            symbol: Trading pair symbol (e.g. BTC/USDT).
            entry_price: Planned entry price.
            stop_loss_price: Planned stop loss price.
            side: 'long' or 'short'.
            **kwargs: Additional parameters for sizing (e.g. volatility).
        
        Returns:
            dict: {
                "allowed": bool,
                "symbol": str,
                "rejection_reason": str | None,
                "position_size": Decimal (if allowed),
                "position_value": Decimal (if allowed)
            }
        """
        res: dict[str, Any] = {"allowed": False, "symbol": symbol, "rejection_reason": None}

        # 1. Check Emergency Stop (Priority)
        if self.emergency_exit:
            res["rejection_reason"] = "Emergency Stop Active"
            logger.warning(f"Trade rejected for {symbol}: {res['rejection_reason']}")
            return res

        # 2. Check Circuit Breaker
        if not self.circuit_breaker.can_trade():
            res["rejection_reason"] = "Circuit breaker is OPEN (trading halted)"
            logger.warning(f"Trade rejected for {symbol}: {res['rejection_reason']}")
            return res

        # 3. Calculate Position Sizing
        try:
            sizing = self.position_sizer.calculate_position_size(
                account_balance=float(self._account_balance),
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                **kwargs
            )
        except Exception as e:
            res["rejection_reason"] = f"Sizing calculation failed: {str(e)}"
            logger.error(f"Sizing error for {symbol}: {e}")
            return res
        
        # 4. Apply Circuit Breaker Multiplier (Risk Dampening)
        mult = Decimal(str(self.circuit_breaker.get_position_multiplier()))
        
        position_size = Decimal(str(sizing["position_size"])) * mult
        position_value = Decimal(str(sizing["position_value"])) * mult

        # 5. Final Sanity Check
        if position_value <= 0:
             res["rejection_reason"] = "Calculated position value is zero or negative"
             return res

        res["allowed"] = True
        res["position_size"] = position_size
        res["position_value"] = position_value
        
        logger.info(f"Trade approved for {symbol}. Size: {position_size}, Value: {position_value} (Mult: {mult})")
        return res

    def record_entry(self, symbol: str, entry_price: float, position_size: float, stop_loss_price: float, exchange: str = "default", **kwargs) -> None:
        with self._lock:
            self._positions[symbol] = {
                "entry_price": entry_price, "size": position_size, "stop_loss": stop_loss_price,
                "value": entry_price * position_size, "entry_time": datetime.utcnow(), **kwargs
            }
            self._update_metrics()

    def record_exit(self, symbol: str, exit_price: float, reason: str = "") -> dict:
        with self._lock:
            if symbol not in self._positions: return {}
            pos = self._positions.pop(symbol)
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
            res = {"symbol": symbol, "pnl": pnl, "pnl_pct": pnl_pct}
            self.circuit_breaker.record_trade(res, pnl_pct)
            self._update_metrics()
            return res

    def _update_metrics(self) -> None:
        total_exposure = sum(p.get("value", 0) for p in self._positions.values())
        unrealized = sum(p.get("unrealized_pnl", 0) for p in self._positions.values())
        total_delta = sum(p.get("delta", 0.0) for p in self._positions.values())
        total_gamma = sum(p.get("gamma", 0.0) for p in self._positions.values())

        cb_status = self.circuit_breaker.get_status()
        self._metrics = RiskMetrics(
            timestamp=datetime.utcnow(),
            total_exposure=Decimal(str(total_exposure)),
            exposure_pct=Decimal(str(total_exposure / float(self._account_balance))) if self._account_balance > 0 else Decimal("0"),
            open_positions=len(self._positions),
            daily_pnl_pct=Decimal(str(cb_status.get("daily_pnl_pct", 0))),
            unrealized_pnl=Decimal(str(unrealized)),
            current_drawdown_pct=Decimal(str(cb_status.get("drawdown_pct", 0))),
            total_delta=Decimal(str(total_delta)),
            total_gamma=Decimal(str(total_gamma)),
            circuit_state=cb_status["state"],
            can_trade=cb_status["can_trade"],
            position_multiplier=Decimal(str(cb_status.get("position_multiplier", 1.0))),
        )

    def emergency_stop(self) -> None:
        self.circuit_breaker.manual_stop()
        self.emergency_exit = True

    def _on_circuit_state_change(self, status: dict) -> None:
        logger.info(f"Circuit Breaker state changed: {status['state']}")

    def get_status(self) -> dict:
        return {"metrics": self._metrics.__dict__, "circuit_breaker": self.circuit_breaker.get_status(), "positions": self._positions}
