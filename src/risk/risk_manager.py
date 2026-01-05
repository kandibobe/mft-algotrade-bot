"""
Stoic Citadel - Integrated Risk Manager
========================================

Central risk management coordination:
- Circuit breaker integration
- Position sizing
- Portfolio risk
- Real-time risk monitoring
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

    # Portfolio metrics
    total_exposure: Decimal = Decimal("0.0")
    exposure_pct: Decimal = Decimal("0.0")
    open_positions: int = 0

    # PnL metrics
    daily_pnl: Decimal = Decimal("0.0")
    daily_pnl_pct: Decimal = Decimal("0.0")
    unrealized_pnl: Decimal = Decimal("0.0")

    # Risk metrics
    current_drawdown_pct: Decimal = Decimal("0.0")
    var_95: Decimal = Decimal("0.0")
    sharpe_estimate: Decimal = Decimal("0.0")

    # Circuit breaker
    circuit_state: str = "closed"
    can_trade: bool = True
    position_multiplier: Decimal = Decimal("1.0")


class RiskManager:
    """
    Central risk management system.

    Coordinates all risk components and provides unified interface
    for trading decisions.
    """

    def __init__(
        self,
        circuit_config: CircuitBreakerConfig | None = None,
        sizing_config: PositionSizingConfig | None = None,
        liquidation_config: LiquidationConfig | None = None,
        enable_notifications: bool = True,
    ):
        # Auto-load from Unified Config if not provided
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
                        max_risk_pct=cfg.risk.max_portfolio_risk,
                    )

                if liquidation_config is None:
                    liquidation_config = LiquidationConfig(buffer_pct=cfg.risk.liquidation_buffer)
            except Exception as e:
                logger.warning(
                    f"Could not load unified config for RiskManager: {e}. Using defaults."
                )

        # Initialize components
        self.circuit_breaker = CircuitBreaker(circuit_config)
        self.position_sizer = PositionSizer(sizing_config)
        self.liquidation_guard = LiquidationGuard(liquidation_config)
        self.correlation_analyzer = CorrelationAnalyzer()

        # State
        self._account_balance: Decimal = Decimal("0.0")
        self._exchange_balances: dict[str, Decimal] = {}  # exchange -> balance
        self._positions: dict[str, dict] = {}  # symbol -> position_details
        self._exchange_positions: dict[str, dict[str, dict]] = {}  # exchange -> symbol -> position
        self._trade_history: list[dict] = []
        self._metrics: RiskMetrics = RiskMetrics()
        self._lock = threading.Lock()

        # Notifications
        self._enable_notifications = enable_notifications
        self._notification_handlers: list[callable] = []

        # Safety Valve
        self.emergency_exit = False

        # Register circuit breaker callback
        self.circuit_breaker.register_callback(self._on_circuit_state_change)

        logger.info("Risk Manager initialized")

    def initialize(
        self,
        account_balance: float | Decimal,
        existing_positions: dict[str, dict] | None = None,
        exchange: str = "default",
    ) -> None:
        """
        Initialize risk manager with account state for a specific exchange.
        """
        with self._lock:
            d_balance = Decimal(str(account_balance))
            self._exchange_balances[exchange] = d_balance
            
            # Update total balance
            self._account_balance = sum(self._exchange_balances.values())
            
            # Update positions
            self._exchange_positions[exchange] = existing_positions or {}
            
            # Flatten positions for backward compatibility with components that don't know about exchanges
            self._positions = {}
            for exch_pos in self._exchange_positions.values():
                self._positions.update(exch_pos)

            # Initialize circuit breaker
            self.circuit_breaker.initialize_session(float(self._account_balance))

            # Update position sizer
            position_values = {s: p.get("value", 0) for s, p in self._positions.items()}
            self.position_sizer.update_positions(position_values)
            self.position_sizer.last_account_balance = float(self._account_balance)

            # Update metrics
            self._update_metrics()

        logger.info(
            f"Risk Manager initialized: Balance=${account_balance:,.2f} on {exchange}, "
            f"Total Balance=${float(self._account_balance):,.2f}"
        )

    def evaluate_trade(
        self,
        symbol: str,
        entry_price: float | Decimal,
        stop_loss_price: float | Decimal,
        side: str = "long",
        sizing_method: str = "optimal",
        exchange: str = "default",
        force_allowed: bool = False,
        adl_score: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate whether a trade should be taken.

        Returns comprehensive trade evaluation including:
        - Whether trade is allowed
        - Recommended position size
        - Risk metrics
        - Warnings
        """
        # Convert inputs to Decimal
        d_entry_price = Decimal(str(entry_price))
        d_stop_loss_price = Decimal(str(stop_loss_price))

        result = {
            "allowed": False,
            "symbol": symbol,
            "side": side,
            "entry_price": d_entry_price,
            "stop_loss_price": d_stop_loss_price,
            "position_size": Decimal("0.0"),
            "position_value": Decimal("0.0"),
            "risk_amount": Decimal("0.0"),
            "warnings": [],
            "rejection_reason": None,
        }

        # Update position values and aggregated balance to ensure sizing is accurate
        with self._lock:
            position_values = {s: p.get("value", 0) for s, p in self._positions.items()}
            self.position_sizer.update_positions(position_values)
            self.position_sizer.last_account_balance = float(self._account_balance)

        # Check circuit breaker
        if not self.circuit_breaker.can_trade():
            result["rejection_reason"] = f"Circuit breaker: {self.circuit_breaker.state.value}"
            return result

        # Check if already in position
        if symbol in self._positions:
            result["rejection_reason"] = f"Already in position for {symbol}"
            return result

        # Check liquidation risk
        leverage = kwargs.get("leverage", 1.0)
        is_safe, liq_reason = self.liquidation_guard.check_trade_safety(
            float(d_entry_price), float(d_stop_loss_price), leverage, side
        )

        if not is_safe:
            result["rejection_reason"] = f"Liquidation Risk: {liq_reason}"
            return result

        # Check ADL Risk
        is_adl_safe, adl_reason = self.liquidation_guard.check_adl_risk(adl_score)
        if not is_adl_safe:
            result["rejection_reason"] = adl_reason
            return result

        # Check correlation risk (using cached correlation matrix if available)
        # Note: CorrelationAnalyzer needs external update via update_market_data or separate job
        # Here we assume PositionSizer handles exposure checks if matrix is set

        # Calculate position size
        try:
            # Pass symbol for HRP
            kwargs["symbol"] = symbol
            # Note: PositionSizer still expects floats, converting for compatibility
            sizing_result = self.position_sizer.calculate_position_size(
                account_balance=float(self._account_balance),
                entry_price=float(d_entry_price),
                stop_loss_price=float(d_stop_loss_price),
                method=sizing_method,
                **kwargs,
            )
        except Exception as e:
            result["rejection_reason"] = f"Position sizing error: {e}"
            return result

        # Apply circuit breaker multiplier
        multiplier = self.circuit_breaker.get_position_multiplier()
        # Convert sizing results to Decimal
        pos_size = Decimal(str(sizing_result["position_size"]))
        pos_value = Decimal(str(sizing_result["position_value"]))

        # Apply multiplier (Decimal arithmetic)
        d_multiplier = Decimal(str(multiplier))
        pos_size *= d_multiplier
        pos_value *= d_multiplier

        # Update sizing result for return
        sizing_result["position_size"] = float(pos_size)
        sizing_result["position_value"] = float(pos_value)

        if multiplier < 1.0:
            result["warnings"].append(
                f"Position reduced to {multiplier:.0%} due to circuit breaker recovery"
            )

        # Check portfolio risk
        # Note: Using float for compatibility with existing position_sizer methods
        allowed, reason = self.position_sizer.check_portfolio_risk(
            sizing_result, symbol, float(self._account_balance)
        )

        if not allowed and not force_allowed:
            result["rejection_reason"] = reason
            return result

        # Trade approved
        result["allowed"] = True
        result["position_size"] = pos_size
        result["position_value"] = pos_value
        result["risk_amount"] = Decimal(str(sizing_result.get("risk_amount", 0)))
        result["sizing_method"] = sizing_result.get("method")
        result["sizing_details"] = sizing_result

        return result

    def record_entry(
        self,
        symbol: str,
        entry_price: float | Decimal,
        position_size: float | Decimal,
        stop_loss_price: float | Decimal,
        exchange: str = "default",
        **kwargs,
    ) -> None:
        """Record position entry on a specific exchange."""
        d_entry_price = Decimal(str(entry_price))
        d_position_size = Decimal(str(position_size))
        d_stop_loss_price = Decimal(str(stop_loss_price))

        with self._lock:
            pos_details = {
                "entry_price": float(d_entry_price),
                "size": float(d_position_size),
                "stop_loss": float(d_stop_loss_price),
                "value": float(d_entry_price * d_position_size),
                "entry_time": datetime.utcnow(),
                "exchange": exchange,
                **kwargs,
            }
            
            self._positions[symbol] = pos_details
            if exchange not in self._exchange_positions:
                self._exchange_positions[exchange] = {}
            self._exchange_positions[exchange][symbol] = pos_details

            # Update position sizer
            position_values = {s: p.get("value", 0) for s, p in self._positions.items()}
            self.position_sizer.update_positions(position_values)
            self.position_sizer.last_account_balance = float(self._account_balance)

            self._update_metrics()

        logger.info(f"Entry recorded: {symbol} @ {entry_price} x {position_size} on {exchange}")

    def record_exit(self, symbol: str, exit_price: float | Decimal, reason: str = "") -> dict:
        """Record position exit and return trade result."""
        d_exit_price = Decimal(str(exit_price))

        with self._lock:
            if symbol not in self._positions:
                logger.warning(f"No position found for {symbol}")
                return {}

            position = self._positions.pop(symbol)
            exchange = position.get("exchange", "default")
            if exchange in self._exchange_positions and symbol in self._exchange_positions[exchange]:
                self._exchange_positions[exchange].pop(symbol)

            # Calculate P&L using Decimal
            d_entry_price = Decimal(str(position["entry_price"]))
            d_size = Decimal(str(position["size"]))

            d_pnl = (d_exit_price - d_entry_price) * d_size
            d_pnl_pct = (d_exit_price - d_entry_price) / d_entry_price if d_entry_price != 0 else 0

            # Convert for downstream consumption
            pnl_pct = float(d_pnl_pct)
            pnl = float(d_pnl)

            trade_result = {
                "symbol": symbol,
                "entry_price": float(d_entry_price),
                "exit_price": float(d_exit_price),
                "size": float(d_size),
                "pnl": float(d_pnl),
                "pnl_pct": float(d_pnl_pct),
                "reason": reason,
                "entry_time": position.get("entry_time"),
                "exit_time": datetime.utcnow(),
                "exchange": exchange
            }

            # Update trade history
            self._trade_history.append(trade_result)

            # Update circuit breaker
            self.circuit_breaker.record_trade(trade_result, pnl_pct)

            # Update balance
            if exchange in self._exchange_balances:
                self._exchange_balances[exchange] += d_pnl
            
            self._account_balance = sum(self._exchange_balances.values())
            self.circuit_breaker.update_balance(float(self._account_balance))

            # Update position sizer
            position_values = {s: p.get("value", 0) for s, p in self._positions.items()}
            self.position_sizer.update_positions(position_values)
            self.position_sizer.last_account_balance = float(self._account_balance)

            self._update_metrics()

            logger.info(
                f"Exit recorded: {symbol} @ {exit_price} on {exchange} | "
                f"PnL: ${pnl:,.2f} ({pnl_pct:+.2%}) | Reason: {reason}"
            )

            return trade_result

    def update_market_data(
        self, symbol: str, current_price: float, volatility: float | None = None
    ) -> None:
        """Update with current market data."""
        # Update unrealized P&L
        if symbol in self._positions:
            position = self._positions[symbol]
            unrealized = (current_price - position["entry_price"]) * position["size"]
            position["unrealized_pnl"] = unrealized
            position["current_price"] = current_price

        # Check volatility circuit breaker
        if volatility is not None:
            self.circuit_breaker.check_volatility(volatility)

        self._update_metrics()

    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return self._metrics

    def get_status(self) -> dict:
        """Get comprehensive risk status."""
        return {
            "metrics": self._metrics.__dict__,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "positions": self._positions.copy(),
            "account_balance": self._account_balance,
            "trade_count_today": len(
                [
                    t
                    for t in self._trade_history
                    if t.get("exit_time", datetime.min).date() == datetime.utcnow().date()
                ]
            ),
        }

    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self.circuit_breaker.manual_stop()
        self.emergency_exit = True
        self._notify("EMERGENCY STOP TRIGGERED - FORCING EXIT", "critical")

    def reset_emergency(self) -> None:
        """Reset emergency status."""
        self.emergency_exit = False
        logger.info("Emergency exit status reset")

    def register_notification_handler(self, handler: callable) -> None:
        """Register notification handler."""
        self._notification_handlers.append(handler)

    def _update_metrics(self) -> None:
        """Update risk metrics."""
        total_exposure = sum(p.get("value", 0) for p in self._positions.values())
        unrealized = sum(p.get("unrealized_pnl", 0) for p in self._positions.values())

        # Convert to Decimal
        d_total_exposure = Decimal(str(total_exposure))
        d_unrealized = Decimal(str(unrealized))

        cb_status = self.circuit_breaker.get_status()

        # Calculate exposure pct safely
        exposure_pct = Decimal("0.0")
        if self._account_balance > 0:
            exposure_pct = d_total_exposure / self._account_balance

        # Calculate daily pnl safely (handling Mocks in tests)
        try:
            curr_bal = self.circuit_breaker.session.current_balance
            init_bal = self.circuit_breaker.session.initial_balance

            # If these are Mocks, the subtraction might fail
            if hasattr(curr_bal, "__sub__") and not isinstance(curr_bal, Any):
                daily_pnl = Decimal(str(curr_bal - init_bal))
            else:
                # Fallback for Mocks
                daily_pnl = Decimal("0.0")
        except Exception:
            daily_pnl = Decimal("0.0")

        self._metrics = RiskMetrics(
            timestamp=datetime.utcnow(),
            total_exposure=d_total_exposure,
            exposure_pct=exposure_pct,
            open_positions=len(self._positions),
            daily_pnl=daily_pnl,
            daily_pnl_pct=Decimal(str(cb_status["daily_pnl_pct"])),
            unrealized_pnl=d_unrealized,
            current_drawdown_pct=Decimal(str(cb_status["drawdown_pct"])),
            var_95=Decimal("0.0"),  # Placeholder as not calculated here
            sharpe_estimate=Decimal("0.0"),  # Placeholder
            circuit_state=cb_status["state"],
            can_trade=cb_status["can_trade"],
            position_multiplier=Decimal(str(cb_status["position_multiplier"])),
        )

        # Export to Prometheus
        if METRICS_AVAILABLE:
            try:
                exporter = get_exporter()
                # Update portfolio metrics
                exporter.update_portfolio_metrics(
                    value=float(self.circuit_breaker.session.current_balance),
                    positions=list(self._positions.values()),
                    pnl_pct=float(cb_status["daily_pnl_pct"]),
                )

                # Sync circuit breaker status
                status_int = 1 if cb_status["state"] != "closed" else 0
                exporter.set_circuit_breaker_status(status_int)
            except Exception as e:
                # Log only once or debug to avoid spam
                logger.debug(f"Failed to export risk metrics: {e}")

    def _on_circuit_state_change(self, status: dict) -> None:
        """Handle circuit breaker state change."""
        state = status.get("state", "unknown")
        reason = status.get("trip_reason", "unknown")

        if state == "open":
            self._notify(f"🚨 Circuit Breaker TRIPPED: {reason}", "critical")
        elif state == "half_open":
            self._notify("⚠️ Circuit Breaker: Testing recovery", "warning")
        elif state == "closed":
            self._notify("✅ Circuit Breaker: Normal operation resumed", "info")

    def _notify(self, message: str, level: str = "info") -> None:
        """Send notification."""
        if not self._enable_notifications:
            return

        for handler in self._notification_handlers:
            try:
                handler(message, level)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")

    @staticmethod
    def calculate_safe_size(
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        max_risk_pct: float = 0.01,
        max_position_pct: float = 0.05,
        leverage: float = 1.0,
    ) -> float:
        """
        Pure, stateless position sizing calculation for backtesting.

        Args:
            account_balance: Total account value
            entry_price: Entry price
            stop_loss_price: Stop loss price
            max_risk_pct: Max risk per trade (default 1%)
            max_position_pct: Max position size (default 5%)
            leverage: Trade leverage

        Returns:
            Position size in stake currency
        """
        if account_balance <= 0 or entry_price <= 0:
            return 0.0

        # 1. Calculate risk per trade
        risk_amount = account_balance * max_risk_pct

        # 2. Calculate stop distance
        stop_dist_pct = abs(entry_price - stop_loss_price) / entry_price

        if stop_dist_pct < 0.001:  # Avoid division by zero or tiny stops
            stop_dist_pct = 0.001

        # 3. Calculate size based on risk
        # Risk = Size * StopDist
        # Size = Risk / StopDist
        size_by_risk = risk_amount / stop_dist_pct

        # 4. Cap by max position size
        size_by_cap = account_balance * max_position_pct * leverage

        # 5. Final size
        final_size = min(size_by_risk, size_by_cap)

        return final_size
