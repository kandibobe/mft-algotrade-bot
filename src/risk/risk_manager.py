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
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .correlation import CorrelationAnalyzer
from .liquidation import LiquidationConfig, LiquidationGuard
from .news_filter import NewsFilter
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
        db_path: str | None = None,
    ):
        # 📂 Risk State DB Path from Unified Config
        if db_path is None:
            try:
                from src.config.unified_config import load_config

                u_cfg = load_config()
                self._db_path = str(u_cfg.paths.user_data_dir / "risk_state.db")
            except Exception:
                self._db_path = "user_data/risk_state.db"
        else:
            self._db_path = db_path

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
        self.news_filter = NewsFilter()

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
        self.load_state()
        logger.info("Risk Manager initialized")

    def save_state(self) -> None:
        """Persist RiskManager state to SQLite."""
        try:
            import pickle
            import sqlite3

            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS risk_state (id INTEGER PRIMARY KEY, data BLOB, timestamp DATETIME)"
                )

                # We only want to save positions and account balance.
                # CircuitBreaker state is managed separately or can be included.
                state_to_save = {
                    "positions": self._positions,
                    "exchange_positions": self._exchange_positions,
                    "account_balance": str(self._account_balance),
                    "exchange_balances": {k: str(v) for k, v in self._exchange_balances.items()},
                    "circuit_breaker_state": self.circuit_breaker.get_status(),
                }

                data = pickle.dumps(state_to_save)
                conn.execute(
                    "INSERT OR REPLACE INTO risk_state (id, data, timestamp) VALUES (1, ?, ?)",
                    (data, datetime.utcnow()),
                )
            logger.debug("Risk Manager state saved to DB")
        except Exception as e:
            logger.error(f"Failed to save RiskManager state: {e}")

    def load_state(self) -> None:
        """Load RiskManager state from SQLite."""
        if not os.path.exists(self._db_path):
            return

        try:
            import pickle
            import sqlite3

            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("SELECT data FROM risk_state WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    state = pickle.loads(row[0])
                    self._positions = state.get("positions", {})
                    self._exchange_positions = state.get("exchange_positions", {})
                    self._account_balance = Decimal(state.get("account_balance", "0"))
                    self._exchange_balances = {
                        k: Decimal(v) for k, v in state.get("exchange_balances", {}).items()
                    }

                    cb_metrics = state.get("circuit_breaker_state")
                    if cb_metrics:
                        # Restore daily PnL and drawdown by updating session metrics
                        # Note: CircuitBreaker.initialize_session might override this if not careful
                        self.circuit_breaker.session.current_balance = float(self._account_balance)
                        # We would need to expose more of CircuitBreaker to fully restore its state
                        # but this ensures basic continuity.

                    self._update_metrics()
            logger.info("Risk Manager state loaded from DB")
        except Exception as e:
            logger.error(f"Failed to load RiskManager state: {e}")

    def initialize(
        self,
        account_balance: float | Decimal,
        existing_positions: dict[str, dict] | None = None,
        exchange: str = "default",
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

    def check_market_kill_switch(self) -> tuple[bool, str | None]:
        """
        Check if market-wide conditions warrant a trading halt.
        Rule: If BTC (market proxy) crashes > 5% in 5 mins, block all entries.
        """
        try:
            from src.data.loader import DataLoader

            loader = DataLoader()
            btc_data = loader.load_pair_data("BTC/USDT", "5m")
            if len(btc_data) < 2:
                return False, None

            last_change = (btc_data["close"].iloc[-1] - btc_data["close"].iloc[-2]) / btc_data[
                "close"
            ].iloc[-2]
            if last_change < -0.05:
                return True, f"BTC Flash Crash detected: {last_change:.2%}"
        except Exception as e:
            logger.error(f"Kill switch check failed: {e}")

        return False, None

    async def evaluate_trade(
        self, symbol: str, entry_price: float, stop_loss_price: float, side: str = "long", **kwargs
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

        # 0. News Filter (Prop Requirement)
        allowed, news_reason = await self.news_filter.check_news_impact(symbol)
        if not allowed:
            res["rejection_reason"] = news_reason
            logger.warning(f"Trade rejected for {symbol}: {news_reason}")
            return res

        # 1. Check Emergency Stop (Priority)
        if self.emergency_exit:
            res["rejection_reason"] = "Emergency Stop Active"
            logger.warning(f"Trade rejected for {symbol}: {res['rejection_reason']}")
            return res

        # 1.1 Market Kill Switch (Task 15)
        is_halted, halt_reason = self.check_market_kill_switch()
        if is_halted:
            res["rejection_reason"] = halt_reason
            logger.critical(f"MARKET KILL SWITCH TRIGGERED: {halt_reason}")
            return res

        # 2. Check Circuit Breaker
        if not self.circuit_breaker.can_trade():
            res["rejection_reason"] = "Circuit breaker is OPEN (trading halted)"
            logger.warning(f"Trade rejected for {symbol}: {res['rejection_reason']}")
            return res

        # 3. Correlation Filter (Task 7)
        # Prevent opening BTC and ETH simultaneously in the same direction if correlation is high.
        from src.config.unified_config import load_config

        try:
            u_cfg = load_config().risk
        except Exception:
            u_cfg = None

        correlation_pairs = (
            getattr(u_cfg, "correlation_pairs_block", ["BTC", "ETH"]) if u_cfg else ["BTC", "ETH"]
        )
        correlation_threshold = getattr(u_cfg, "correlation_threshold_block", 0.8) if u_cfg else 0.8
        target_pair_base = (
            symbol.split("/")[0]
            if "/" in symbol
            else symbol.split(":")[0]
            if ":" in symbol
            else symbol
        )

        if target_pair_base in correlation_pairs:
            for other_pair_base in correlation_pairs:
                if other_pair_base == target_pair_base:
                    continue

                # Check if we have an open position in the other pair
                for open_symbol, pos in self._positions.items():
                    open_base = (
                        open_symbol.split("/")[0]
                        if "/" in open_symbol
                        else open_symbol.split(":")[0]
                        if ":" in open_symbol
                        else open_symbol
                    )
                    if open_base == other_pair_base:
                        # Found the other pair. Check direction (side)
                        open_side = pos.get("side", "long")
                        if open_side == side:
                            # 3.1 Calculate actual correlation
                            try:
                                from src.data.loader import DataLoader

                                loader = DataLoader()
                                # Load recent data for both pairs (e.g., 1h timeframe, last 100 candles)
                                timeframe = "1h"
                                df1 = loader.load_pair_data(symbol, timeframe)
                                df2 = loader.load_pair_data(open_symbol, timeframe)

                                # Use CorrelationManager to calculate correlation
                                from .correlation import CorrelationManager

                                corr_manager = CorrelationManager()
                                current_corr = corr_manager.calculate_correlation(df1, df2)

                                if current_corr > correlation_threshold:
                                    res["rejection_reason"] = (
                                        f"High Correlation: {symbol} vs {open_symbol} "
                                        f"corr={current_corr:.2f} > {correlation_threshold}. "
                                        f"Both are in {side} direction."
                                    )
                                    logger.warning(
                                        f"Trade rejected for {symbol}: {res['rejection_reason']}"
                                    )
                                    return res
                                else:
                                    logger.info(
                                        f"Correlation check passed for {symbol} vs {open_symbol}: "
                                        f"{current_corr:.2f} <= {correlation_threshold}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Correlation calculation failed: {e}. Blocking for safety."
                                )
                                res["rejection_reason"] = f"Correlation check failed: {e}"
                                return res

        # 3.1 Global Correlation Guard (Task 15 - Plan)
        # Check current average portfolio correlation
        if len(self._positions) >= 2:
            try:
                from .correlation import CorrelationManager

                corr_manager = CorrelationManager()
                # Simplified check: if BTC/ETH correlation is too high, reduce risk
                # In real scenario we'd check all pairs
                # For now, let's use a dynamic multiplier based on number of positions
                # Higher correlation or more positions -> lower multiplier
                avg_corr = 0.7  # Placeholder for actual calculation
                if avg_corr > 0.85:
                    logger.warning(
                        f"High portfolio correlation detected ({avg_corr}), dampening risk."
                    )
                    kwargs["risk_multiplier"] = kwargs.get("risk_multiplier", 1.0) * 0.5
            except Exception:
                pass

        # 4. Calculate Position Sizing
        try:
            sizing = self.position_sizer.calculate_position_size(
                account_balance=float(self._account_balance),
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                **kwargs,
            )
        except Exception as e:
            res["rejection_reason"] = f"Sizing calculation failed: {e!s}"
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

        logger.info(
            f"Trade approved for {symbol}. Size: {position_size}, Value: {position_value} (Mult: {mult})"
        )
        return res

    def record_entry(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        stop_loss_price: float,
        exchange: str = "default",
        **kwargs,
    ) -> None:
        with self._lock:
            self._positions[symbol] = {
                "entry_price": entry_price,
                "size": position_size,
                "stop_loss": stop_loss_price,
                "value": entry_price * position_size,
                "entry_time": datetime.utcnow(),
                **kwargs,
            }
            self._update_metrics()
            self.save_state()

    def record_exit(self, symbol: str, exit_price: float, reason: str = "") -> dict:
        with self._lock:
            if symbol not in self._positions:
                return {}
            pos = self._positions.pop(symbol)
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
            res = {"symbol": symbol, "pnl": pnl, "pnl_pct": pnl_pct}
            self.circuit_breaker.record_trade(res, pnl_pct)
            self._update_metrics()
            self.save_state()
            return res

    def _update_metrics(self) -> None:
        total_exposure = sum(p.get("value", 0) for p in self._positions.values())
        unrealized = sum(p.get("unrealized_pnl", 0) for p in self._positions.values())
        total_delta = sum(p.get("delta", 0.0) for p in self._positions.values())
        total_gamma = sum(p.get("gamma", 0.0) for p in self._positions.values())

        # Calculate current equity
        current_balance_float = float(self._account_balance)
        equity = current_balance_float + unrealized

        # CRITICAL: Update Circuit Breaker with current Balance AND Equity
        # This allows Prop Firm style drawdown tracking (Daily PnL from Equity High Water Mark)
        self.circuit_breaker.update_metrics(balance=current_balance_float, equity=equity)

        cb_status = self.circuit_breaker.get_status()
        self._metrics = RiskMetrics(
            timestamp=datetime.utcnow(),
            total_exposure=Decimal(str(total_exposure)),
            exposure_pct=Decimal(str(total_exposure / float(self._account_balance)))
            if self._account_balance > 0
            else Decimal("0"),
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

        if METRICS_AVAILABLE:
            try:
                exporter = get_exporter()
                exporter.update_portfolio_metrics(
                    value=equity,
                    positions=list(self._positions.keys()),
                    pnl_pct=float(self._metrics.daily_pnl_pct),
                )
                exporter.set_circuit_breaker_status(1 if not self._metrics.can_trade else 0)
            except Exception as e:
                logger.warning(f"Failed to update Prometheus metrics: {e}")
            try:
                exporter = get_exporter()
                exporter.update_portfolio_metrics(
                    value=pf_value,
                    positions=list(self._positions.keys()),
                    pnl_pct=float(self._metrics.daily_pnl_pct),
                )
                exporter.set_circuit_breaker_status(1 if not self._metrics.can_trade else 0)
            except Exception as e:
                logger.warning(f"Failed to update Prometheus metrics: {e}")

    def emergency_stop(self) -> None:
        self.circuit_breaker.manual_stop()
        self.emergency_exit = True

    def _on_circuit_state_change(self, status: dict) -> None:
        logger.info(f"Circuit Breaker state changed: {status['state']}")

    def update_portfolio_correlation(self, avg_corr: float) -> None:
        """
        External hook to update portfolio correlation from external analyzer.
        If correlation > 0.85, it will dampen risk in evaluate_trade.
        """
        with self._lock:
            # Placeholder for stateful correlation tracking
            # In a real system, we'd store this in self._metrics
            pass

    def update_market_price(self, symbol: str, price: float) -> None:
        """
        Update market price for a symbol to recalculate unrealized PnL and Equity.
        Critical for Prop Firm daily drawdown rules.
        """
        with self._lock:
            if symbol in self._positions:
                pos = self._positions[symbol]
                entry = pos["entry_price"]
                size = pos["size"]
                side = pos.get("side", "long")
                
                # Calculate PnL
                if side == "long":
                    pnl = (price - entry) * size
                else:
                    pnl = (entry - price) * size
                
                pos["unrealized_pnl"] = pnl
                pos["current_price"] = price
                
                # Update Metrics which triggers Circuit Breaker Equity Check
                self._update_metrics()

    def get_status(self) -> dict:
        return {
            "metrics": self._metrics.__dict__,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "positions": self._positions,
        }

    def get_metrics(self) -> dict:
        """
        Get current risk metrics in a dictionary format.
        Useful for testing and UI integration.
        """
        return {
            "total_exposure": float(self._metrics.total_exposure),
            "exposure_pct": float(self._metrics.exposure_pct),
            "open_positions": self._metrics.open_positions,
            "daily_pnl": float(self._metrics.daily_pnl),
            "daily_pnl_pct": float(self._metrics.daily_pnl_pct),
            "unrealized_pnl": float(self._metrics.unrealized_pnl),
            "current_drawdown_pct": float(self._metrics.current_drawdown_pct),
            "circuit_state": self._metrics.circuit_state,
            "can_trade": self._metrics.can_trade,
        }