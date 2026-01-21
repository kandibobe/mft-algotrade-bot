"""
Stoic Citadel - Integrated Risk Manager (Refined)
================================================

Central risk management coordination:
- Circuit breaker integration
- Position sizing with regime awareness
- Portfolio risk & Correlation Guard
- Real-time Equity & uPnL monitoring
- Atomic safety checks for MFT execution
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .correlation import CorrelationAnalyzer
from .liquidation import LiquidationConfig, LiquidationGuard
from .news_filter import NewsFilter
from .position_sizing import PositionSizer, PositionSizingConfig

# Ensure high precision for financial calculations
getcontext().prec = 28

# Try to import metrics exporter
try:
    from src.monitoring.metrics_exporter import get_exporter
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Current risk metrics snapshot with high-precision decimals."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_exposure: Decimal = Decimal("0.0")
    exposure_pct: Decimal = Decimal("0.0")
    open_positions: int = 0
    daily_pnl: Decimal = Decimal("0.0")
    daily_pnl_pct: Decimal = Decimal("0.0")
    unrealized_pnl: Decimal = Decimal("0.0")
    equity: Decimal = Decimal("0.0")
    current_drawdown_pct: Decimal = Decimal("0.0")
    circuit_state: str = "closed"
    can_trade: bool = True
    position_multiplier: Decimal = Decimal("1.0")
    total_delta: Decimal = Decimal("0.0")
    total_gamma: Decimal = Decimal("0.0")

class RiskManager:
    """
    Refined Risk Manager for Production.
    Supports atomic updates, multi-exchange balance tracking, and real-time equity protection.
    """

    def __init__(
        self,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        sizing_config: Optional[PositionSizingConfig] = None,
        liquidation_config: Optional[LiquidationConfig] = None,
        enable_notifications: bool = True,
        db_path: Optional[str] = None,
    ):
        # 📂 Persistent state path
        if db_path is None:
            try:
                from src.config.unified_config import load_config
                u_cfg = load_config()
                self._db_path = str(u_cfg.paths.user_data_dir / "risk_state_v2.db")
            except Exception:
                self._db_path = "user_data/risk_state_v2.db"
        else:
            self._db_path = db_path

        # Components
        self.circuit_breaker = CircuitBreaker(circuit_config)
        self.position_sizer = PositionSizer(sizing_config)
        self.liquidation_guard = LiquidationGuard(liquidation_config)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.news_filter = NewsFilter()

        # State (Internal)
        self._exchange_balances: Dict[str, Decimal] = {}
        self._exchange_positions: Dict[str, Dict[str, Dict]] = {} # exchange -> symbol -> pos_data
        self._metrics: RiskMetrics = RiskMetrics()
        
        # Concurrency
        self._lock = threading.RLock()
        
        # Flags
        self._enable_notifications = enable_notifications
        self.emergency_exit = False
        
        # Register CB callback
        self.circuit_breaker.register_callback(self._on_circuit_state_change)
        
        # Load persisted state
        self.load_state()
        logger.info("Risk Manager (Refined) initialized")

    def _get_total_balance(self) -> Decimal:
        """Calculate total wallet balance across all exchanges (Atomic)."""
        return sum(self._exchange_balances.values(), Decimal("0.0"))

    def _get_total_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized PnL across all exchanges (Atomic)."""
        total_pnl = Decimal("0.0")
        for exch_pos in self._exchange_positions.values():
            for pos in exch_pos.values():
                total_pnl += Decimal(str(pos.get("unrealized_pnl", 0.0)))
        return total_pnl

    def _update_metrics(self) -> None:
        """Update metrics and trigger circuit breaker checks (Must be called under lock)."""
        balance = self._get_total_balance()
        u_pnl = self._get_total_unrealized_pnl()
        equity = balance + u_pnl
        
        # Calculate exposure
        total_exposure = Decimal("0.0")
        total_delta = Decimal("0.0")
        total_gamma = Decimal("0.0")
        open_count = 0
        
        for exch_pos in self._exchange_positions.values():
            for pos in exch_pos.values():
                total_exposure += Decimal(str(pos.get("value", 0.0)))
                total_delta += Decimal(str(pos.get("delta", 0.0)))
                total_gamma += Decimal(str(pos.get("gamma", 0.0)))
                open_count += 1

        # Update Circuit Breaker (Equity-based for Prop Firm safety)
        self.circuit_breaker.update_metrics(balance=float(balance), equity=float(equity))
        
        cb_status = self.circuit_breaker.get_status()
        
        # Refresh snapshot
        self._metrics = RiskMetrics(
            timestamp=datetime.utcnow(),
            total_exposure=total_exposure,
            exposure_pct=(total_exposure / balance) if balance > 0 else Decimal("0"),
            open_positions=open_count,
            daily_pnl_pct=Decimal(str(cb_status.get("daily_pnl_pct", 0))),
            unrealized_pnl=u_pnl,
            equity=equity,
            current_drawdown_pct=Decimal(str(cb_status.get("drawdown_pct", 0))),
            circuit_state=cb_status["state"],
            can_trade=cb_status["can_trade"],
            position_multiplier=Decimal(str(cb_status.get("position_multiplier", 1.0))),
            total_delta=total_delta,
            total_gamma=total_gamma
        )

        # Export to Prometheus if available
        if METRICS_AVAILABLE:
            try:
                exporter = get_exporter()
                exporter.update_portfolio_metrics(
                    value=float(equity),
                    positions=self.get_all_symbols(),
                    pnl_pct=float(self._metrics.daily_pnl_pct)
                )
                exporter.set_circuit_breaker_status(1 if not self._metrics.can_trade else 0)
            except Exception as e:
                logger.warning(f"Failed to export risk metrics: {e}")

    def update_balance(self, exchange: str, balance: float | Decimal) -> None:
        """Update balance for a specific exchange atomically."""
        with self._lock:
            self._exchange_balances[exchange] = Decimal(str(balance))
            self._update_metrics()
            self.save_state()
            logger.debug(f"Balance updated for {exchange}: {balance}")

    def update_position_pnl(self, exchange: str, symbol: str, price: float, unrealized_pnl: float) -> None:
        """Update real-time uPnL for a position."""
        with self._lock:
            if exchange in self._exchange_positions and symbol in self._exchange_positions[exchange]:
                pos = self._exchange_positions[exchange][symbol]
                pos["current_price"] = price
                pos["unrealized_pnl"] = unrealized_pnl
                
                # Recalculate position value based on new price
                pos["value"] = price * pos.get("size", 0.0)
                
                self._update_metrics()

    async def evaluate_safety(self, symbol: str, side: str, amount: float, price: float) -> Tuple[bool, str]:
        """
        Atomic Safety Check before order execution.
        Verifies: Circuit Breaker, News, Exposure Limits, and Correlation.
        """
        with self._lock:
            # 1. Circuit Breaker
            if not self._metrics.can_trade:
                return False, f"Circuit Breaker is {self._metrics.circuit_state.upper()}"

            # 2. Exposure Limit (Max Positions)
            try:
                from src.config.unified_config import load_config
                max_pos = load_config().strategy.max_positions
            except Exception:
                max_pos = 5
                
            if self._metrics.open_positions >= max_pos:
                 # Check if we are reducing a position
                 current_pos = self._get_position(symbol)
                 if not current_pos or (current_pos['side'] == side):
                     return False, f"Max positions reached: {self._metrics.open_positions}"

            # 3. Liquidation Risk
            if self.liquidation_guard.is_near_liquidation(symbol, price):
                return False, "Liquidation risk too high for this symbol"

        # 4. News Filter (Outside lock as it might involve async I/O)
        allowed, news_reason = await self.news_filter.check_news_impact(symbol)
        if not allowed:
            return False, f"News Impact: {news_reason}"

        return True, ""

    def record_entry(self, exchange: str, symbol: str, size: float, price: float, side: str, **kwargs) -> None:
        """Atomic record of a new trade entry."""
        with self._lock:
            if exchange not in self._exchange_positions:
                self._exchange_positions[exchange] = {}
            
            self._exchange_positions[exchange][symbol] = {
                "exchange": exchange,
                "symbol": symbol,
                "size": size,
                "entry_price": price,
                "current_price": price,
                "side": side,
                "value": size * price,
                "unrealized_pnl": 0.0,
                "entry_time": datetime.utcnow().isoformat(),
                **kwargs
            }
            self._update_metrics()
            self.save_state()
            logger.info(f"Entry recorded: {exchange} {symbol} {side} @ {price}")

    def record_exit(self, exchange: str, symbol: str, size: float, price: float) -> Dict:
        """Atomic record of a trade exit (Partial or Full)."""
        with self._lock:
            if exchange not in self._exchange_positions or symbol not in self._exchange_positions[exchange]:
                logger.warning(f"Exit recorded for unknown position: {exchange} {symbol}")
                return {}
            
            pos = self._exchange_positions[exchange][symbol]
            old_size = pos["size"]
            
            # PnL Calculation
            pnl = (price - pos["entry_price"]) * size if pos["side"] == "long" else (pos["entry_price"] - price) * size
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] if pos["side"] == "long" else (pos["entry_price"] - price) / pos["entry_price"]

            if size >= old_size:
                # Full Exit
                self._exchange_positions[exchange].pop(symbol)
                logger.info(f"Full Exit recorded: {symbol} @ {price}. PnL: {pnl:.4f}")
            else:
                # Partial Exit
                pos["size"] -= size
                pos["value"] = pos["size"] * price
                logger.info(f"Partial Exit recorded: {symbol} ({size}). Remaining: {pos['size']}")

            # Record trade in circuit breaker for stats
            self.circuit_breaker.record_trade({"symbol": symbol, "pnl": float(pnl)}, float(pnl_pct))
            
            self._update_metrics()
            self.save_state()
            return {"pnl": pnl, "pnl_pct": pnl_pct, "fully_closed": size >= old_size}

    def get_all_symbols(self) -> List[str]:
        """Get unique list of all open symbols."""
        symbols = set()
        for exch in self._exchange_positions.values():
            symbols.update(exch.keys())
        return list(symbols)

    def _get_position(self, symbol: str) -> Optional[Dict]:
        """Find position for a symbol across all exchanges."""
        for exch in self._exchange_positions.values():
            if symbol in exch:
                return exch[symbol]
        return None

    def _on_circuit_state_change(self, status: dict) -> None:
        logger.info(f"Circuit Breaker state changed: {status['state']}")
        if status["state"] == "open":
            self.emergency_exit = True
            logger.critical("🚨 TRADING HALTED BY RISK MANAGER")

    def save_state(self) -> None:
        """Persist state using high-performance serialization."""
        try:
            import pickle
            import sqlite3
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS risk_v2 (id INTEGER PRIMARY KEY, data BLOB)")
                state = {
                    "balances": {k: str(v) for k, v in self._exchange_balances.items()},
                    "positions": self._exchange_positions,
                    "metrics": self._metrics
                }
                conn.execute("INSERT OR REPLACE INTO risk_v2 (id, data) VALUES (1, ?)", (pickle.dumps(state),))
        except Exception as e:
            logger.error(f"Failed to save RiskManager state: {e}")

    def load_state(self) -> None:
        """Load persisted state."""
        if not os.path.exists(self._db_path):
            return
        try:
            import pickle
            import sqlite3
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute("SELECT data FROM risk_v2 WHERE id = 1").fetchone()
                if row:
                    state = pickle.loads(row[0])
                    self._exchange_balances = {k: Decimal(v) for k, v in state.get("balances", {}).items()}
                    self._exchange_positions = state.get("positions", {})
                    # Metrics will be recalculated on update_metrics
                    with self._lock:
                        self._update_metrics()
            logger.info("Risk Manager state restored successfully")
        except Exception as e:
            logger.error(f"Failed to load RiskManager state: {e}")

    def get_status(self) -> Dict:
        """Thread-safe status snapshot."""
        with self._lock:
            return {
                "metrics": self._metrics.__dict__,
                "open_positions": self._exchange_positions,
                "balances": {k: float(v) for k, v in self._exchange_balances.items()},
                "circuit_breaker": self.circuit_breaker.get_status()
            }