"""
Stoic Citadel - Trading Metrics
================================

Prometheus metrics for trading monitoring:
- PnL tracking
- Drawdown monitoring
- Trade statistics
- Risk metrics
- System health
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime
import time

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TradingMetrics:
    """
    Prometheus metrics exporter for trading bot.
    
    Exposes key metrics for monitoring dashboards.
    """
    
    def __init__(
        self,
        namespace: str = "stoic_citadel",
        registry: Optional[Any] = None
    ):
        self.namespace = namespace
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not available, metrics disabled")
            self._enabled = False
            return
        
        self._enabled = True
        self.registry = registry or CollectorRegistry()
        
        self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        # === Account Metrics ===
        self.account_balance = Gauge(
            f"{self.namespace}_account_balance_usd",
            "Current account balance in USD",
            registry=self.registry
        )
        
        self.account_equity = Gauge(
            f"{self.namespace}_account_equity_usd",
            "Current account equity (balance + unrealized PnL)",
            registry=self.registry
        )
        
        # === PnL Metrics ===
        self.daily_pnl = Gauge(
            f"{self.namespace}_daily_pnl_usd",
            "Daily realized PnL in USD",
            registry=self.registry
        )
        
        self.daily_pnl_pct = Gauge(
            f"{self.namespace}_daily_pnl_pct",
            "Daily PnL as percentage",
            registry=self.registry
        )
        
        self.unrealized_pnl = Gauge(
            f"{self.namespace}_unrealized_pnl_usd",
            "Unrealized PnL in USD",
            registry=self.registry
        )
        
        self.total_pnl = Counter(
            f"{self.namespace}_total_pnl_usd",
            "Total cumulative PnL in USD",
            registry=self.registry
        )
        
        # === Drawdown Metrics ===
        self.current_drawdown = Gauge(
            f"{self.namespace}_current_drawdown_pct",
            "Current drawdown percentage from peak",
            registry=self.registry
        )
        
        self.max_drawdown = Gauge(
            f"{self.namespace}_max_drawdown_pct",
            "Maximum drawdown percentage ever",
            registry=self.registry
        )
        
        # === Trade Metrics ===
        self.trades_total = Counter(
            f"{self.namespace}_trades_total",
            "Total number of trades executed",
            ['side', 'result'],  # labels: buy/sell, win/loss
            registry=self.registry
        )
        
        self.trade_pnl_histogram = Histogram(
            f"{self.namespace}_trade_pnl_pct",
            "Distribution of trade PnL percentages",
            buckets=[-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20],
            registry=self.registry
        )
        
        self.trade_duration = Histogram(
            f"{self.namespace}_trade_duration_hours",
            "Distribution of trade durations in hours",
            buckets=[0.5, 1, 2, 4, 8, 12, 24, 48, 72, 168],
            registry=self.registry
        )
        
        self.open_positions = Gauge(
            f"{self.namespace}_open_positions",
            "Number of currently open positions",
            registry=self.registry
        )
        
        # === Win Rate Metrics ===
        self.win_rate = Gauge(
            f"{self.namespace}_win_rate",
            "Current win rate (0-1)",
            registry=self.registry
        )
        
        self.profit_factor = Gauge(
            f"{self.namespace}_profit_factor",
            "Profit factor (gross profit / gross loss)",
            registry=self.registry
        )
        
        # === Risk Metrics ===
        self.sharpe_ratio = Gauge(
            f"{self.namespace}_sharpe_ratio",
            "Rolling Sharpe ratio",
            registry=self.registry
        )
        
        self.sortino_ratio = Gauge(
            f"{self.namespace}_sortino_ratio",
            "Rolling Sortino ratio",
            registry=self.registry
        )
        
        self.var_95 = Gauge(
            f"{self.namespace}_var_95_pct",
            "95% Value at Risk",
            registry=self.registry
        )
        
        # === Exposure Metrics ===
        self.total_exposure = Gauge(
            f"{self.namespace}_total_exposure_usd",
            "Total position exposure in USD",
            registry=self.registry
        )
        
        self.exposure_pct = Gauge(
            f"{self.namespace}_exposure_pct",
            "Exposure as percentage of equity",
            registry=self.registry
        )
        
        # === Circuit Breaker ===
        self.circuit_breaker_state = Gauge(
            f"{self.namespace}_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            registry=self.registry
        )
        
        self.circuit_breaker_trips = Counter(
            f"{self.namespace}_circuit_breaker_trips_total",
            "Total circuit breaker trips",
            ['reason'],
            registry=self.registry
        )
        
        # === Order Metrics ===
        self.orders_submitted = Counter(
            f"{self.namespace}_orders_submitted_total",
            "Total orders submitted",
            ['type'],  # market, limit, stop
            registry=self.registry
        )
        
        self.orders_filled = Counter(
            f"{self.namespace}_orders_filled_total",
            "Total orders filled",
            registry=self.registry
        )
        
        self.slippage_total = Counter(
            f"{self.namespace}_slippage_total_usd",
            "Total slippage cost in USD",
            registry=self.registry
        )
        
        self.commission_total = Counter(
            f"{self.namespace}_commission_total_usd",
            "Total commission paid in USD",
            registry=self.registry
        )
        
        # === System Metrics ===
        self.strategy_latency = Summary(
            f"{self.namespace}_strategy_latency_seconds",
            "Strategy signal calculation latency",
            registry=self.registry
        )
        
        self.data_latency = Gauge(
            f"{self.namespace}_data_latency_seconds",
            "Current market data latency",
            registry=self.registry
        )
        
        self.api_errors = Counter(
            f"{self.namespace}_api_errors_total",
            "Total API errors",
            ['endpoint'],
            registry=self.registry
        )
        
        self.last_update_timestamp = Gauge(
            f"{self.namespace}_last_update_timestamp",
            "Timestamp of last metric update",
            registry=self.registry
        )
    
    def update_account(
        self,
        balance: float,
        equity: float,
        unrealized: float
    ) -> None:
        """Update account metrics."""
        if not self._enabled:
            return
        
        self.account_balance.set(balance)
        self.account_equity.set(equity)
        self.unrealized_pnl.set(unrealized)
        self._update_timestamp()
    
    def update_pnl(
        self,
        daily_pnl: float,
        daily_pnl_pct: float,
        trade_pnl: Optional[float] = None
    ) -> None:
        """Update PnL metrics."""
        if not self._enabled:
            return
        
        self.daily_pnl.set(daily_pnl)
        self.daily_pnl_pct.set(daily_pnl_pct)
        
        if trade_pnl is not None and trade_pnl > 0:
            self.total_pnl.inc(trade_pnl)
        
        self._update_timestamp()
    
    def update_drawdown(
        self,
        current: float,
        maximum: float
    ) -> None:
        """Update drawdown metrics."""
        if not self._enabled:
            return
        
        self.current_drawdown.set(current)
        self.max_drawdown.set(maximum)
    
    def record_trade(
        self,
        side: str,
        pnl_pct: float,
        duration_hours: float
    ) -> None:
        """Record a completed trade."""
        if not self._enabled:
            return
        
        result = "win" if pnl_pct > 0 else "loss"
        self.trades_total.labels(side=side, result=result).inc()
        self.trade_pnl_histogram.observe(pnl_pct * 100)  # Convert to percentage
        self.trade_duration.observe(duration_hours)
        self._update_timestamp()
    
    def update_positions(self, count: int) -> None:
        """Update open positions count."""
        if not self._enabled:
            return
        self.open_positions.set(count)
    
    def update_performance(
        self,
        win_rate: float,
        profit_factor: float,
        sharpe: float,
        sortino: float,
        var_95: float
    ) -> None:
        """Update performance metrics."""
        if not self._enabled:
            return
        
        self.win_rate.set(win_rate)
        self.profit_factor.set(profit_factor)
        self.sharpe_ratio.set(sharpe)
        self.sortino_ratio.set(sortino)
        self.var_95.set(var_95)
    
    def update_exposure(
        self,
        total_usd: float,
        percentage: float
    ) -> None:
        """Update exposure metrics."""
        if not self._enabled:
            return
        
        self.total_exposure.set(total_usd)
        self.exposure_pct.set(percentage)
    
    def set_circuit_breaker_state(
        self,
        state: str,
        trip_reason: Optional[str] = None
    ) -> None:
        """Update circuit breaker state."""
        if not self._enabled:
            return
        
        state_map = {"closed": 0, "half_open": 1, "open": 2}
        self.circuit_breaker_state.set(state_map.get(state, -1))
        
        if trip_reason:
            self.circuit_breaker_trips.labels(reason=trip_reason).inc()
    
    def record_order(
        self,
        order_type: str,
        filled: bool = True,
        slippage: float = 0,
        commission: float = 0
    ) -> None:
        """Record order execution."""
        if not self._enabled:
            return
        
        self.orders_submitted.labels(type=order_type).inc()
        
        if filled:
            self.orders_filled.inc()
            if slippage > 0:
                self.slippage_total.inc(slippage)
            if commission > 0:
                self.commission_total.inc(commission)
    
    def record_api_error(self, endpoint: str) -> None:
        """Record API error."""
        if not self._enabled:
            return
        self.api_errors.labels(endpoint=endpoint).inc()
    
    def observe_latency(
        self,
        latency_seconds: float,
        metric_type: str = "strategy"
    ) -> None:
        """Observe latency."""
        if not self._enabled:
            return
        
        if metric_type == "strategy":
            self.strategy_latency.observe(latency_seconds)
        elif metric_type == "data":
            self.data_latency.set(latency_seconds)
    
    def _update_timestamp(self) -> None:
        """Update last update timestamp."""
        self.last_update_timestamp.set(time.time())
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        if not self._enabled:
            return b""
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get content type for Prometheus."""
        if not self._enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST


# Global metrics instance
_metrics: Optional[TradingMetrics] = None


def get_metrics() -> TradingMetrics:
    """Get global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics()
    return _metrics


def init_metrics(**kwargs) -> TradingMetrics:
    """Initialize global metrics."""
    global _metrics
    _metrics = TradingMetrics(**kwargs)
    return _metrics
