"""
Stoic Citadel - Prometheus Metrics Exporter
============================================

Exposes trading metrics via HTTP endpoint for Prometheus scraping.
This module starts a simple HTTP server that serves metrics in Prometheus format.

Usage:
    python -m src.monitoring.metrics_exporter
    or import and call start_metrics_server()

Environment variables:
    METRICS_PORT: Port to expose metrics on (default: 8000)
    METRICS_HOST: Host to bind to (default: 0.0.0.0)
    METRICS_ENABLED: Enable/disable metrics server (default: true)
"""

import logging
import os
import threading
import time

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("prometheus_client not available, metrics will be disabled")

# Import existing trading metrics
try:
    from src.monitoring.trading_metrics import TradingMetrics, get_metrics

    TRADING_METRICS_AVAILABLE = True
except ImportError:
    TRADING_METRICS_AVAILABLE = False


logger = logging.getLogger(__name__)


class TradingMetricsExporter:
    """
    Prometheus metrics exporter for trading bot.

    This class provides a simplified interface for recording trading metrics
    and exposes them via HTTP endpoint for Prometheus scraping.
    """

    def __init__(self, namespace: str = "stoic_citadel"):
        """
        Initialize metrics exporter.

        Args:
            namespace: Prometheus metrics namespace
        """
        self.namespace = namespace
        self._enabled = PROMETHEUS_AVAILABLE

        if not self._enabled:
            logger.warning("Prometheus client not available, metrics disabled")
            return

        # Initialize metrics
        self._init_metrics()

        # Use existing trading metrics if available
        if TRADING_METRICS_AVAILABLE:
            self.trading_metrics = get_metrics()
            logger.info("Using existing TradingMetrics instance")
        else:
            self.trading_metrics = None
            logger.info("Created standalone metrics exporter")

    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Counters
        self.trades_total = Counter(
            f"{self.namespace}_trades_total", "Total trades executed", ["side", "status"]
        )

        self.orders_total = Counter(
            f"{self.namespace}_orders_total", "Total orders submitted", ["order_type"]
        )

        # Histograms
        self.order_latency = Histogram(
            f"{self.namespace}_order_latency_seconds",
            "Order execution latency in seconds",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        )

        self.ml_inference_latency = Histogram(
            f"{self.namespace}_ml_inference_latency_ms",
            "ML inference time in milliseconds",
            buckets=[1, 5, 10, 50, 100, 500, 1000, 5000],
        )

        # Gauges
        self.portfolio_value = Gauge(
            f"{self.namespace}_portfolio_value_usd", "Current portfolio value in USD"
        )

        self.open_positions = Gauge(f"{self.namespace}_open_positions", "Number of open positions")

        self.daily_pnl_pct = Gauge(f"{self.namespace}_daily_pnl_pct", "Daily PnL percentage")

        self.current_drawdown_pct = Gauge(
            f"{self.namespace}_current_drawdown_pct", "Current Daily Drawdown percentage (Equity based)"
        )

        self.total_exposure_usd = Gauge(
            f"{self.namespace}_total_exposure_usd", "Total open exposure in USD"
        )

        self.rolling_sharpe = Gauge(
            f"{self.namespace}_rolling_sharpe", "Rolling Sharpe Ratio Estimate"
        )

        self.circuit_breaker_status = Gauge(
            f"{self.namespace}_circuit_breaker_status", "Circuit breaker status (0=off, 1=on)"
        )

        # MFT / Micro-structure metrics
        self.market_spread_pct = Gauge(
            f"{self.namespace}_market_spread_pct", "Current bid-ask spread percentage", ["symbol"]
        )

        self.orderbook_imbalance = Gauge(
            f"{self.namespace}_orderbook_imbalance",
            "Current orderbook imbalance (-1 to 1)",
            ["symbol"],
        )

        self.ws_message_latency = Histogram(
            f"{self.namespace}_ws_message_latency_ms",
            "WebSocket message processing latency in milliseconds",
            buckets=[0.1, 0.5, 1, 5, 10, 50, 100],
        )

        # Smart limit order metrics
        self.fee_savings_total = Gauge(
            f"{self.namespace}_fee_savings_total_usd",
            "Total fee savings from smart limit orders in USD",
        )

        self.fee_savings_per_trade = Histogram(
            f"{self.namespace}_fee_savings_per_trade_usd",
            "Fee savings per trade from smart limit orders in USD",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        )

        self.smart_limit_maker_fills = Counter(
            f"{self.namespace}_smart_limit_maker_fills_total",
            "Total maker fills from smart limit orders",
        )

        self.smart_limit_taker_fills = Counter(
            f"{self.namespace}_smart_limit_taker_fills_total",
            "Total taker fills from smart limit orders",
        )

        # ML metrics
        self.ml_prediction_confidence = Gauge(
            f"{self.namespace}_ml_prediction_confidence",
            "ML model prediction confidence (0.0 to 1.0)",
        )

        self.ml_prediction_confidence_histogram = Histogram(
            f"{self.namespace}_ml_prediction_confidence_distribution",
            "Distribution of ML prediction confidence values",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        self.ml_predictions_total = Counter(
            f"{self.namespace}_ml_predictions_total",
            "Total ML predictions made",
            ["model", "prediction_type"],
        )

        # HRP Metrics
        self.hrp_weights = Gauge(
            f"{self.namespace}_hrp_asset_weight", "Calculated HRP weight for an asset", ["asset"]
        )

        # TWAP/VWAP Metrics
        self.twap_vwap_orders_total = Counter(
            f"{self.namespace}_twap_vwap_orders_total", "Total TWAP/VWAP orders", ["order_type"]
        )
        self.twap_vwap_slippage = Histogram(
            f"{self.namespace}_twap_vwap_slippage_pct",
            "Slippage of TWAP/VWAP orders in percent",
            ["order_type"],
        )

        # Rebalancer Metrics
        self.rebalancer_runs_total = Counter(
            f"{self.namespace}_rebalancer_runs_total", "Total rebalancer runs"
        )
        self.portfolio_deviation = Gauge(
            f"{self.namespace}_portfolio_deviation_pct",
            "Portfolio deviation from target in percent",
        )

        # System metrics
        self.metrics_up = Gauge(f"{self.namespace}_up", "Metrics exporter status (1=up, 0=down)")
        self.metrics_up.set(1)

        logger.debug("Initialized Prometheus metrics")

    def record_trade(self, side: str, status: str, execution_time: float) -> None:
        """
        Record a trade execution.

        Args:
            side: 'buy' or 'sell'
            status: 'filled', 'partial', 'cancelled', 'rejected'
            execution_time: Execution latency in seconds
        """
        if not self._enabled:
            return

        self.trades_total.labels(side=side, status=status).inc()
        self.order_latency.observe(execution_time)

        # Also update existing trading metrics if available
        if self.trading_metrics:
            self.trading_metrics.record_trade(side, 0.0, 0.0)  # Placeholder

    def record_order(self, order_type: str, filled: bool = True) -> None:
        """
        Record an order submission.

        Args:
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            filled: Whether the order was filled
        """
        if not self._enabled:
            return

        self.orders_total.labels(order_type=order_type).inc()

        if self.trading_metrics:
            self.trading_metrics.record_order(order_type, filled)

    def update_portfolio_metrics(
        self,
        value: float,
        positions: list,
        pnl_pct: float,
        drawdown_pct: float = 0.0,
        exposure: float = 0.0,
        sharpe: float = 0.0,
    ) -> None:
        """
        Update portfolio metrics.

        Args:
            value: Current portfolio value in USD
            positions: List of open positions
            pnl_pct: Daily PnL percentage
            drawdown_pct: Current drawdown percentage
            exposure: Total exposure in USD
            sharpe: Rolling Sharpe ratio
        """
        if not self._enabled:
            return

        self.portfolio_value.set(value)
        self.open_positions.set(len(positions))
        self.daily_pnl_pct.set(pnl_pct)
        self.current_drawdown_pct.set(drawdown_pct)
        self.total_exposure_usd.set(exposure)
        self.rolling_sharpe.set(sharpe)

        if self.trading_metrics:
            self.trading_metrics.update_account(value, value, 0.0)
            self.trading_metrics.update_pnl(0.0, pnl_pct)
            self.trading_metrics.update_positions(len(positions))

    def set_circuit_breaker_status(self, status: int) -> None:
        """
        Set circuit breaker status.

        Args:
            status: 0 = off, 1 = on
        """
        if not self._enabled:
            return

        self.circuit_breaker_status.set(status)

        if self.trading_metrics:
            state = "open" if status == 1 else "closed"
            self.trading_metrics.set_circuit_breaker_state(state)

    def record_fee_savings(self, savings_usd: float, trade_type: str = "smart_limit") -> None:
        """
        Record fee savings from smart limit orders.

        Args:
            savings_usd: Fee savings in USD
            trade_type: Type of trade (smart_limit, regular, etc.)
        """
        if not self._enabled:
            return

        self.fee_savings_total.inc(savings_usd)
        self.fee_savings_per_trade.observe(savings_usd)

        if self.trading_metrics:
            # Update existing metrics if available
            pass

    def record_smart_limit_fill(self, fill_type: str) -> None:
        """
        Record smart limit order fill type.

        Args:
            fill_type: 'maker' or 'taker'
        """
        if not self._enabled:
            return

        if fill_type == "maker":
            self.smart_limit_maker_fills.inc()
        elif fill_type == "taker":
            self.smart_limit_taker_fills.inc()

    def record_ml_prediction(
        self, confidence: float, model: str = "ensemble", prediction_type: str = "binary"
    ) -> None:
        """
        Record ML prediction with confidence.

        Args:
            confidence: Prediction confidence (0.0 to 1.0)
            model: Model name (ensemble, xgboost, etc.)
            prediction_type: Type of prediction (binary, regression, etc.)
        """
        if not self._enabled:
            return

        self.ml_prediction_confidence.set(confidence)
        self.ml_prediction_confidence_histogram.observe(confidence)
        self.ml_predictions_total.labels(model=model, prediction_type=prediction_type).inc()

        if self.trading_metrics:
            self.trading_metrics.record_prediction(confidence)

    def record_ml_inference(self, latency_ms: float) -> None:
        """
        Record ML inference latency.

        Args:
            latency_ms: Inference time in milliseconds
        """
        if not self._enabled:
            return

        self.ml_inference_latency.observe(latency_ms)

        if self.trading_metrics:
            self.trading_metrics.observe_latency(latency_ms / 1000.0, "strategy")

    def record_ws_metrics(self, symbol: str, spread_pct: float, imbalance: float) -> None:
        """Record real-time market microstructure metrics."""
        if not self._enabled:
            return
        self.market_spread_pct.labels(symbol=symbol).set(spread_pct)
        self.orderbook_imbalance.labels(symbol=symbol).set(imbalance)

    def record_ws_latency(self, latency_ms: float) -> None:
        """Record websocket processing latency."""
        if not self._enabled:
            return
        self.ws_message_latency.observe(latency_ms)

    def record_hrp_weights(self, weights: dict) -> None:
        """Record HRP weights."""
        if not self._enabled:
            return
        for asset, weight in weights.items():
            self.hrp_weights.labels(asset=asset).set(weight)

    def record_twap_vwap_order(self, order_type: str, slippage_pct: float) -> None:
        """Record TWAP/VWAP order."""
        if not self._enabled:
            return
        self.twap_vwap_orders_total.labels(order_type=order_type).inc()
        self.twap_vwap_slippage.labels(order_type=order_type).observe(slippage_pct)

    def record_rebalancer_run(self, deviation_pct: float) -> None:
        """Record rebalancer run."""
        if not self._enabled:
            return
        self.rebalancer_runs_total.inc()
        self.portfolio_deviation.set(deviation_pct)


# Global exporter instance
_exporter: TradingMetricsExporter | None = None
_server_thread: threading.Thread | None = None


def get_exporter() -> TradingMetricsExporter:
    """
    Get global metrics exporter instance.

    Returns:
        TradingMetricsExporter instance
    """
    global _exporter
    if _exporter is None:
        _exporter = TradingMetricsExporter()
    return _exporter


def start_metrics_server(
    host: str = "0.0.0.0", port: int = 8000, daemon: bool = True
) -> threading.Thread | None:
    """
    Start Prometheus metrics HTTP server in a background thread.

    Args:
        host: Host to bind to
        port: Port to listen on
        daemon: Whether the server thread should be daemon

    Returns:
        Thread object if server started, None otherwise
    """
    global _server_thread

    if not PROMETHEUS_AVAILABLE:
        logger.error("Cannot start metrics server: prometheus_client not available")
        return None

    if _server_thread is not None and _server_thread.is_alive():
        logger.warning("Metrics server already running")
        return _server_thread

    def run_server():
        """Run the metrics server."""
        try:
            logger.info(f"Starting Prometheus metrics server on {host}:{port}")
            start_http_server(port, addr=host)
            logger.info(f"Metrics available at http://{host}:{port}/metrics")

            # Keep the thread alive
            while True:
                time.sleep(60)
        except Exception as e:
            logger.error(f"Metrics server error: {e}")

    _server_thread = threading.Thread(target=run_server, daemon=daemon)
    _server_thread.start()

    # Initialize exporter
    get_exporter()

    return _server_thread


def stop_metrics_server() -> None:
    """Stop the metrics server thread."""
    global _server_thread
    if _server_thread is not None:
        # Note: start_http_server doesn't have a stop method
        # In production, use proper shutdown
        logger.info("Stopping metrics server")
        _server_thread = None


def main():
    """Main entry point for standalone metrics exporter."""
    import argparse

    parser = argparse.ArgumentParser(description="Stoic Citadel Prometheus Metrics Exporter")
    parser.add_argument(
        "--host",
        default=os.getenv("METRICS_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("METRICS_PORT", "8000")),
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not PROMETHEUS_AVAILABLE:
        logger.error("prometheus_client is not installed. Please install it with:")
        logger.error("  pip install prometheus-client")
        return 1

    logger.info(f"Starting Stoic Citadel Metrics Exporter on {args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")

    try:
        # Start server in main thread (blocking)
        start_http_server(args.port, addr=args.host)
        logger.info(f"Metrics available at http://{args.host}:{args.port}/metrics")

        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())