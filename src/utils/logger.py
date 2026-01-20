"""
Structured logging with structlog for ELK stack integration.

This module provides structured logging capabilities using structlog,
which outputs JSON logs that can be easily ingested by ELK stack
(Elasticsearch, Logstash, Kibana) for advanced log analysis and alerting.

Features:
- JSON-formatted logs for ELK integration
- Structured context (key-value pairs)
- Automatic timestamp, log level, logger name
- Stack traces and exception formatting
- Compatible with standard logging module

Usage:
    from src.utils.logger import setup_structured_logging, log

    # Initialize structured logging (call once at application start)
    setup_structured_logging()

    # Log with structured context
    log.info("trade_executed",
        symbol="BTC/USDT",
        side="buy",
        quantity=0.1,
        price=50000.0,
        pnl=150.50,
        strategy="ensemble_v1"
    )

    # In Kibana, you can now query:
    # strategy:"ensemble_v1" AND pnl:<0
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import sentry_sdk
import structlog
from sentry_sdk.integrations.logging import LoggingIntegration


def setup_structured_logging(
    level: str = "INFO",
    json_output: bool = True,
    enable_console: bool = True,
    enable_file: bool = False,
    file_path: str | None = None,
    sentry_dsn: str | None = None,
) -> None:
    """
    Configure structured logging with structlog.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON format for ELK. If False, use console-friendly format.
        enable_console: Enable logging to console (stdout)
        enable_file: Enable logging to file
        file_path: Path to log file (required if enable_file=True)

    Returns:
        None
    """
    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[],  # We'll add handlers below
    )

    # Create handlers
    handlers = []

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        handlers.append(console_handler)

    if enable_file and file_path:
        # Ensure log directory exists
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        handlers.append(file_handler)

        # Separate error.log
        error_log_path = file_path.replace(".log", "_error.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)

        # Separate trades.log
        trades_log_path = file_path.replace(".log", "_trades.log")
        trades_handler = logging.handlers.RotatingFileHandler(
            trades_log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )

        class TradeFilter(logging.Filter):
            def filter(self, record):
                # Capture trade executions and order updates for the trades log
                msg_str = str(record.msg)
                return hasattr(record, "msg") and (
                    '"event_type": "trade_executed"' in msg_str
                    or '"event_type": "order_update"' in msg_str
                )

        trades_handler.addFilter(TradeFilter())
        handlers.append(trades_handler)

    # Initialize Sentry if DSN is provided
    if sentry_dsn:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,  # Capture info and above as breadcrumbs
            event_level=logging.ERROR,  # Send errors as events
        )
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[sentry_logging],
            traces_sample_rate=0.1,
            # Set environment and release if available
            environment=getattr(sys, "frozen", "development")
            if not hasattr(sys, "frozen")
            else "production",
        )

    # Configure structlog processors
    # These processors are applied to both structlog and standard logging calls
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create formatter that renders JSON or Console
    # This formatter is used by standard logging handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        # These run on the message after it's been processed by shared_processors
        processor=structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(),
        # These run on standard logging messages before they reach the processor
        foreign_pre_chain=shared_processors,
    )

    # Apply formatter to all handlers and add to root logger
    root_logger = logging.getLogger()
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Log initialization
    log = structlog.get_logger()
    log.info(
        "structured_logging_initialized",
        level=level,
        json_output=json_output,
        enable_console=enable_console,
        enable_file=enable_file,
        file_path=file_path,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__). If None, returns root logger.

    Returns:
        structlog BoundLogger instance
    """
    return structlog.get_logger(name)


# Global logger instance for convenience
log = get_logger()

# Default initialization with console JSON output
# This ensures logs work even if setup_structured_logging() wasn't called
# Users should call setup_structured_logging() at application start for proper configuration
_structlog_initialized = False


def ensure_initialized() -> None:
    """Ensure structlog is initialized with default settings if not already configured."""
    global _structlog_initialized
    if not _structlog_initialized:
        # Check if structlog is already configured
        try:
            structlog.get_config()
        except KeyError:
            # Not configured, set up with defaults
            setup_structured_logging(
                level="INFO", json_output=True, enable_console=True, enable_file=False
            )
        _structlog_initialized = True


# Convenience functions for common log patterns
def log_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    pnl: float | None = None,
    strategy: str | None = None,
    order_id: str | None = None,
    exchange: str | None = None,
    **additional_context: Any,
) -> None:
    """
    Log a trade execution with structured data.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        side: "buy" or "sell"
        quantity: Trade quantity
        price: Execution price
        pnl: Profit and loss (if available)
        strategy: Strategy name
        order_id: Order identifier
        exchange: Exchange name
        **additional_context: Additional key-value pairs to log
    """
    ensure_initialized()
    context = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "event_type": "trade_executed",
    }
    if pnl is not None:
        context["pnl"] = pnl
    if strategy is not None:
        context["strategy"] = strategy
    if order_id is not None:
        context["order_id"] = order_id
    if exchange is not None:
        context["exchange"] = exchange
    context.update(additional_context)

    log.info("trade_executed", **context)


def log_order(
    order_id: str,
    symbol: str,
    order_type: str,
    side: str,
    quantity: float,
    price: float | None = None,
    status: str = "created",
    reason: str | None = None,
    **additional_context: Any,
) -> None:
    """
    Log order creation/update with structured data.

    Args:
        order_id: Order identifier
        symbol: Trading pair
        order_type: "market", "limit", "stop_loss", "take_profit"
        side: "buy" or "sell"
        quantity: Order quantity
        price: Order price (for limit orders)
        status: Order status ("created", "submitted", "filled", "cancelled", "rejected")
        reason: Reason for status change
        **additional_context: Additional key-value pairs to log
    """
    ensure_initialized()
    context = {
        "order_id": order_id,
        "symbol": symbol,
        "order_type": order_type,
        "side": side,
        "quantity": quantity,
        "status": status,
        "event_type": "order_update",
        "trace_id": additional_context.get("trace_id")
        or f"trc_{order_id.split('_')[-1] if '_' in order_id else order_id[:8]}",
    }
    if price is not None:
        context["price"] = price
    if reason is not None:
        context["reason"] = reason
    context.update(additional_context)

    log.info("order_update", **context)


def log_strategy_signal(
    strategy: str,
    symbol: str,
    signal: str,
    confidence: float | None = None,
    indicators: dict[str, Any] | None = None,
    **additional_context: Any,
) -> None:
    """
    Log strategy signal with structured data.

    Args:
        strategy: Strategy name
        symbol: Trading pair
        signal: "buy", "sell", "hold"
        confidence: Signal confidence (0.0 to 1.0)
        indicators: Dictionary of indicator values
        **additional_context: Additional key-value pairs to log
    """
    ensure_initialized()
    context = {
        "strategy": strategy,
        "symbol": symbol,
        "signal": signal,
        "event_type": "strategy_signal",
    }
    if confidence is not None:
        context["confidence"] = confidence
    if indicators is not None:
        context["indicators"] = indicators
    context.update(additional_context)

    log.info("strategy_signal", **context)


def log_error(
    error_type: str,
    message: str,
    exception: Exception | None = None,
    component: str | None = None,
    **additional_context: Any,
) -> None:
    """
    Log error with structured data.

    Args:
        error_type: Error category (e.g., "connection", "validation", "execution")
        message: Error message
        exception: Exception object (for stack trace)
        component: Component where error occurred
        **additional_context: Additional key-value pairs to log
    """
    ensure_initialized()
    context = {"error_type": error_type, "message": message, "event_type": "error"}
    if component is not None:
        context["component"] = component
    context.update(additional_context)

    if exception is not None:
        log.error("error_occurred", exc_info=exception, **context)
    else:
        log.error("error_occurred", **context)


def log_metric(
    name: str,
    value: float,
    metric_type: str = "gauge",
    tags: dict[str, str] | None = None,
    **additional_context: Any,
) -> None:
    """
    Log metric for monitoring.

    Args:
        name: Metric name
        value: Metric value
        metric_type: "gauge", "counter", "histogram", "summary"
        tags: Key-value tags for metric
        **additional_context: Additional context
    """
    ensure_initialized()
    context = {
        "metric_name": name,
        "metric_value": value,
        "metric_type": metric_type,
        "event_type": "metric",
    }
    if tags is not None:
        context["tags"] = tags
    context.update(additional_context)

    log.info("metric_recorded", **context)


# Initialize on module import for backward compatibility
ensure_initialized()
