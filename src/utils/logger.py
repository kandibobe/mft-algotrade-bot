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
import sys
from typing import Any, Dict, Optional

import structlog


def setup_structured_logging(
    level: str = "INFO",
    json_output: bool = True,
    enable_console: bool = True,
    enable_file: bool = False,
    file_path: Optional[str] = None,
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
        format="%(message)s", level=log_level, handlers=[]  # We'll add handlers below
    )

    # Create handlers
    handlers = []

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        handlers.append(console_handler)

    if enable_file and file_path:
        file_handler = logging.FileHandler(file_path)
        handlers.append(file_handler)

    # Apply handlers to root logger
    root_logger = logging.getLogger()
    for handler in handlers:
        root_logger.addHandler(handler)

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add JSON renderer for ELK or console renderer for development
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
        wrapper_class=structlog.stdlib.BoundLogger,
    )

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


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
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
    pnl: Optional[float] = None,
    strategy: Optional[str] = None,
    order_id: Optional[str] = None,
    exchange: Optional[str] = None,
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
    price: Optional[float] = None,
    status: str = "created",
    reason: Optional[str] = None,
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
    confidence: Optional[float] = None,
    indicators: Optional[Dict[str, Any]] = None,
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
    exception: Optional[Exception] = None,
    component: Optional[str] = None,
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
    tags: Optional[Dict[str, str]] = None,
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
