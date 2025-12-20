# Structured Logging with ELK Stack Integration

## Overview

This document describes the structured logging implementation using `structlog` for ELK (Elasticsearch, Logstash, Kibana) stack integration in the MFT Algotrade Bot.

## Problem Statement

Traditional text-based logs are difficult to analyze and create alerts from. The new structured logging system addresses these issues by:

1. **Structured Data**: Logs as JSON with key-value pairs instead of unstructured text
2. **ELK Integration**: JSON format optimized for Elasticsearch ingestion
3. **Rich Context**: Automatic inclusion of timestamps, log levels, logger names
4. **Queryable**: Easy filtering and alerting in Kibana

## Implementation

### Core Components

The structured logging system is implemented in `src/utils/logger.py` and provides:

1. **`setup_structured_logging()`**: Configures logging with JSON output for ELK
2. **Convenience Functions**: Specialized logging for trading events
3. **Backward Compatibility**: Works with existing logging code

### Key Features

- **JSON Output**: Structured logs for ELK stack
- **Development Mode**: Human-readable console output
- **Automatic Fields**: Timestamp, log level, logger name, stack traces
- **Trading-specific**: Predefined log functions for trades, orders, signals

## Usage

### Basic Setup

```python
from src.utils.logger import setup_structured_logging, log

# Initialize once at application start
setup_structured_logging(
    level="INFO",           # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    json_output=True,       # True for ELK, False for development
    enable_console=True,    # Output to console
    enable_file=False,      # Output to file
    file_path=None          # Path to log file if enable_file=True
)

# Basic logging
log.info("event_name", key1="value1", key2="value2")
```

### Trading-specific Logging

```python
from src.utils.logger import (
    log_trade,
    log_order,
    log_strategy_signal,
    log_error,
    log_metric
)

# Log a trade execution
log_trade(
    symbol="BTC/USDT",
    side="buy",
    quantity=0.1,
    price=50000.0,
    pnl=150.50,
    strategy="ensemble_v1",
    order_id="order_123",
    exchange="binance"
)

# Log order creation/update
log_order(
    order_id="order_456",
    symbol="ETH/USDT",
    order_type="limit",
    side="sell",
    quantity=0.5,
    price=3500.0,
    status="filled",
    reason="take_profit"
)

# Log strategy signal
log_strategy_signal(
    strategy="momentum_v2",
    symbol="SOL/USDT",
    signal="buy",
    confidence=0.85,
    indicators={"rsi": 35.2, "macd": 1.5}
)

# Log error with context
try:
    # Some operation
    pass
except Exception as e:
    log_error(
        error_type="validation",
        message="Validation failed",
        exception=e,
        component="order_validator"
    )

# Log metrics for monitoring
log_metric(
    name="latency_ms",
    value=45.2,
    metric_type="gauge",
    tags={"component": "order_executor", "mode": "live"}
)
```

## ELK Stack Integration

### Log Format

Logs are output as JSON with the following structure:

```json
{
  "level": "info",
  "event": "trade_executed",
  "symbol": "BTC/USDT",
  "side": "buy",
  "quantity": 0.1,
  "price": 50000.0,
  "pnl": 150.5,
  "strategy": "ensemble_v1",
  "logger": "root",
  "timestamp": "2025-12-20T11:12:35.676444Z"
}
```

### Kibana Queries

With structured logging, you can create powerful queries in Kibana:

```kibana
# Find all losing trades for a specific strategy
strategy:"ensemble_v1" AND pnl:<0

# Find failed orders in the last hour
event:"order_update" AND status:"failed" AND @timestamp:[now-1h TO now]

# Monitor high-latency executions
event:"order_execution_success" AND latency_ms:>100

# Track errors by component
event:"error_occurred" AND component:"order_executor"
```

### Alerting Examples

Create alerts based on structured data:

1. **PNL Alert**: Alert when strategy has consecutive losing trades
2. **Latency Alert**: Alert when execution latency exceeds threshold
3. **Error Rate Alert**: Alert when error rate exceeds threshold
4. **Circuit Breaker**: Alert when circuit breaker trips

## Migration Guide

### Updating Existing Code

Replace standard logging with structured logging:

**Before:**
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Order executed: {symbol} {side} {quantity}")
```

**After:**
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
logger.info("order_executed", symbol=symbol, side=side, quantity=quantity)
```

### OrderExecutor Example

The `OrderExecutor` class has been updated to use structured logging:

```python
# Old style (text-based)
logger.info(f"OrderExecutor initialized in {mode.value} mode with DI")

# New style (structured)
logger.info("order_executor_initialized",
    mode=mode.value,
    max_retries=max_retries,
    retry_delay_ms=retry_delay_ms,
    has_circuit_breaker=self.circuit_breaker is not None,
    has_slippage_simulator=self.slippage_simulator is not None
)
```

## Configuration

### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General operational information
- **WARNING**: Warning conditions
- **ERROR**: Error conditions
- **CRITICAL**: Critical conditions

### Output Options

```python
# Production (ELK integration)
setup_structured_logging(
    level="INFO",
    json_output=True,      # JSON for ELK
    enable_console=True,
    enable_file=True,
    file_path="/var/log/trading/app.log"
)

# Development
setup_structured_logging(
    level="DEBUG",
    json_output=False,     # Human-readable
    enable_console=True,
    enable_file=False
)

# Testing
setup_structured_logging(
    level="WARNING",
    json_output=True,
    enable_console=False,
    enable_file=False
)
```

## Best Practices

1. **Use Descriptive Event Names**: `trade_executed`, `order_created`, `signal_generated`
2. **Include Relevant Context**: Always include `symbol`, `strategy`, `order_id` when applicable
3. **Use Consistent Field Names**: Stick to naming conventions (snake_case)
4. **Avoid Sensitive Data**: Never log API keys, passwords, or personal information
5. **Structured Errors**: Include error type, message, and component in error logs
6. **Performance Metrics**: Use `log_metric()` for monitoring key performance indicators

## Testing

Run the test suite to verify logging functionality:

```bash
python test_structured_logging.py
```

## Dependencies

- `structlog>=24.0.0`: Structured logging library
- Standard Python `logging` module

Added to `pyproject.toml` and `requirements.txt`.

## Monitoring and Alerting

### Example Alert Rules

```yaml
# Alert on high error rate
- alert: HighErrorRate
  expr: rate(error_occurred_total[5m]) > 0.1
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value }} per second"

# Alert on losing streak
- alert: LosingStreak
  expr: count_over_time(pnl < 0 [1h]) > 5
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Losing streak detected"
    description: "{{ $value }} consecutive losing trades"
```

## Troubleshooting

### Common Issues

1. **Missing structlog**: Install with `pip install structlog>=24.0.0`
2. **JSON parsing errors**: Ensure logs are valid JSON (no extra formatting)
3. **Missing fields**: All logs include timestamp, level, event, and logger
4. **Performance impact**: Structured logging has minimal performance overhead

### Debug Mode

Enable debug logging for troubleshooting:

```python
setup_structured_logging(level="DEBUG", json_output=False)
```

## Conclusion

Structured logging with ELK integration provides:

1. **Better Observability**: Rich, queryable logs
2. **Easier Debugging**: Structured context for troubleshooting
3. **Powerful Alerting**: Alert on specific conditions
4. **Performance Monitoring**: Track key metrics over time

This implementation maintains backward compatibility while providing modern logging capabilities suitable for production trading systems.
