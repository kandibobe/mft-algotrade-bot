# Prometheus Metrics Exporter

## Overview

This implementation adds Prometheus metrics export capability to the Stoic Citadel trading bot. The solution addresses the problem of missing metrics export for Prometheus and difficulty tracking performance degradation.

## Files Created/Modified

### New Files

1. **`src/monitoring/metrics_exporter.py`** - Main metrics exporter module
   - `TradingMetricsExporter` class with simplified interface for recording metrics
   - HTTP server for Prometheus scraping (port 8000 by default)
   - Integration with existing `TradingMetrics` class
   - Support for environment variable configuration

### Modified Files

1. **`src/order_manager/order_executor.py`** - Order execution metrics
   - Added import of `metrics_exporter`
   - Added metrics recording in `execute()` method
   - Records trade success/failure, latency, order type

2. **`src/ml/inference_service.py`** - ML inference metrics
   - Added import of `metrics_exporter`
   - Added ML inference latency recording in `predict()` method

3. **`src/risk/circuit_breaker.py`** - Circuit breaker metrics
   - Added import of `metrics_exporter`
   - Added circuit breaker status updates in `_trip()`, `_reset()`, and `_should_auto_reset()` methods

## Metrics Exposed

### Counters
- `stoic_citadel_trades_total` - Total trades executed (labels: side, status)
- `stoic_citadel_orders_total` - Total orders submitted (labels: order_type)

### Histograms
- `stoic_citadel_order_latency_seconds` - Order execution latency distribution
- `stoic_citadel_ml_inference_latency_ms` - ML inference time distribution

### Gauges
- `stoic_citadel_portfolio_value_usd` - Current portfolio value
- `stoic_citadel_open_positions` - Number of open positions
- `stoic_citadel_daily_pnl_pct` - Daily PnL percentage
- `stoic_citadel_circuit_breaker_status` - Circuit breaker status (0=off, 1=on)
- `stoic_citadel_up` - Metrics exporter status (1=up)

## Usage

### Starting the Metrics Server

```python
from src.monitoring.metrics_exporter import start_metrics_server

# Start server on default port 8000
server_thread = start_metrics_server()

# Or with custom configuration
server_thread = start_metrics_server(host="0.0.0.0", port=8000)
```

### Recording Metrics

```python
from src.monitoring.metrics_exporter import get_exporter

exporter = get_exporter()

# Record a trade
exporter.record_trade("buy", "filled", 0.125)

# Record an order
exporter.record_order("limit", filled=True)

# Update portfolio metrics
exporter.update_portfolio_metrics(
    value=10000.0,
    positions=[{"symbol": "BTC/USDT", "size": 0.5}],
    pnl_pct=2.5
)

# Set circuit breaker status
exporter.set_circuit_breaker_status(1)  # 1 = tripped

# Record ML inference latency
exporter.record_ml_inference(45.7)  # milliseconds
```

### Command Line Interface

```bash
# Start standalone metrics exporter
python -m src.monitoring.metrics_exporter --port 8000

# With custom host
python -m src.monitoring.metrics_exporter --host 0.0.0.0 --port 8000
```

## Environment Variables

- `METRICS_PORT` - Port to expose metrics on (default: 8000)
- `METRICS_HOST` - Host to bind to (default: 0.0.0.0)
- `METRICS_ENABLED` - Enable/disable metrics server (default: true)

## Integration with Existing System

The metrics exporter integrates seamlessly with the existing trading bot:

1. **Order Execution** - Metrics automatically recorded when orders are executed
2. **ML Inference** - Latency metrics recorded during model predictions
3. **Circuit Breaker** - Status updates when circuit breaker trips/resets
4. **Existing TradingMetrics** - Backward compatibility with existing metrics system

## Testing

Run the test script to verify functionality:

```bash
python test_metrics.py
```

## Dependencies

- `prometheus-client` (optional, with graceful fallback)
- Existing project dependencies

## Monitoring Stack Integration

The metrics are compatible with the existing Docker monitoring stack (`docker-compose.monitoring.yml`). Prometheus can be configured to scrape metrics from the exporter endpoint.

Example Prometheus configuration:
```yaml
scrape_configs:
  - job_name: 'stoic_citadel'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

## Benefits

1. **Performance Monitoring** - Track order execution latency and ML inference times
2. **Risk Management** - Monitor circuit breaker status and trading limits
3. **Portfolio Tracking** - Real-time portfolio value and PnL metrics
4. **System Health** - Exporter status and uptime monitoring
5. **Integration Ready** - Seamless integration with Grafana dashboards

## Future Enhancements

1. Add more detailed trade metrics (slippage, commission)
2. Implement histogram buckets customization
3. Add authentication for metrics endpoint
4. Support for pushgateway mode
5. Additional system metrics (memory, CPU, network)
