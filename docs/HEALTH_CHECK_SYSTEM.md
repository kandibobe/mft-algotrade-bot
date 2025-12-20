# Health Check System for Kubernetes Orchestration

## Overview

The Stoic Citadel Health Check System provides comprehensive health monitoring for all critical components of the trading bot. It's designed for Kubernetes orchestration with liveness and readiness probes, enabling automatic recovery and traffic management in production environments.

## Problem Statement

Traditional trading bots lack proper health monitoring, making it difficult to:
1. Detect component failures before they affect trading
2. Implement graceful degradation
3. Integrate with container orchestration systems
4. Provide real-time system status to operators

## Architecture

### Core Components

The health check system consists of:

1. **`HealthCheck` Class** - Main health check orchestrator
2. **FastAPI Application** - REST API for Kubernetes probes
3. **Component Checkers** - Individual health checks for each subsystem

### Supported Health Checks

| Component | Description | Criticality |
|-----------|-------------|-------------|
| **Exchange Connection** | Verifies API connectivity and latency | High |
| **Database** | Tests database connection and queries | High |
| **ML Model Inference** | Validates ML model responsiveness | Medium |
| **Circuit Breaker** | Checks risk protection status | High |
| **Redis** | Tests Redis cache connectivity | Medium |
| **System Resources** | Monitors CPU, memory, disk usage | Medium |

## Usage

### As a Standalone Service

```bash
# Start health check service
uvicorn src.monitoring.health_check:app --host 0.0.0.0 --port 8080

# Or using the provided script
python -m src.monitoring.health_check
```

### Integration with Main Bot

```python
from src.monitoring.health_check import HealthCheck

# Initialize health check system
health_check = HealthCheck(bot=your_bot_instance)

# Run all checks
results = await health_check.run_all_checks()

# Check specific component
exchange_status = await health_check.check_exchange()
```

## Kubernetes Integration

### Liveness Probe
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3
```

### Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 1
```

### Deployment Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stoic-citadel
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: trading-bot
        image: stoic-citadel:latest
        ports:
        - containerPort: 8080
        env:
        - name: HEALTH_CHECK_PORT
          value: "8080"
```

## API Endpoints

### `GET /health` - Liveness Probe
Returns 200 OK if the process is alive.

**Response:**
```json
{
  "status": "alive",
  "timestamp": "2025-12-20T11:12:35.676444Z"
}
```

### `GET /ready` - Readiness Probe
Returns 200 OK if all critical components are healthy, 503 Service Unavailable otherwise.

**Response (Healthy):**
```json
{
  "status": "ready",
  "timestamp": "2025-12-20T11:12:35.676444Z",
  "checks": {
    "exchange_connection": {
      "status": "healthy",
      "details": {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "last_price": 50000.0,
        "latency_seconds": 0.123
      },
      "healthy": true
    },
    "database": {
      "status": "healthy",
      "details": {
        "query": "SELECT 1",
        "result": 1,
        "latency_seconds": 0.045
      },
      "healthy": true
    }
  }
}
```

### `GET /health/detailed` - Detailed Health Check
Returns comprehensive health status of all components.

### `GET /health/{component}` - Component-specific Check
Check health of a specific component.

## Configuration

### Environment Variables

```bash
# Health check configuration
HEALTH_CHECK_PORT=8080
HEALTH_CHECK_TIMEOUT=5.0
HEALTH_CHECK_ENABLE_METRICS=true

# Component-specific timeouts
EXCHANGE_CHECK_TIMEOUT=5.0
DATABASE_CHECK_TIMEOUT=3.0
ML_MODEL_CHECK_TIMEOUT=10.0
```

### Customizing Health Checks

```python
from src.monitoring.health_check import HealthCheck

class CustomHealthCheck(HealthCheck):
    def __init__(self, bot=None):
        super().__init__(bot)
        
        # Add custom checks
        self.checks["custom_service"] = self.check_custom_service
    
    async def check_custom_service(self):
        """Custom health check for your service."""
        try:
            # Your health check logic
            return {
                "status": "healthy",
                "details": {"custom": "metric"},
                "healthy": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": f"Custom service error: {str(e)}",
                "healthy": False
            }
```

## Alerting and Monitoring

### Prometheus Metrics

The health check system exposes Prometheus metrics:

```python
# Example metrics
health_check_status{component="exchange"} 1  # 1 = healthy, 0 = unhealthy
health_check_latency_seconds{component="exchange"} 0.123
health_check_last_run_timestamp 1679323456.789
```

### Grafana Dashboard

Create a dashboard with:
1. Component health status (green/red indicators)
2. Check latency over time
3. Error rates and trends
4. System resource utilization

### Alert Rules

```yaml
# Alert on unhealthy components
- alert: ComponentUnhealthy
  expr: health_check_status == 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Component {{ $labels.component }} is unhealthy"
    description: "Component {{ $labels.component }} has been unhealthy for 2 minutes"

# Alert on high latency
- alert: HealthCheckHighLatency
  expr: health_check_latency_seconds > 5
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Health check latency is high"
    description: "Component {{ $labels.component }} latency is {{ $value }} seconds"
```

## Testing

### Unit Tests

```bash
# Run health check tests
pytest tests/test_monitoring/test_health_check.py -v
```

### Integration Tests

```bash
# Test with Docker Compose
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Manual Testing

```bash
# Test liveness probe
curl http://localhost:8080/health

# Test readiness probe
curl http://localhost:8080/ready

# Test detailed health
curl http://localhost:8080/health/detailed

# Test specific component
curl http://localhost:8080/health/exchange_connection
```

## Best Practices

### 1. Critical vs Non-critical Checks
- Mark exchange and database as critical (affects readiness)
- Mark system resources as non-critical (warning only)

### 2. Timeout Configuration
- Set appropriate timeouts for each component
- Use shorter timeouts for liveness probes
- Use longer timeouts for detailed checks

### 3. Graceful Degradation
- Continue operating if non-critical components fail
- Implement circuit breakers for failed components
- Provide fallback mechanisms

### 4. Security
- Don't expose sensitive information in health responses
- Use internal network for health check endpoints
- Implement authentication for detailed endpoints

## Troubleshooting

### Common Issues

1. **Exchange API Timeouts**
   - Check network connectivity
   - Verify API keys and permissions
   - Consider rate limiting

2. **Database Connection Failures**
   - Check database service status
   - Verify connection strings
   - Monitor connection pool

3. **ML Model Unresponsive**
   - Check model loading
   - Verify feature dimensions
   - Monitor inference latency

4. **High System Resource Usage**
   - Check for memory leaks
   - Optimize resource-intensive operations
   - Consider horizontal scaling

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Dependencies

- `fastapi>=0.104.0` - REST API framework
- `uvicorn>=0.24.0` - ASGI server
- `psutil>=5.9.0` - System resource monitoring
- `redis>=4.5.0` - Redis client (optional)
- `ccxt>=4.0.0` - Exchange API (optional)

## Conclusion

The Health Check System provides:

1. **Production Readiness** - Kubernetes integration with liveness/readiness probes
2. **Comprehensive Monitoring** - Health status of all critical components
3. **Automatic Recovery** - Enables automatic pod restart on failure
4. **Operational Visibility** - Real-time system status for operators
5. **Graceful Degradation** - Continue operation with degraded functionality

This implementation ensures the trading bot can be reliably deployed and managed in production Kubernetes environments.
