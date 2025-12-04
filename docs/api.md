# API Reference

## Freqtrade REST API

Base URL: `http://localhost:8080/api/v1/`

### Authentication

All endpoints require JWT authentication.

```bash
# Get token
curl -X POST http://localhost:8080/api/v1/token/login \
  -H "Content-Type: application/json" \
  -d '{"username": "stoic_admin", "password": "YOUR_PASSWORD"}'
```

### Endpoints

#### Status

```bash
# Ping
GET /api/v1/ping

# Bot status
GET /api/v1/show_config
GET /api/v1/status
```

#### Trading

```bash
# Current trades
GET /api/v1/status

# Open orders
GET /api/v1/orders

# Trade history
GET /api/v1/trades
GET /api/v1/trades?limit=50

# Force entry
POST /api/v1/forcebuy
{"pair": "BTC/USDT"}

# Force exit
POST /api/v1/forcesell
{"tradeid": 1}
```

#### Performance

```bash
# Profit
GET /api/v1/profit

# Performance per pair
GET /api/v1/performance

# Balance
GET /api/v1/balance
```

#### Control

```bash
# Start/stop trading
POST /api/v1/start
POST /api/v1/stop

# Reload config
POST /api/v1/reload_config
```

## WebSocket API

Connect: `ws://localhost:8080/api/v1/message/ws?token=YOUR_WS_TOKEN`

### Messages

```json
// Subscribe
{"type": "subscribe", "data": ["whitelist", "analyzed_df"]}

// Incoming updates
{"type": "new_candle", "data": {...}}
{"type": "analyzed_df", "data": {...}}
```

## PostgreSQL

Connection: `postgresql://stoic_trader:PASSWORD@localhost:5433/trading_analytics`

### Tables

- `trades` - Trade history
- `performance_metrics` - Daily metrics
- `market_data` - OHLCV data

## Python SDK

```python
from freqtrade_client import FtRestClient

client = FtRestClient(
    "http://localhost:8080",
    "stoic_admin",
    "YOUR_PASSWORD"
)

# Get status
status = client.status()

# Get profit
profit = client.profit()
```
