# Stoic Citadel - Quick Start Guide

## Prerequisites

- Python 3.11+
- Git
- Docker (optional, for production)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Running Tests

### Quick Test (recommended first step)

```bash
# Run all tests
pytest tests/ -v

# Run with summary only
pytest tests/ -q
```

### Test by Module

```bash
# Order Management (12 tests)
pytest tests/test_order_manager/ -v

# ML Pipeline (43 tests)
pytest tests/test_ml/ -v

# Strategies (41 tests)
pytest tests/test_strategies/ -v

# Utils & Data (31 tests)
pytest tests/test_utils/ tests/test_data/ -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Expected Result

```
======================== 174 tests collected ========================
======================= 174 passed in ~30s ==========================
```

---

## Running Examples

```bash
# Order Management demo
python examples/order_management_example.py
```

---

## Docker Deployment

### Start Services

```bash
# Copy environment template
cp .env.example .env

# Edit credentials
nano .env  # or any editor

# Start Freqtrade + FreqUI
docker-compose up -d

# View logs
docker-compose logs -f freqtrade
```

### Access Dashboards

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| FreqUI | http://localhost:3000 | stoic_admin / StoicTrade2025!Secure |
| Jupyter | http://localhost:8888 | Token: JupyterStoic2025!Token |

---

## Project Structure

```
hft-algotrade-bot/
├── src/                      # Core modules
│   ├── order_manager/        # Order Management System
│   ├── ml/training/          # ML Pipeline
│   ├── risk/                 # Risk Management
│   └── utils/                # Utilities
├── user_data/strategies/     # Trading Strategies
├── tests/                    # Test Suite (174 tests)
├── examples/                 # Working Examples
├── docs/                     # Documentation
└── scripts/                  # Utility Scripts
```

---

## Key Features

### Order Management
- 5 order types (Market, Limit, StopLoss, TrailingStop, Bracket)
- Position tracking with real-time PnL
- Circuit breaker protection
- Slippage simulation

### ML Pipeline
- Feature engineering (50+ indicators)
- Model training (RF, XGBoost, LightGBM)
- Experiment tracking (W&B/MLflow)
- Model registry with versioning

### Risk Management
- Daily loss limits
- Consecutive loss protection
- Max drawdown limits
- Dynamic position sizing

---

## Backtesting

### Download Data

```bash
docker-compose run --rm freqtrade download-data \
    --pairs BTC/USDT ETH/USDT \
    --timerange 20240101-20241201 \
    -t 5m 1h
```

### Run Backtest

```bash
docker-compose run --rm freqtrade backtesting \
    --strategy StoicEnsembleStrategy \
    --timerange 20240601-20241201
```

---

## Configuration

### Paper Trading (Default)

```env
DRY_RUN=true
DRY_RUN_WALLET=10000
```

### Live Trading

```env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
DRY_RUN=false
```

> **Warning**: Start with small amounts! Test thoroughly in paper trading first.

---

## Troubleshooting

### Tests Fail

```bash
# Check Python version
python --version  # Should be 3.11+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Docker Issues

```bash
# Check status
docker-compose ps

# Restart services
docker-compose restart

# View logs
docker-compose logs -f freqtrade
```

---

## Documentation

- [Order Management API](docs/ORDER_MANAGEMENT.md)
- [ML Training Pipeline](docs/ML_TRAINING_PIPELINE.md)
- [Strategy Development](docs/STRATEGY_DEVELOPMENT_GUIDE.md)
- [Testing Guide](docs/TESTING_GUIDE.md)
- [Deployment](docs/deployment.md)

---

## Support

- **Issues**: https://github.com/kandibobe/hft-algotrade-bot/issues
- **Docs**: See `docs/` directory

---

**Stoic Citadel** v1.3.0 - Trade with wisdom, not emotion.
