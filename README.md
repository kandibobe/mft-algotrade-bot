# Stoic Citadel 🏛️

**Professional HFT-lite Algorithmic Trading Infrastructure**

> *"In research, we seek truth. In trading, we execute truth."*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Strategy Development Workflow](#strategy-development-workflow)
- [Risk Management](#risk-management)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Overview

Stoic Citadel is a professional-grade algorithmic trading ecosystem designed for serious traders who understand that **profitability comes from research, not guesswork**.

Unlike typical trading bots that execute random strategies, Stoic Citadel separates:
- **Research Lab** (Jupyter + VectorBT) - Where you discover edge
- **Execution Engine** (Freqtrade) - Where you deploy proven strategies

### Philosophy

1. **Research First** - Find strategies in the lab, not in production
2. **Risk Management** - Capital preservation > profit maximization
3. **Automation** - Let the machine execute, let the human research
4. **Discipline** - No revenge trading, no emotional decisions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STOIC CITADEL                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────┐     ┌─────────────────────┐        │
│  │  RESEARCH LAB      │     │  EXECUTION ENGINE   │        │
│  │  ─────────────     │     │  ────────────────   │        │
│  │  • Jupyter Lab     │ ──► │  • Freqtrade        │        │
│  │  • VectorBT        │     │  • FreqUI           │        │
│  │  • ML Models       │     │  • WebSocket API    │        │
│  │  • Backtesting     │     │  • Order Execution  │        │
│  └────────────────────┘     └─────────────────────┘        │
│           │                           │                     │
│           └───────────┬───────────────┘                     │
│                       │                                     │
│  ┌────────────────────▼──────────────────────┐             │
│  │          INFRASTRUCTURE                   │             │
│  │          ──────────────                   │             │
│  │  • PostgreSQL (Analytics DB)              │             │
│  │  • Telegram Bot (Alerts)                  │             │
│  │  • Portainer (Container Management)       │             │
│  └───────────────────────────────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose | Port |
|-----------|---------|------|
| **Freqtrade** | Trading bot execution engine | 8080 |
| **FreqUI** | Web dashboard for monitoring | 3000 |
| **Jupyter Lab** | Research environment | 8888 |
| **PostgreSQL** | Trade analytics database | 5432 |
| **Portainer** | Docker management UI | 9000 |

---

## ✨ Features

### Research Lab
- 🔬 **VectorBT Integration** - Backtest years of data in seconds
- 📊 **Comprehensive Indicators** - 50+ technical indicators pre-configured
- 🤖 **ML Pipeline** - XGBoost, LightGBM, CatBoost ready to use
- 📈 **Advanced Visualization** - Plotly-based interactive charts
- 🧪 **Parameter Optimization** - Grid search with heatmaps

### Execution Engine
- ⚡ **Low Latency** - Optimized for sub-second execution
- 🔒 **Risk Management** - Hard stops, cooldowns, max drawdown protection
- 📱 **Telegram Alerts** - Real-time notifications
- 🌐 **Multi-Exchange** - Binance, Bybit, and more
- 💾 **Database Logging** - Full trade history in PostgreSQL

### Infrastructure
- 🐳 **Fully Dockerized** - Portable across any system
- 🔐 **Security First** - API keys encrypted, no plaintext secrets
- 📦 **One-Command Deploy** - Setup in minutes, not hours
- 🛡️ **Production Ready** - Designed for 24/7 operation

---

## 📦 Prerequisites

- **Docker** (>= 20.10)
- **Docker Compose** (>= 2.0)
- **Git**
- **8GB RAM** (minimum)
- **20GB Disk Space** (for data storage)

### Optional (for VPS deployment)
- **Hetzner Cloud Account** (or any VPS provider)
- **Domain name** (for HTTPS access)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stoic-citadel.git
cd stoic-citadel
```

### 2. Initial Setup (Recommended - Interactive Wizard)

```bash
# Run interactive setup wizard
make setup
# OR
python3 scripts/setup_wizard.py
```

**Alternative - Manual Setup:**

```bash
# Make control script executable
chmod +x scripts/citadel.sh scripts/download_data.sh
chmod +x scripts/verify_data.py

# Run first-time setup
./scripts/citadel.sh setup
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys (optional for dry-run)
nano .env
```

### 4. Download Historical Data

```bash
# Download 90 days of 5-minute candles
./scripts/citadel.sh download
```

### 5. Start Research Environment

```bash
# Launch Jupyter Lab
./scripts/citadel.sh research
```

Open your browser: `http://localhost:8888` (token: `stoic2024`)

### 6. Start Trading (Dry-Run)

```bash
# Start bot with fake money
./scripts/citadel.sh trade
```

Access dashboard: `http://localhost:3000`

---

## 💻 Usage

### Makefile Commands (Recommended)

Stoic Citadel now includes a comprehensive Makefile for streamlined development:

```bash
make help  # Show all available commands
```

**Common Commands:**

| Command | Description |
|---------|-------------|
| `make setup` | Run interactive setup wizard |
| `make start` | Start all services |
| `make stop` | Stop all services |
| `make test` | Run full test suite |
| `make lint` | Check code quality |
| `make format` | Auto-format code |
| `make trade-dry` | Start paper trading |
| `make backtest STRATEGY=MyStrategy` | Run backtest |
| `make research` | Start Jupyter Lab |
| `make monitoring` | Start monitoring stack |
| `make logs SERVICE=freqtrade` | View logs |
| `make clean` | Remove containers |

### Master Control Script (Alternative)

All operations can also be managed through `citadel.sh`:

```bash
./scripts/citadel.sh [command]
```

#### Available Commands

| Command | Description |
|---------|-------------|
| `setup` | First-time setup (builds containers) |
| `start` | Start all services |
| `stop` | Stop all services |
| `restart` | Restart all services |
| `logs [service]` | View logs (default: freqtrade) |
| `status` | Show service status |
| `research` | Launch Jupyter Lab |
| `trade` | Start trading bot (dry-run) |
| `trade-live` | Start LIVE trading ⚠️ |
| `backtest [strategy]` | Run backtest |
| `download` | Download historical data |
| `verify` | Verify data quality |
| `clean` | Remove containers and volumes |

#### Examples

```bash
# Setup environment
./scripts/citadel.sh setup

# Download data
./scripts/citadel.sh download

# Start research
./scripts/citadel.sh research

# Run backtest
./scripts/citadel.sh backtest StoicEnsembleStrategy

# View logs
./scripts/citadel.sh logs freqtrade

# Start dry-run trading
./scripts/citadel.sh trade
```

### Manual Docker Commands

If you prefer direct Docker control:

```bash
# Build containers
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f freqtrade

# Stop all services
docker-compose down

# Enter Jupyter container
docker-compose exec jupyter bash

# Run backtest
docker-compose run --rm freqtrade backtesting \
  --strategy StoicEnsembleStrategy \
  --timerange 20240101-
```

---

## 📁 Project Structure

```
stoic-citadel/
├── .github/
│   └── workflows/
│       └── ci.yml                      # CI/CD pipeline
│
├── docker/
│   ├── Dockerfile.jupyter              # Research environment
│   ├── Dockerfile.test                 # Test container
│   └── requirements-research.txt       # Python dependencies
│
├── user_data/
│   ├── config/
│   │   ├── config_production.json      # Production config
│   │   └── config_dryrun.json          # Testing config
│   ├── strategies/
│   │   └── StoicEnsembleStrategy.py    # Template strategy
│   ├── data/                           # Historical data
│   ├── logs/                           # Bot logs
│   └── notebooks/                      # Saved notebooks
│
├── tests/                              # ⭐ NEW: Test suite
│   ├── conftest.py                     # Test fixtures
│   ├── test_strategies/                # Strategy tests
│   │   ├── test_indicators.py
│   │   └── test_stoic_ensemble.py
│   └── test_integration/               # Integration tests
│       └── test_trading_flow.py
│
├── monitoring/                         # ⭐ NEW: Monitoring stack
│   ├── prometheus/                     # Metrics collection
│   ├── grafana/                        # Dashboards
│   └── alertmanager/                   # Alert management
│
├── research/
│   └── 01_research_template.ipynb      # Research notebook template
│
├── scripts/
│   ├── citadel.sh                      # Master control script
│   ├── setup_wizard.py                 # ⭐ NEW: Interactive setup
│   ├── download_data.sh                # Data downloader
│   ├── verify_data.py                  # Data quality checker
│   ├── validate_config.py              # Config validator
│   └── walk_forward.py                 # Walk-forward validation
│
├── Makefile                            # ⭐ NEW: Build automation
├── pyproject.toml                      # ⭐ NEW: Project config
├── .pre-commit-config.yaml             # ⭐ NEW: Pre-commit hooks
├── docker-compose.yml                  # Infrastructure definition
├── docker-compose.test.yml             # ⭐ NEW: Test environment
├── docker-compose.monitoring.yml       # ⭐ NEW: Monitoring stack
├── .env.example                        # Environment template
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## ⚙️ Configuration

### Exchange Configuration

Edit `user_data/config/config_production.json`:

```json
{
  "exchange": {
    "name": "binance",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET",
    "ccxt_config": {
      "enableRateLimit": true
    }
  }
}
```

### Telegram Alerts

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Update `.env`:

```env
TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

4. Enable in config:

```json
{
  "telegram": {
    "enabled": true,
    "token": "${TELEGRAM_TOKEN}",
    "chat_id": "${TELEGRAM_CHAT_ID}"
  }
}
```

### Risk Management Settings

Critical settings in `config_production.json`:

```json
{
  "max_open_trades": 3,
  "stake_amount": "unlimited",
  "tradable_balance_ratio": 0.99,

  "stoploss": -0.05,
  "trailing_stop": true,
  "trailing_stop_positive": 0.01,

  "protections": [
    {
      "method": "StoplossGuard",
      "trade_limit": 3,
      "stop_duration_candles": 24
    },
    {
      "method": "MaxDrawdown",
      "max_allowed_drawdown": 0.15
    }
  ]
}
```

---

## 🔬 Strategy Development Workflow

### The Stoic Method

1. **Research Phase** (Jupyter Lab)
   - Load historical data
   - Calculate indicators
   - Generate signals
   - Backtest with VectorBT
   - Optimize parameters
   - Validate with walk-forward testing

2. **Implementation Phase** (Freqtrade)
   - Convert logic to Freqtrade strategy
   - Backtest with Freqtrade
   - Paper trade (dry-run)
   - Monitor for 1-2 weeks

3. **Deployment Phase** (Production)
   - Small capital allocation
   - Monitor closely
   - Scale up gradually

### Example Research Workflow

Open `research/01_research_template.ipynb` in Jupyter Lab and follow the guided workflow.

---

## 🧪 Testing & Quality Assurance

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run with coverage report
make test-coverage
```

### Code Quality

```bash
# Check code quality
make lint

# Auto-format code
make format

# Run pre-commit hooks
make pre-commit
```

### Test Structure

- **Unit Tests**: `tests/test_strategies/` - Test individual components
- **Integration Tests**: `tests/test_integration/` - Test complete workflows
- **Fixtures**: `tests/conftest.py` - Reusable test data and mocks

### Continuous Integration

Every push and PR automatically runs:
- ✅ Code formatting checks (Black)
- ✅ Linting (Flake8)
- ✅ Type checking (MyPy)
- ✅ Security scanning (Bandit)
- ✅ Unit tests
- ✅ Integration tests
- ✅ Docker build validation
- ✅ Strategy validation
- ✅ Configuration validation

View CI/CD status in `.github/workflows/ci.yml`

---

## 📊 Monitoring & Observability

### Starting the Monitoring Stack

```bash
# Start Prometheus + Grafana
make monitoring

# Stop monitoring
make monitoring-stop
```

### Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3001 | admin/admin |
| **Prometheus** | http://localhost:9090 | - |
| **Alertmanager** | http://localhost:9093 | - |

### Pre-built Dashboards

- **Trading Overview** - P&L, win rate, open trades, drawdown
- **System Metrics** - CPU, memory, disk usage
- **Container Metrics** - Docker resource usage
- **Custom Metrics** - Add your own!

### Setting Up Alerts

Edit `monitoring/alertmanager/config.yml` to configure:
- Telegram notifications
- Email alerts
- Webhook integrations

---

## 🛡️ Risk Management

### The Stoic Guard (Built-in Protections)

| Protection | Purpose | Configuration |
|------------|---------|---------------|
| **Hard Stoploss** | Limit losses per trade | `stoploss: -0.05` |
| **Trailing Stop** | Lock in profits | `trailing_stop: true` |
| **Stoploss Guard** | Prevent revenge trading | Stop after 3 losses |
| **Max Drawdown** | Circuit breaker | Stop at 15% drawdown |
| **Cooldown Period** | Forced break | 2-4 hours after losses |
| **Position Sizing** | Volatility-adjusted | Based on ATR |

### Emergency Procedures

#### Panic Button (Immediate Stop)

```bash
# Stop all trading immediately
./scripts/citadel.sh stop

# Or force kill all positions
docker-compose down
```

---

## 🚀 Deployment

### Local Development

Already covered in [Quick Start](#quick-start).

### VPS Deployment (Production)

See detailed deployment guide in the README for production setup on Hetzner Cloud or any VPS provider.

---

## 🔧 Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
./scripts/citadel.sh logs [service]

# Rebuild container
docker-compose build --no-cache [service]
```

#### No Data Available

```bash
# Re-download data
./scripts/citadel.sh download

# Check data quality
./scripts/citadel.sh verify
```

#### Strategy Not Loading

```bash
# List available strategies
docker-compose run --rm freqtrade list-strategies

# Test strategy
./scripts/citadel.sh backtest StoicEnsembleStrategy
```

---

## ⚠️ Disclaimer

**IMPORTANT LEGAL NOTICE:**

- This software is for **educational purposes only**
- Trading cryptocurrencies carries **significant risk**
- **Past performance does not guarantee future results**
- You can **lose all your capital**
- The authors are **not responsible for your trading losses**
- **Always test extensively** in dry-run mode first
- **Never invest more than you can afford to lose**

By using this software, you acknowledge that you understand these risks.

---

## 📄 License

MIT License

---

**Built with discipline. Traded with wisdom. Executed with precision.**

*"The wise trader knows that the best trade is often no trade at all."*

🏛️ **Stoic Citadel** - Where reason rules, not emotion.
