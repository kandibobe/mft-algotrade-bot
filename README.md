<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Freqtrade-Powered-orange?style=for-the-badge" alt="Freqtrade">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <img src="https://github.com/kandibobe/hft-algotrade-bot/workflows/Stoic%20Citadel%20CI%2FCD/badge.svg" alt="CI/CD">
  <img src="https://img.shields.io/github/last-commit/kandibobe/hft-algotrade-bot?style=flat-square" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/kandibobe/hft-algotrade-bot?style=flat-square" alt="Issues">
  <img src="https://img.shields.io/github/stars/kandibobe/hft-algotrade-bot?style=flat-square" alt="Stars">
</p>

<h1 align="center">🏛️ Stoic Citadel</h1>

<p align="center">
  <strong>Professional HFT-lite Algorithmic Trading Infrastructure</strong>
  <br>
  <em>"In research, we seek truth. In trading, we execute truth."</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="docs/QUICKSTART_RU.md">Документация RU</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#️-architecture)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Strategy Development Workflow](#-strategy-development-workflow)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Monitoring & Observability](#-monitoring--observability)
- [Risk Management](#️-risk-management)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

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
│  │  • Prometheus + Grafana (Monitoring)      │             │
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
| **Prometheus** | Metrics collection | 9090 |
| **Grafana** | Dashboards & visualization | 3001 |
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
- 📊 **Full Observability** - Prometheus + Grafana monitoring

### Developer Experience
- ✅ **CI/CD Pipeline** - Automated testing on every push
- 🧪 **Comprehensive Tests** - Unit, integration, and strategy validation
- 🎨 **Code Quality** - Black, Flake8, MyPy pre-configured
- 📝 **Pre-commit Hooks** - Catch issues before commit

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
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot
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
make download
# OR
./scripts/citadel.sh download
```

### 5. Start Research Environment

```bash
# Launch Jupyter Lab
make research
```

Open your browser: `http://localhost:8888` (token: `stoic2024`)

### 6. Start Trading (Dry-Run)

```bash
# Start bot with fake money
make trade-dry
```

Access dashboard: `http://localhost:3000`

---

## 💻 Usage

### Makefile Commands (Recommended)

Stoic Citadel includes a comprehensive Makefile for streamlined development:

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

---

## 📁 Project Structure

```
stoic-citadel/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                      # CI/CD pipeline
│   ├── ISSUE_TEMPLATE/                 # Issue templates
│   ├── PULL_REQUEST_TEMPLATE.md        # PR template
│   └── dependabot.yml                  # Auto-updates
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
│   │   ├── StoicEnsembleStrategy.py    # Main strategy
│   │   └── StoicStrategyV1.py          # Alternative strategy
│   ├── data/                           # Historical data
│   └── logs/                           # Bot logs
│
├── tests/                              # Test suite
│   ├── conftest.py                     # Test fixtures
│   ├── test_strategies/                # Strategy tests
│   └── test_integration/               # Integration tests
│
├── monitoring/                         # Monitoring stack
│   ├── prometheus/                     # Metrics collection
│   ├── grafana/                        # Dashboards
│   └── alertmanager/                   # Alert management
│
├── scripts/
│   ├── citadel.sh                      # Master control script
│   ├── setup_wizard.py                 # Interactive setup
│   ├── health_check.py                 # System health check
│   ├── download_data.sh                # Data downloader
│   ├── verify_data.py                  # Data quality checker
│   └── walk_forward.py                 # Walk-forward validation
│
├── docs/                               # Documentation
│   ├── QUICKSTART_RU.md                # Quick start (Russian)
│   ├── API_SETUP_RU.md                 # API setup guide
│   └── TELEGRAM_SETUP_RU.md            # Telegram setup
│
├── Makefile                            # Build automation
├── pyproject.toml                      # Project config
├── .pre-commit-config.yaml             # Pre-commit hooks
├── docker-compose.yml                  # Main infrastructure
├── docker-compose.test.yml             # Test environment
├── docker-compose.monitoring.yml       # Monitoring stack
├── CONTRIBUTING.md                     # Contribution guide
├── SECURITY.md                         # Security policy
├── CODE_OF_CONDUCT.md                  # Code of conduct
├── LICENSE                             # MIT License
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
```

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

### Health Check

```bash
python3 scripts/health_check.py
```

---

## 🛡️ Risk Management

### Built-in Protections

| Protection | Purpose | Configuration |
|------------|---------|---------------|
| **Hard Stoploss** | Limit losses per trade | `stoploss: -0.05` |
| **Trailing Stop** | Lock in profits | `trailing_stop: true` |
| **Stoploss Guard** | Prevent revenge trading | Stop after 3 losses |
| **Max Drawdown** | Circuit breaker | Stop at 15% drawdown |
| **Cooldown Period** | Forced break | 2-4 hours after losses |

### Emergency Stop

```bash
# Stop all trading immediately
make stop
# OR
./scripts/citadel.sh stop
```

---

## 🚀 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed production deployment guide.

---

## 🔧 Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
make logs SERVICE=freqtrade

# Rebuild container
docker-compose build --no-cache
```

#### No Data Available

```bash
# Re-download data
make download

# Check data quality
python3 scripts/verify_data.py
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🔒 Security

For security concerns, please read [SECURITY.md](SECURITY.md).

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

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with discipline. Traded with wisdom. Executed with precision.</strong>
  <br><br>
  <em>"The wise trader knows that the best trade is often no trade at all."</em>
  <br><br>
  🏛️ <strong>Stoic Citadel</strong> - Where reason rules, not emotion.
</p>
