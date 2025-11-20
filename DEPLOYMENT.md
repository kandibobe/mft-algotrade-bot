# Stoic Citadel - Production Deployment Guide

**Version:** 1.0.0
**Date:** 2025-11-20
**Status:** Production Ready

---

## Overview

This guide provides step-by-step instructions for deploying Stoic Citadel, a professional algorithmic trading infrastructure with walk-forward validation to prevent overfitting.

### Key Features

- **Market Regime Filter**: BTC/USDT 1d trend analysis
- **Walk-Forward Optimization**: Train/test split to prevent overfitting
- **Security Hardened**: All services bound to localhost (127.0.0.1)
- **HyperOpt Integration**: Automated parameter optimization
- **Production Ready**: Designed for 24/7 operation

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 22.04 LTS recommended) or macOS
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 50GB free space
- **CPU**: 4 cores minimum

### Software Requirements

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **Python** >= 3.9 (for local scripts)
- **Git**

### Installation Commands

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose python3 python3-pip git

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

---

## Deployment Steps

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/stoic-citadel.git
cd stoic-citadel

# Make scripts executable
chmod +x scripts/*.sh scripts/*.py
```

### Step 2: Initial Setup

Run the automated setup script:

```bash
./scripts/deploy.sh --setup
```

This will:
1. Create directory structure
2. Pull Docker images (Freqtrade, FreqUI)
3. Build Research Lab container (Jupyter with VectorBT)
4. Set file permissions

**Expected output:**
```
✅ Docker and Docker Compose are installed
✅ Docker permissions OK
✅ Directories created
✅ Permissions set
✅ Docker images pulled
✅ Research Lab built
✅ Setup completed successfully!
```

### Step 3: Download Historical Data

Download 2 years of data for backtesting and walk-forward validation:

```bash
./scripts/deploy.sh --data
```

This downloads:
- **BTC/USDT** (5m, 1d) - 2 years
- **ETH/USDT** (5m) - 2 years
- **BNB/USDT** (5m) - 2 years
- **SOL/USDT** (5m) - 2 years

**Duration**: 10-20 minutes depending on connection speed.

**Data location**: `./user_data/data/binance/`

### Step 4: Launch Research Environment

```bash
./scripts/deploy.sh --research
```

**Access**: http://127.0.0.1:8888 (token: `stoic2024`)

**Security Note**: Jupyter is bound to localhost only. To access from a remote machine, use SSH tunnel:

```bash
# From your local machine
ssh -L 8888:localhost:8888 user@your-server
```

### Step 5: Develop and Refine Strategy

#### 5.1 Open Jupyter Lab

Navigate to: http://127.0.0.1:8888

#### 5.2 Review Strategy

Open: `strategies/StoicStrategyV1.py`

Key components:
- **Market Regime Filter**: BTC/USDT 1d EMA200
- **Entry Logic**: RSI oversold + trend confirmation
- **Exit Logic**: RSI overbought or trend reversal
- **Risk Management**: -5% hard stop, scalping ROI

#### 5.3 Quick Backtest

```bash
# Run simple backtest
docker-compose run --rm freqtrade backtesting \
  --strategy StoicStrategyV1 \
  --config /freqtrade/user_data/config/config_production.json \
  --timerange 20230101-20231231
```

### Step 6: Walk-Forward Validation

**CRITICAL**: This step prevents overfitting by validating on out-of-sample data.

```bash
./scripts/deploy.sh --validate
```

**Interactive prompts:**
- Strategy name: `StoicStrategyV1`
- Train months: `3` (default)
- Test months: `1` (default)
- HyperOpt epochs: `100` (default)

**What happens:**
1. Data is split into train/test windows
2. HyperOpt runs on train data (3 months)
3. Best parameters tested on unseen test data (1 month)
4. Process repeats for multiple windows
5. Strategy passes if 70%+ of windows meet criteria

**Duration**: 2-8 hours depending on data and epochs.

**Success criteria:**
- Sharpe Ratio > 1.5
- Max Drawdown < 20%
- Win Rate > 50%
- Minimum 5 trades per window

**Output:**
```
✅ STRATEGY APPROVED
Strategy shows consistent performance across time periods.
```

or

```
❌ STRATEGY REJECTED
Strategy fails on out-of-sample data. High overfitting risk.
```

### Step 7: Paper Trading (Dry-Run)

If validation passes:

```bash
./scripts/citadel.sh trade
```

**Access dashboard**: http://127.0.0.1:3000

**Monitor for 1-2 weeks** before considering live trading.

### Step 8: Live Trading (Optional)

⚠️ **EXTREME CAUTION REQUIRED**

1. **Add API Keys**

Edit `.env`:
```env
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

2. **Update Config**

Edit `user_data/config/config_production.json`:
```json
{
  "dry_run": false,
  "exchange": {
    "name": "binance",
    "key": "${BINANCE_API_KEY}",
    "secret": "${BINANCE_API_SECRET}"
  }
}
```

3. **Start with Small Capital**

Initial allocation: $100-500 maximum

4. **Enable Telegram Alerts**

See README.md for Telegram bot setup.

5. **Launch**

```bash
./scripts/citadel.sh trade-live
```

---

## Architecture

### Container Structure

```
┌─────────────────────────────────────────────┐
│  DOCKER COMPOSE                             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │  Freqtrade   │  │    FreqUI    │        │
│  │  (Trading)   │  │ (Dashboard)  │        │
│  │              │  │              │        │
│  │  127.0.0.1:  │  │  127.0.0.1:  │        │
│  │    8080      │  │    3000      │        │
│  └──────────────┘  └──────────────┘        │
│                                             │
│  ┌──────────────┐                          │
│  │   Jupyter    │                          │
│  │  (Research)  │                          │
│  │              │                          │
│  │  127.0.0.1:  │                          │
│  │    8888      │                          │
│  └──────────────┘                          │
│                                             │
└─────────────────────────────────────────────┘
```

### Volume Mounts

| Container   | Mount                          | Access    |
|-------------|--------------------------------|-----------|
| freqtrade   | ./user_data                    | read-write|
| freqtrade   | ./user_data/strategies         | read-write|
| jupyter     | ./user_data                    | **read-only** |
| jupyter     | ./user_data/strategies         | read-write|
| jupyter     | ./research                     | read-write|

**Security**: Jupyter has read-only access to data to prevent accidental modifications.

---

## File Structure

```
stoic_citadel/
│
├── docker-compose.yml              # Infrastructure definition
│
├── user_data/
│   ├── config/
│   │   ├── config_production.json  # Production config
│   │   └── config_dryrun.json      # Paper trading config
│   │
│   ├── strategies/
│   │   ├── StoicStrategyV1.py      # Main strategy (v1)
│   │   └── StoicEnsembleStrategy.py # Template strategy
│   │
│   ├── data/                       # Historical OHLCV data
│   ├── logs/                       # Bot logs
│   └── hyperopt_results/           # HyperOpt outputs
│
├── research/
│   └── 01_research_template.ipynb  # Jupyter notebook template
│
├── scripts/
│   ├── deploy.sh                   # Deployment automation
│   ├── citadel.sh                  # Master control script
│   ├── walk_forward.py             # Walk-forward validator
│   ├── download_data.sh            # Data downloader
│   └── verify_data.py              # Data quality checker
│
└── DEPLOYMENT.md                   # This file
```

---

## Strategy Configuration

### StoicStrategyV1 Parameters

#### HyperOptimizable Parameters

```python
# Entry
buy_rsi_threshold = Integer(20, 40, default=30)

# Exit
sell_rsi_threshold = Integer(60, 80, default=70)

# Regime
regime_ema_period = Integer(150, 250, default=200)
```

#### Risk Management

```python
# Hard stop loss
stoploss = -0.05  # -5%

# ROI (scalping profile)
minimal_roi = {
    "0": 0.06,   # 6% immediate
    "20": 0.04,  # 4% after 100 min
    "40": 0.02,  # 2% after 200 min
    "60": 0.01   # 1% after 300 min
}
```

#### Market Regime Filter

```python
# Only trade when BTC > EMA200 on 1d
regime_bull = (dataframe['close'] > dataframe['ema_regime'])
```

---

## Walk-Forward Optimization

### Methodology

1. **Split Data**: Divide historical data into overlapping train/test windows
2. **Optimize**: Run HyperOpt on training window (3 months)
3. **Validate**: Test best parameters on unseen test window (1 month)
4. **Roll Forward**: Move window forward and repeat
5. **Aggregate**: Strategy must pass 70%+ of windows

### Example Timeline

```
Window 1:  [Train: Jan-Mar] [Test: Apr]
Window 2:  [Train: Feb-Apr] [Test: May]
Window 3:  [Train: Mar-May] [Test: Jun]
...
```

### Acceptance Criteria

| Metric           | Threshold | Reason                          |
|------------------|-----------|---------------------------------|
| Sharpe Ratio     | > 1.5     | Risk-adjusted returns           |
| Max Drawdown     | < 20%     | Capital preservation            |
| Win Rate         | > 50%     | Consistency                     |
| Total Trades     | >= 5      | Statistical significance        |
| Pass Rate        | >= 70%    | Robustness across time periods  |

---

## Security Best Practices

### 1. Port Binding

All services are bound to **127.0.0.1** (localhost only):

```yaml
ports:
  - "127.0.0.1:8080:8080"  # Freqtrade API
  - "127.0.0.1:3000:8080"  # FreqUI
  - "127.0.0.1:8888:8888"  # Jupyter Lab
```

**Remote access**: Use SSH tunneling, not public binding.

### 2. API Keys

Never commit API keys to git:

```bash
# .gitignore includes:
.env
*.env
user_data/config/config_production.json  # If it contains keys
```

### 3. Firewall Rules

```bash
# Allow only SSH (if on VPS)
sudo ufw allow 22/tcp
sudo ufw enable
```

### 4. Read-Only Data Access

Jupyter has read-only access to prevent accidental data corruption:

```yaml
volumes:
  - ./user_data:/home/jovyan/user_data:ro  # Read-only
```

---

## Monitoring

### Real-Time Monitoring

```bash
# View bot logs
./scripts/citadel.sh logs freqtrade

# Check service status
./scripts/citadel.sh status

# Access dashboard
http://127.0.0.1:3000
```

### Telegram Alerts

1. Create bot with [@BotFather](https://t.me/botfather)
2. Get chat ID from [@userinfobot](https://t.me/userinfobot)
3. Update `.env`:

```env
TELEGRAM_TOKEN=123456789:ABCdef...
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

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs freqtrade

# Rebuild
docker-compose build --no-cache

# Restart
docker-compose restart
```

### Data Download Fails

```bash
# Check exchange connectivity
docker-compose run --rm freqtrade list-exchanges

# Manual download
docker-compose run --rm freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT \
  --timeframe 5m \
  --days 30
```

### HyperOpt Timeout

Reduce epochs or train window:

```bash
python3 scripts/walk_forward.py \
  --strategy StoicStrategyV1 \
  --epochs 50 \
  --train-months 2
```

### Strategy Not Loading

```bash
# List available strategies
docker-compose run --rm freqtrade list-strategies

# Validate syntax
python3 -m py_compile user_data/strategies/StoicStrategyV1.py
```

---

## Performance Tuning

### For Faster Backtests

1. Use **VectorBT** in Jupyter for rapid iteration
2. Reduce timeframe (e.g., 15m instead of 5m for initial tests)
3. Limit pairs (test on BTC/USDT first)

### For Lower Memory Usage

```yaml
# docker-compose.yml
services:
  freqtrade:
    mem_limit: 2g
  jupyter:
    mem_limit: 4g
```

---

## Maintenance

### Daily

- Check bot logs: `./scripts/citadel.sh logs`
- Review dashboard: http://127.0.0.1:3000
- Monitor Telegram alerts

### Weekly

- Verify data quality: `python3 scripts/verify_data.py`
- Download latest data: `./scripts/deploy.sh --data`
- Review trade performance

### Monthly

- Re-run walk-forward validation: `./scripts/deploy.sh --validate`
- Update strategy if market conditions change
- Backup database: `cp user_data/tradesv3.sqlite backups/`

---

## Emergency Procedures

### Panic Button (Immediate Stop)

```bash
# Stop all trading
./scripts/citadel.sh stop

# Or force kill
docker-compose down
```

### Check Open Positions

```bash
# Via API
curl http://127.0.0.1:8080/api/v1/status

# Via dashboard
http://127.0.0.1:3000
```

### Manual Exit

```bash
# Force sell all positions
docker-compose exec freqtrade freqtrade forceexit all
```

---

## Disclaimer

⚠️ **CRITICAL WARNING**

- Trading cryptocurrencies involves **substantial risk of loss**
- **Past performance does not guarantee future results**
- Walk-forward validation reduces but **does not eliminate** overfitting risk
- Always test extensively in paper trading before live deployment
- **Never invest more than you can afford to lose**
- The authors are **not responsible** for any financial losses

By deploying this system, you acknowledge these risks.

---

## Support

- **Issues**: https://github.com/yourusername/stoic-citadel/issues
- **Documentation**: README.md
- **Strategy Guide**: User Guide (coming soon)

---

**Built with discipline. Deployed with precision. Traded with wisdom.**

*"The best trade is often the one you don't make."*

🏛️ **Stoic Citadel** - Production Ready
