# 🏛️ Stoic Citadel - HFT Algorithmic Trading Bot

[![CI/CD](https://github.com/kandibobe/hft-algotrade-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/kandibobe/hft-algotrade-bot/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Freqtrade](https://img.shields.io/badge/freqtrade-2024.11-green.svg)](https://www.freqtrade.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"The wise man accepts losses with equanimity."*

Professional-grade algorithmic trading system built on Freqtrade with:
- 🎯 **Ensemble Strategies** - Multiple sub-strategies voting for signals
- 📊 **Regime Detection** - Adapts to market conditions automatically  
- ⚡ **Vectorized Indicators** - Fast, efficient technical analysis
- 🛡️ **Risk Management** - Position sizing, drawdown protection
- 🧪 **Walk-Forward Optimization** - Robust parameter selection

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Docker & Docker Compose
- Git

### 1. Clone & Setup

```bash
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot

# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for backtesting)
nano .env
```

### 2. Download Data

```bash
# Download historical data for backtesting
make -f Makefile.backtest download PAIRS="BTC/USDT ETH/USDT" TIMERANGE="20240101-20240601"
```

### 3. Run Backtest

```bash
# Run backtest with default strategy
make -f Makefile.backtest backtest

# Or with specific parameters
make -f Makefile.backtest backtest STRATEGY=StoicEnsembleStrategyV2 TIMERANGE="20240101-20240301"
```

### 4. View Results

```bash
# Generate HTML report
make -f Makefile.backtest report

# Open reports/backtest_report.html in browser
```

---

## 📁 Project Structure

```
hft-algotrade-bot/
├── src/                          # Core modules
│   ├── data/                     # Data loading & validation
│   ├── strategies/               # Strategy base classes
│   └── utils/                    # Indicators, risk, regime
├── user_data/                    # Freqtrade user data
│   ├── strategies/               # Trading strategies
│   ├── config/                   # Configuration files
│   └── data/                     # Historical data
├── config/                       # Strategy configs (YAML)
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── docker-compose.yml            # Production setup
└── docker-compose.backtest.yml   # Backtest environment
```

---

## 🛠️ Available Commands

### Backtesting

```bash
# Download data
make -f Makefile.backtest download

# Run backtest
make -f Makefile.backtest backtest

# Run hyperopt (parameter optimization)
make -f Makefile.backtest hyperopt EPOCHS=100

# Generate report
make -f Makefile.backtest report

# Full workflow (download + backtest + report)
make -f Makefile.backtest full

# Quick smoke test
make -f Makefile.backtest smoke
```

### Development

```bash
# Run tests
pytest tests/ -v

# Run only unit tests
pytest tests/test_utils/ tests/test_data/ -v

# Run integration tests
pytest tests/test_integration/ -v -m integration

# Format code
black src/ user_data/strategies/
isort src/ user_data/strategies/
```

### Docker

```bash
# Start trading (dry-run)
docker-compose up -d freqtrade frequi

# Start with monitoring
docker-compose --profile analytics up -d

# View logs
docker-compose logs -f freqtrade

# Stop all
docker-compose down
```

---

## 📊 Strategies

### StoicEnsembleStrategyV2 (Recommended)

Advanced ensemble strategy with regime detection:

| Feature | Description |
|---------|-------------|
| **Ensemble Voting** | Combines Momentum + Mean Reversion + Breakout |
| **Regime Aware** | Adapts risk based on market conditions |
| **Dynamic Sizing** | Position size based on volatility |
| **Time Filter** | Avoids low-liquidity hours |

**Default Parameters:**
- Stoploss: -5%
- Trailing Stop: +1% / 1.5% offset
- Max Positions: 3
- Risk per Trade: 2%

### Strategy Configuration

Edit `config/strategy_config.yaml`:

```yaml
# Risk Management
risk_per_trade: 0.02    # 2% risk per trade
max_positions: 3        # Max concurrent positions
stoploss: -0.05         # 5% stop loss

# Entry Parameters
rsi_oversold: 30        # RSI entry threshold
min_adx: 20             # Minimum trend strength

# Regime Detection
regime_aware: true      # Enable/disable regime adaptation
```

---

## 🧪 Walk-Forward Optimization

Run robust parameter optimization:

```bash
python scripts/walk_forward_optimization.py \
    --strategy StoicEnsembleStrategyV2 \
    --timerange 20230101-20240101 \
    --windows 6 \
    --epochs 100
```

This divides data into rolling windows:
1. **Train** (60%): Optimize parameters
2. **Validate** (20%): Verify performance
3. **Test** (20%): Out-of-sample check

Results saved to `reports/wfo/`

---

## 📊 Indicators Module

Vectorized technical indicators in `src/utils/indicators.py`:

```python
from src.utils.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_all_indicators
)

# Calculate single indicator
rsi = calculate_rsi(df['close'], period=14)

# Calculate all indicators at once
df = calculate_all_indicators(df)
```

**Available Indicators:**
- EMA, SMA (multiple periods)
- RSI, Stochastic
- MACD (with histogram)
- Bollinger Bands
- ATR, ADX
- VWAP, OBV

---

## ⚠️ Risk Management

Built-in protections in `src/utils/risk.py`:

```python
from src.utils.risk import (
    calculate_position_size_fixed_risk,
    calculate_max_drawdown,
    calculate_sharpe_ratio
)

# Fixed risk position sizing
position = calculate_position_size_fixed_risk(
    account_balance=10000,
    risk_per_trade=0.02,  # 2%
    entry_price=50000,
    stop_loss_price=47500
)
```

**Freqtrade Protections:**
- StoplossGuard: Pause after consecutive losses
- MaxDrawdown: Stop at maximum drawdown
- CooldownPeriod: Wait between trades

---

## 📈 Regime Detection

Automatic market regime detection in `src/utils/regime_detection.py`:

| Regime | Behavior |
|--------|----------|
| **Aggressive** | Higher risk, more positions |
| **Normal** | Standard parameters |
| **Cautious** | Reduced risk |
| **Defensive** | Minimal exposure |

```python
from src.utils.regime_detection import (
    calculate_regime_score,
    get_regime_parameters
)

# Get current regime
regime_data = calculate_regime_score(high, low, close, volume)
score = regime_data['regime_score'].iloc[-1]

# Get adjusted parameters
params = get_regime_parameters(score)
print(f"Mode: {params['mode']}, Risk: {params['risk_per_trade']}")
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_utils/test_indicators.py -v
```

**Test Structure:**
- `tests/test_utils/` - Unit tests for indicators, risk
- `tests/test_data/` - Data loading tests
- `tests/test_strategies/` - Strategy tests
- `tests/test_integration/` - End-to-end tests

---

## 🔧 Configuration

### Environment Variables

```bash
# .env file
FREQTRADE_API_USERNAME=admin
FREQTRADE_API_PASSWORD=your_secure_password
JUPYTER_TOKEN=your_jupyter_token
POSTGRES_PASSWORD=your_db_password

# Strategy selection
STRATEGY=StoicEnsembleStrategyV2
```

### Backtest Config

Edit `user_data/config/config_backtest.json`:

```json
{
    "max_open_trades": 3,
    "stake_amount": "unlimited",
    "dry_run_wallet": 10000,
    "fee": 0.001
}
```

---

## 📖 Documentation

- [Development Plan](DEVELOPMENT_PLAN.md) - Roadmap and phases
- [Architecture](ARCHITECTURE_ANALYSIS.md) - System design
- [Deployment](DEPLOYMENT.md) - Production setup
- [Quick Start (Windows)](QUICKSTART_WINDOWS.md) - Windows guide

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest tests/ -v`
4. Commit: `git commit -m 'feat: add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## ⚠️ Disclaimer

**This software is for educational purposes only.**

- Trading cryptocurrencies involves substantial risk
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- Always test thoroughly in dry-run mode first

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>🏛️ Stoic Citadel</b><br>
  <i>"The wise man adapts to circumstances like water."</i>
</p>
