# 🏛️ Stoic Citadel - Algorithmic Trading Bot

[![CI/CD](https://github.com/kandibobe/stoic-citadel/actions/workflows/ci.yml/badge.svg)](https://github.com/kandibobe/stoic-citadel/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Freqtrade](https://img.shields.io/badge/freqtrade-2024.11-green.svg)](https://www.freqtrade.io/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

> *"The wise man accepts losses with equanimity."*

Professional-grade algorithmic trading system built on Freqtrade with advanced order management, ML pipeline, and production-ready risk management.

## ⚠️ IMPORTANT NOTICE

**This software is PROPRIETARY and CONFIDENTIAL.**

- ❌ **NO unauthorized copying, modification, or distribution**
- ❌ **NO commercial use without explicit permission**
- ❌ **Trading strategies and ML models are trade secrets**
- ✅ **Personal use only by authorized individuals**

See [LICENSE](LICENSE) for full terms. Unauthorized use is strictly prohibited.

**Trading Risk Warning:** This software is for educational purposes. Cryptocurrency trading carries significant risk. You can lose all your capital. Never invest more than you can afford to lose.

## ✨ Key Features

### 🎯 Trading Features
- **Ensemble Strategies** - Multiple sub-strategies voting for robust signals
- **Regime Detection** - Adapts to bull/bear/sideways markets automatically
- **Advanced Order Management** - 5 order types with state machine
- **Circuit Breaker** - Automatic risk protection against catastrophic losses
- **Position Management** - Real-time PnL tracking with stop-loss/take-profit

### 🤖 ML Pipeline (MLOps)
- **Feature Engineering** - 50+ technical indicators with stationarity (log returns)
- **Triple Barrier Labeling** - Proper ML labels accounting for fees and holding period
- **Model Training** - Random Forest, XGBoost, LightGBM with hyperparameter optimization
- **Experiment Tracking** - W&B / MLflow integration
- **Model Registry** - Version management with production promotion

### 🛡️ Production Ready
- **Risk Management** - Position sizing, drawdown protection, daily loss limits
- **Slippage Simulation** - Realistic execution modeling for backtests
- **Comprehensive Testing** - 25+ unit tests
- **Full Documentation** - API docs, guides, examples

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Git
- Python 3.11+ (for local development)

### 1. Clone Repository

```bash
git clone https://github.com/kandibobe/stoic-citadel.git
cd stoic-citadel
```

### 2. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit credentials (see CREDENTIALS.md for details)
nano .env
```

### 3. Start Services

```bash
# Start all services (Freqtrade, FreqUI, Jupyter, PostgreSQL)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f freqtrade
```

### 4. Access Dashboards

- **FreqUI**: http://localhost:3000
  - Login: `stoic_admin`
  - Password: See `CREDENTIALS.md`

- **Jupyter Lab**: http://localhost:8888
  - Token: See `CREDENTIALS.md`

- **Portainer**: http://localhost:9000

---

## 📁 Project Structure

```
stoic-citadel/
├── src/                              # Core modules
│   ├── order_manager/                # ✅ Order Management System
│   │   ├── order_types.py            # 5 order types with state machine
│   │   ├── position_manager.py       # Position tracking & PnL
│   │   ├── circuit_breaker.py        # Risk protection
│   │   ├── slippage_simulator.py     # Execution simulation
│   │   └── smart_limit_executor.py   # Smart limit orders (fee optimization)
│   ├── ml/                           # ML modules
│   │   ├── training/                 # ✅ ML Training Pipeline
│   │   │   ├── feature_engineering.py    # Feature pipeline (no data leakage)
│   │   │   ├── labeling.py               # Triple Barrier labeling
│   │   │   ├── feature_selection.py      # SHAP/Permutation importance
│   │   │   ├── model_trainer.py          # Model training
│   │   │   ├── experiment_tracker.py     # Experiment tracking
│   │   │   └── model_registry.py         # Model versioning
│   │   └── inference_service.py      # ML inference
│   ├── strategies/                   # Strategy base classes
│   │   └── market_regime.py          # Market regime filter
│   ├── risk/                         # Risk management
│   │   ├── position_sizing.py        # Dynamic/ATR-based sizing
│   │   ├── circuit_breaker.py        # Circuit breaker
│   │   └── correlation.py            # Correlation/drawdown monitor
│   ├── data/                         # Data loading & validation
│   └── utils/                        # Indicators, risk, regime
├── user_data/                        # Freqtrade user data
│   ├── strategies/                   # Trading strategies
│   ├── config/                       # Configuration files
│   ├── data/                         # Historical data
│   └── models/                       # Trained ML models
├── tests/                            # Test suite (190+ tests)
├── scripts/                          # Utility scripts
│   └── optimize_strategy.py          # Hyperopt with Optuna
├── docs/                             # Documentation
├── Makefile                          # Build commands
├── docker-compose.yml                # Production setup
└── README.md                         # This file
```

---

## 📚 Documentation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
- **[CREDENTIALS.md](CREDENTIALS.md)** - All access credentials and passwords
- **[TESTING.md](TESTING.md)** - How to run tests and validate

### API Documentation
- **[docs/ORDER_MANAGEMENT.md](docs/ORDER_MANAGEMENT.md)** - Order Management System API
- **[docs/ML_TRAINING_PIPELINE.md](docs/ML_TRAINING_PIPELINE.md)** - ML Pipeline API
- **[docs/STRATEGY_DEVELOPMENT_GUIDE.md](docs/STRATEGY_DEVELOPMENT_GUIDE.md)** - Strategy development

---

## 🛠️ Usage Examples

### Smart Limit Order Execution

```python
from src.order_manager import (
    SmartLimitExecutor,
    SmartLimitConfig,
    LimitOrder,
    OrderSide,
    CircuitBreaker,
)

# Smart Limit Orders save fees: Maker (~0.02%) vs Taker (~0.1%)
config = SmartLimitConfig(
    initial_offset_bps=2.0,       # Start 2 bps inside spread
    max_offset_bps=10.0,          # Max chase distance
    chase_interval_seconds=5.0,   # Check every 5 seconds
    max_wait_seconds=60.0,        # Max wait time
    convert_to_market=True,       # Fallback to market if timeout
)

executor = SmartLimitExecutor(config)
circuit_breaker = CircuitBreaker()

# Check if can trade
if not circuit_breaker.is_operational:
    print("Trading halted by circuit breaker!")
    exit()

# Create limit order
order = LimitOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    quantity=0.1,
    price=50000.0
)

# Get current orderbook
orderbook = exchange.fetch_order_book("BTC/USDT")

# Execute with smart limit (state machine handles chasing)
result = executor.execute(order, orderbook)

if result.success:
    print(f"Filled at {result.average_price:.2f}")
    print(f"Fee saved vs market: ${result.fee_saved_vs_market:.2f}")
    print(f"Execution type: {result.execution_type}")  # 'maker' or 'taker'
```

### ML Training Pipeline with Triple Barrier Labeling

```python
from src.ml.training import (
    FeatureEngineer,
    FeatureConfig,
    TripleBarrierLabeler,
    TripleBarrierConfig,
    ModelTrainer,
    TrainingConfig,
)
from sklearn.model_selection import train_test_split

# 1. Create labels with Triple Barrier Method (NOT shift(-1) > close!)
labeler = TripleBarrierLabeler(TripleBarrierConfig(
    take_profit_pct=0.02,    # 2% TP
    stop_loss_pct=0.01,      # 1% SL
    max_holding_period=24,   # 24 candles max
    fee_pct=0.001,           # 0.1% fee (accounts for trading costs)
))

df_labeled = labeler.create_labels(ohlcv_df)
# Labels: 1 = hit TP first, -1 = hit SL first, 0 = timeout

# 2. Split data BEFORE feature engineering (prevents data leakage)
train_df, test_df = train_test_split(df_labeled, test_size=0.2, shuffle=False)

# 3. Feature engineering with proper scaling
engineer = FeatureEngineer(FeatureConfig(
    scale_features=True,
    scaling_method="standard",
))

# FIT scaler on training data only (prevents data leakage!)
train_features = engineer.fit_transform(train_df)

# TRANSFORM test data using pre-fitted scaler (no refitting!)
test_features = engineer.transform(test_df)

# Save scaler for production
engineer.save_scaler("models/scaler.joblib")

# 4. Prepare train/test sets
feature_cols = engineer.get_feature_names()
X_train = train_features[feature_cols].dropna()
y_train = train_features.loc[X_train.index, 'label']

X_test = test_features[feature_cols].dropna()
y_test = test_features.loc[X_test.index, 'label']

# 5. Train model
trainer = ModelTrainer(TrainingConfig(
    model_type="lightgbm",
    optimize_hyperparams=True,
))
model, metrics = trainer.train(X_train, y_train, X_test, y_test)

print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
print(f"Test F1 Score: {metrics['test_f1']:.2%}")
```

### Market Regime Filter

```python
from src.strategies.market_regime import MarketRegimeFilter, RegimeFilterConfig

# Only trade when market conditions match your model's training data
filter = MarketRegimeFilter(RegimeFilterConfig(
    ema_period=200,
    adx_trend_threshold=25.0,    # ADX > 25 = trending
    adx_sideways_threshold=20.0, # ADX < 20 = sideways (no trade!)
    allow_long_in_bull=True,
    allow_short_in_bull=False,   # Don't short in uptrend
    allow_trade_in_sideways=False,  # Skip choppy markets
))

# Check before entering trade
should_trade, reason = filter.should_trade(dataframe, side='buy')

if not should_trade:
    print(f"Trade blocked: {reason}")
    # e.g., "SIDEWAYS regime: All trades blocked (ADX=15.3)"
else:
    # Execute trade
    pass
```

### Dynamic Position Sizing

```python
from src.risk import PositionSizer, PositionSizingConfig

sizer = PositionSizer(PositionSizingConfig(
    max_position_pct=0.10,   # Max 10% per position
    max_portfolio_risk=0.02, # Max 2% portfolio risk
))

# ATR-based sizing (volatility-adjusted)
result = sizer.calculate_atr_based_size(
    account_balance=10000,
    entry_price=50000,
    dataframe=ohlcv_df,
    atr_multiplier=2.0,      # Stop at 2x ATR
    risk_per_trade=0.01,     # Risk 1% per trade
)

print(f"Position size: {result['position_size']:.4f} BTC")
print(f"Stop loss: ${result['stop_loss_price']:.2f}")
print(f"Risk amount: ${result['risk_amount']:.2f}")
```

---

## 🧪 Testing

### Using Makefile (Recommended)

```bash
# Setup environment
make setup

# Run all tests
make test

# Run tests with coverage
make coverage

# Train ML model
make train

# Run hyperparameter optimization
make optimize

# Start trading (dry-run mode)
make trade
```

### Manual Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_order_manager/ -v
pytest tests/test_ml/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Type checking
mypy src/
```

---

## 🚀 Production Deployment

### 1. Configure Exchange API

Edit `.env`:

```env
# Binance API (or your exchange)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Enable live trading (CAREFUL!)
DRY_RUN=false
```

### 2. Start Trading

```bash
# Start production services
docker-compose up -d freqtrade frequi

# Monitor logs
docker-compose logs -f freqtrade
```

### 3. Monitor Performance

- FreqUI Dashboard: http://localhost:3000
- Check circuit breaker status
- Monitor position PnL
- Review trade history

⚠️ **WARNING**: Start with small amounts! Test thoroughly in paper trading mode first.

---

## 📊 Features Status

### ✅ Phase 1: Order Management System (COMPLETE)
- [x] 5 order types with state machine
- [x] Position tracking with real-time PnL
- [x] Circuit breaker protection
- [x] Slippage simulation
- [x] Order execution engine
- [x] **Smart Limit Orders** - Fee-optimized execution (Maker vs Taker)
- [x] 25+ unit tests (100% pass)
- [x] Complete documentation

### ✅ Phase 2: ML Training Pipeline (COMPLETE)
- [x] Feature engineering (50+ indicators)
- [x] **Triple Barrier Labeling** - Proper ML labels with TP/SL/Time barriers
- [x] **Dynamic Barrier Labeling** - ATR-adjusted barriers
- [x] Model training (RF, XGBoost, LightGBM)
- [x] Hyperparameter optimization (Optuna)
- [x] Experiment tracking (W&B/MLflow)
- [x] Model registry with versioning
- [x] Complete documentation

### ✅ Phase 3: Testing & Validation (COMPLETE)
- [x] ML Pipeline unit tests (40+ tests)
- [x] Risk management tests
- [x] Strategy tests
- [x] Labeling tests (21 tests)
- [x] 190+ total tests

### ✅ Phase 4: Monitoring & Metrics (COMPLETE)
- [x] Prometheus metrics export
- [x] Grafana dashboards
- [x] Alerting (Slack/Email/Telegram)
- [x] **ELK Stack for logs** - Structured logging with structlog
- [x] **Health Check System** - Kubernetes-ready health checks with FastAPI
- [x] **Load Testing** - Locust-based performance testing

### 📋 Phase 5: Live Trading Enhancements (IN PROGRESS)
- [x] **Structured Logging** - JSON logs for ELK integration
- [x] **Property-based Testing** - Hypothesis for robust testing
- [ ] Real-time ML inference integration
- [ ] Advanced position sizing algorithms
- [ ] Multi-exchange support
- [ ] Portfolio optimization

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on [Freqtrade](https://www.freqtrade.io/)
- Inspired by stoic philosophy principles
- Community-driven development

---

## 📧 Support

- 📖 **Documentation**: See [docs/](docs/) directory
- 🐛 **Bug Reports**: [Open an issue](https://github.com/kandibobe/stoic-citadel/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/kandibobe/stoic-citadel/discussions)

---

**🏛️ Stoic Citadel** - Trade with wisdom, not emotion.

**Status**: Production Ready
**Version**: 2.0.0
**Last Updated**: 2025-12-20
**Tests**: 190+ passing
