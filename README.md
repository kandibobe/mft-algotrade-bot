# 🏛️ Stoic Citadel - HFT Algorithmic Trading Bot

[![CI/CD](https://github.com/kandibobe/hft-algotrade-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/kandibobe/hft-algotrade-bot/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Freqtrade](https://img.shields.io/badge/freqtrade-2024.11-green.svg)](https://www.freqtrade.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *"The wise man accepts losses with equanimity."*

Professional-grade algorithmic trading system built on Freqtrade with advanced order management, ML pipeline, and production-ready features.

## ✨ Key Features

### 🎯 Trading Features
- **Ensemble Strategies** - Multiple sub-strategies voting for robust signals
- **Regime Detection** - Adapts to bull/bear/sideways markets automatically
- **Advanced Order Management** - 5 order types with state machine
- **Circuit Breaker** - Automatic risk protection against catastrophic losses
- **Position Management** - Real-time PnL tracking with stop-loss/take-profit

### 🤖 ML Pipeline (MLOps)
- **Feature Engineering** - 50+ technical indicators
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
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot
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
hft-algotrade-bot/
├── src/                              # Core modules
│   ├── order_manager/                # ✅ Order Management System
│   │   ├── order_types.py            # 5 order types with state machine
│   │   ├── position_manager.py       # Position tracking & PnL
│   │   ├── circuit_breaker.py        # Risk protection
│   │   ├── slippage_simulator.py     # Execution simulation
│   │   └── order_executor.py         # Order execution engine
│   ├── ml/                           # ML modules
│   │   ├── training/                 # ✅ ML Training Pipeline
│   │   │   ├── feature_engineering.py    # Feature pipeline
│   │   │   ├── model_trainer.py          # Model training
│   │   │   ├── experiment_tracker.py     # Experiment tracking
│   │   │   └── model_registry.py         # Model versioning
│   │   └── inference_service.py      # ML inference
│   ├── strategies/                   # Strategy base classes
│   ├── data/                         # Data loading & validation
│   └── utils/                        # Indicators, risk, regime
├── user_data/                        # Freqtrade user data
│   ├── strategies/                   # Trading strategies
│   ├── config/                       # Configuration files
│   ├── data/                         # Historical data
│   └── models/                       # Trained ML models
├── tests/                            # Test suite
│   └── test_order_manager/           # ✅ 25 unit tests
├── examples/                         # Working examples
│   └── order_management_example.py   # ✅ Complete demo
├── docs/                             # Documentation
│   ├── ORDER_MANAGEMENT.md           # ✅ Order Management API
│   ├── ML_TRAINING_PIPELINE.md       # ✅ ML Pipeline API
│   ├── STRATEGY_DEVELOPMENT_GUIDE.md
│   ├── TESTING_GUIDE.md
│   └── ...
├── config/                           # Strategy configs (YAML)
├── scripts/                          # Utility scripts
├── docker-compose.yml                # Production setup
├── QUICKSTART.md                     # ✅ Quick start guide
├── CREDENTIALS.md                    # ✅ Access credentials
└── README.md                         # This file
```

---

## 📚 Documentation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
- **[CREDENTIALS.md](CREDENTIALS.md)** - All access credentials and passwords
- **[START_HERE.md](START_HERE.md)** - Worktree guide (if using git worktree)

### API Documentation
- **[docs/ORDER_MANAGEMENT.md](docs/ORDER_MANAGEMENT.md)** - Order Management System API
- **[docs/ML_TRAINING_PIPELINE.md](docs/ML_TRAINING_PIPELINE.md)** - ML Pipeline API
- **[docs/STRATEGY_DEVELOPMENT_GUIDE.md](docs/STRATEGY_DEVELOPMENT_GUIDE.md)** - Strategy development
- **[docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)** - Testing guide

### Progress & Summary
- **[PROGRESS_SUMMARY.md](PROGRESS_SUMMARY.md)** - Development progress
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Complete implementation summary

---

## 🛠️ Usage Examples

### Order Management

```python
from src.order_manager import (
    OrderExecutor,
    CircuitBreaker,
    PositionManager,
    MarketOrder,
    OrderSide,
    ExecutionMode,
)

# Initialize components
circuit_breaker = CircuitBreaker()
position_manager = PositionManager(max_positions=3)
executor = OrderExecutor(
    mode=ExecutionMode.BACKTEST,
    circuit_breaker=circuit_breaker,
)

# Check if can trade
if not circuit_breaker.is_operational:
    print("Trading halted by circuit breaker!")
    exit()

# Create and execute order
order = MarketOrder(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    quantity=0.1
)

result = executor.execute(order, market_data={
    "price": 50000.0,
    "volume_24h": 1_000_000_000.0,
})

if result.success:
    position = position_manager.open_position(
        symbol=order.symbol,
        side="long",
        entry_price=result.execution_price,
        quantity=result.filled_quantity,
        stop_loss=result.execution_price * 0.95,
        take_profit=result.execution_price * 1.10,
    )
    print(f"Position opened: {position.position_id}")
```

### ML Training Pipeline

```python
from src.ml.training import (
    FeatureEngineer,
    ModelTrainer,
    ModelConfig,
    ExperimentTracker,
    ModelRegistry
)

# Feature engineering
engineer = FeatureEngineer()
features_df = engineer.transform(ohlcv_df)

# Prepare data
X_train = features_df[engineer.get_feature_names()]
y_train = (features_df['close'].shift(-1) > features_df['close']).astype(int)

# Track experiment
tracker = ExperimentTracker(project="stoic-citadel-ml")
tracker.start_run("xgboost_trend_v1")

# Train model
config = ModelConfig(model_type="xgboost", optimize_hyperparams=True)
trainer = ModelTrainer(config)
model, metrics = trainer.train(X_train, y_train)

# Log results
tracker.log_metrics(metrics)
tracker.log_model("models/xgboost_v1.pkl")
tracker.finish()

# Register model
registry = ModelRegistry()
metadata = registry.register_model(
    model_name="trend_classifier",
    model_path="models/xgboost_v1.pkl",
    metrics=metrics,
    feature_names=list(X_train.columns),
)

# Validate and promote to production
if registry.validate_model("trend_classifier", metadata.version):
    registry.promote_to_production("trend_classifier", metadata.version)
    print("Model deployed to production!")
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run order management tests
pytest tests/test_order_manager/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Windows: use run_tests.bat
run_tests.bat
```

### Run Examples

```bash
# Order management demo
python examples/order_management_example.py
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
- [x] 25 unit tests (100% pass)
- [x] Complete documentation

### ✅ Phase 2: ML Training Pipeline (COMPLETE)
- [x] Feature engineering (50+ indicators)
- [x] Model training (RF, XGBoost, LightGBM)
- [x] Hyperparameter optimization (Optuna)
- [x] Experiment tracking (W&B/MLflow)
- [x] Model registry with versioning
- [x] Complete documentation

### 📋 Phase 3: Testing & Validation (TODO)
- [ ] ML Pipeline unit tests
- [ ] Integration tests
- [ ] Automated backtest validation
- [ ] Performance benchmarks

### 📋 Phase 4: Monitoring & Metrics (TODO)
- [ ] Prometheus metrics export
- [ ] Grafana dashboards
- [ ] Alerting (Slack/Email)
- [ ] ELK Stack for logs

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
- 🐛 **Bug Reports**: [Open an issue](https://github.com/kandibobe/hft-algotrade-bot/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/kandibobe/hft-algotrade-bot/discussions)

---

**🏛️ Stoic Citadel** - Trade with wisdom, not emotion.

**Status**: Production Ready
**Version**: 1.2.0
**Last Updated**: 2025-12-17
