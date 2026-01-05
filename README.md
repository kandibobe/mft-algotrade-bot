# 🛡️ Stoic Citadel: Hybrid MFT Trading System

Stoic Citadel is an institutional-grade, hybrid Mid-Frequency Trading (MFT) system designed for robustness, speed, and intelligence. It combines the strategic depth of Freqtrade with a custom-built, low-latency execution layer based on Python's AsyncIO.

## 🏛 Architecture

The system is decoupled into two primary layers to balance statistical depth with execution speed:

- **Macro Layer (Intelligence):** Handles long-term alpha generation, regime detection, and portfolio optimization. Powered by an ensemble of ML models (XGBoost/LightGBM) and Freqtrade.
- **Micro Layer (Execution):** A high-speed execution loop (<100ms) managing real-time orderbook dynamics, smart limit chasing, and emergency safety guards.

## 🚀 Key Features

### 🧠 Advanced ML Pipeline
- **Triple Barrier Labeling:** Statistically sound trade labeling following Marcos Lopez de Prado's methodology.
- **Meta-Labeling:** A secondary ML layer that predicts the probability of primary signal success, effectively filtering out low-confidence trades.
- **Walk-Forward Optimization (WFO):** Automated sliding-window training and validation to adapt to changing market regimes.
- **SHAP Feature Selection:** Mathematical identification of the most predictive indicators to prevent overfitting.

### 🛡️ Institutional Risk Management
- **Hierarchical Risk Parity (HRP):** Advanced portfolio allocation that uses clustering to group correlated assets and balance risk.
- **Fractional Kelly Criterion:** Mathematically optimal position sizing based on model confidence and historical win rates.
- **Circuit Breakers:** Multi-level safety switches that halt trading during extreme volatility or system anomalies.
- **Drift Analysis:** Daily automated comparison between backtest expectations and live execution results.

### ⚡ Smart Execution Engine
- **ChaseLimit Logic:** Dynamically adjusts order prices to stay at the top of the orderbook, maximizing maker-fee rebates.
- **Iceberg Orders:** Conceals large order sizes by splitting them into visible and hidden portions.
- **Maker-Fee Optimization:** Enforced `Post-Only` execution to minimize trading costs.
- **Self-Healing:** Automated recovery of WebSocket streams and critical async tasks.

## 🛠 Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Master Key for encryption

### 2. Installation
```bash
git clone https://github.com/kandibobe/mft-algotrade-bot.git
cd mft-algotrade-bot
pip install -r requirements.txt
pre-commit install
```

### 3. Setup Secret Keys
Encrypt your API keys before adding them to configuration files:
```bash
$env:STOIC_MASTER_KEY="your-master-password"
python -m src.utils.secret_manager "your-binance-api-key"
```
Copy the output (starting with `ENC:`) to your `config.json`.

### 4. Running with Docker
```bash
docker-compose up -d
```

## 📊 Monitoring & Control

- **Telegram Bot:** Interactive control via `/status`, `/balance`, and the **🚨 Panic Stop** button.
- **Dashboard:** Real-time analytics on Streamlit (PNL curves, Monte Carlo simulations, execution quality).
- **Metrics:** Detailed Prometheus metrics available at `:8000/metrics`.

## 📖 Documentation
Detailed technical guides are available in the `/docs` directory:
- [Architecture Overview](docs/ARCHITECTURE.md)
- [MFT Implementation Guide](docs/MFT_ARCHITECTURE.md)
- [Risk Management Specification](docs/RISK_MANAGEMENT_SPEC.md)
- [Walk-Forward Optimization Guide](docs/WALK_FORWARD_OPTIMIZATION_GUIDE.md)

---
*Stoic Citadel - Built for stability, optimized for speed, driven by data.*
