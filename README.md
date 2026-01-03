# 🚀 Stoic Citadel - Hybrid MFT Trading System

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/github/actions/workflow/status/kandibobe/mft-algotrade-bot/ci.yml?branch=main)
![Architecture](https://img.shields.io/badge/architecture-hybrid-purple)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue)

**The Bridge Between Swing Trading and Mid-Frequency Execution**

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Roadmap](#-roadmap) • [Docs](docs/)

</div>

---

## 📋 Overview

Stoic Citadel is an advanced algorithmic trading system that fuses **Machine Learning** with **Mid-Frequency Trading (MFT)** concepts.

Unlike traditional bots that rely solely on lagging indicators, Stoic Citadel implements a **Hybrid Architecture**:
1.  **Macro View:** ML models (XGBoost/LightGBM) analyze 5m/1h candles to determine the trend and regime.
2.  **Micro View:** A real-time Websocket Aggregator monitors spread and orderbook pressure for optimal entry execution.

### 🎯 Key Features

-   **🤖 Regime-Adaptive ML**: Dynamically switches strategies based on Volatility (Hurst Exponent) and Market Phase.
-   **🛡️ Institutional Risk Management**: Volatility-adjusted position sizing, correlation de-risking, and circuit breakers.
-   **⚡ MFT Smart Execution**: Uses `ChaseLimitOrder` logic to dynamically adjust order prices based on real-time orderbook updates.
-   **🗄️ Persistence Layer**: Unified SQLAlchemy-based database abstraction.
-   **📊 Real-time Monitoring**: Prometheus/Grafana dashboard for spreads, execution latency, and strategy health.
-   **🔬 Advanced Pipeline**: Feature engineering, time-series cross-validation, and Optuna hyperparameter optimization.

## 🏗️ Architecture

See detailed [MFT Architecture Guide](docs/MFT_ARCHITECTURE.md).

```mermaid
graph TB
    subgraph "Macro Layer (Freqtrade)"
        A[Market Data (OHLCV)] --> B[Feature Engineering]
        B --> C[ML Inference]
        C --> D[Strategy Decision]
    end

    subgraph "Micro Layer (AsyncIO)"
        E[Websocket Stream] --> F[Data Aggregator]
        F --> G[Real-Time Metrics]
        G --> H[Execution Gate]
    end

    D --> H
    H --> I[Smart Order Executor]
    I --> J[Exchange API]
```

## 🗺️ Roadmap to Version 6.0

We have successfully transitioned to a Hybrid MFT system.

-   [x] **Phase 1: Foundation** - Robust ML pipeline, Risk Mixins, Unified Config.
-   [x] **Phase 2: Hybrid Connector** - Websocket Aggregator & Strategy Bridge.
-   [x] **Phase 3: Smart Execution** - Async Order Executor & Chase Limit Logic.
-   [x] **Phase 4: Optimization** - Latency reduction and safety checks.

## 🚀 Quick Start

### Prerequisites
-   Python 3.10+
-   Docker & Docker Compose (Recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kandibobe/mft-algotrade-bot.git
    cd mft-algotrade-bot
    ```

2.  **Install dependencies:**
    ```bash
    make install
    ```

3.  **Configure:**
    ```bash
    cp .env.example .env
    # Edit .env with your exchange keys
    ```

### Usage

**Run Backtest:**
```bash
python manage.py backtest
```

**Run Hyperopt:**
```bash
python manage.py optimize
```

**Start Production (Docker):**
```bash
docker-compose up -d
```

## 📚 Documentation

-   **[📖 MFT Architecture](docs/MFT_ARCHITECTURE.md)** - detailed system design.
-   **[📂 Project Structure](docs/project_structure.md)** - directory layout.
-   **[🧠 ML Guide](docs/ADVANCED_PIPELINE_GUIDE.md)** - machine learning pipeline.
-   **[🩺 System Health](docs/HEALTH_CHECK_SYSTEM.md)** - monitoring setup.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## ⚠️ Risk Disclaimer

This software is for educational purposes. **Trading cryptocurrency involves significant risk.** The authors are not responsible for any financial losses incurred.
