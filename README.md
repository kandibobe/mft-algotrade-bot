# Stoic Citadel: Hybrid MFT Trading System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Active](https://img.shields.io/badge/Status-Active-green)](https://github.com/kandibobe/mft-algotrade-bot)

**Stoic Citadel** is an advanced, hybrid Mid-Frequency Trading (MFT) bot designed for high-performance cryptocurrency trading. It bridges the gap between robust strategy management (Freqtrade) and low-latency execution (AsyncIO/Aiohttp), providing a sophisticated platform for quantitative trading.

---

## 🚀 Key Features

### 🧠 Hybrid Architecture
*   **Macro Strategy Layer (Freqtrade):** Leveraging the battle-tested Freqtrade framework for strategy definition, backtesting, and signal generation.
*   **Micro Execution Layer (Custom Async):** A high-performance, asynchronous execution engine (`ChaseLimit`) for optimal entry and exit, minimizing slippage and maximizing fill rates.

### 🛡️ Institutional-Grade Risk Management
*   **Circuit Breakers:** Automatic system pauses during extreme volatility or consecutively losing trades.
*   **Dynamic Position Sizing:** ATR-based volatility sizing to normalize risk across different assets.
*   **Correlation Protection:** Prevents overexposure to highly correlated assets.

### 🤖 Interactive Telegram Companion
A full-featured Telegram bot for real-time monitoring and control:
*   **Smart Alerts:** Set custom price triggers (e.g., `BTC > 100k`, `ETH +5%`).
*   **Portfolio Tracking:** Monitor your holdings and performance on the go.
*   **Market Intelligence:** Access real-time news, volatility scanners, and fear/greed indices directly from chat.

### 📈 Advanced Strategy: Stoic Ensemble
*   **Regime Detection:** Automatically adapts trading logic based on market conditions (Trending vs. Ranging).
*   **Multi-Strategy Ensemble:** Combines signals from multiple sub-strategies for robust decision-making.

---

## 🛠️ Quick Start

### Prerequisites
*   Python 3.10 or higher
*   Docker & Docker Compose (recommended)
*   Telegram Bot Token (from [@BotFather](https://t.me/BotFather))

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/kandibobe/mft-algotrade-bot.git
    cd mft-algotrade-bot
    ```

2.  **Set Up Environment**
    Copy the example configuration and fill in your details:
    ```bash
    cp .env.example .env
    ```
    *Edit `.env` to add your Exchange API Keys and Telegram Token.*

3.  **Install Dependencies** (for local development)
    ```bash
    pip install -r requirements.txt
    ```

### Running the System

**Option A: Full System (Docker)**
```bash
docker-compose up -d
```

**Option B: Hybrid Mode (Local)**
1.  Start the Trading Engine:
    ```bash
    freqtrade trade --config config.json --strategy StoicEnsembleStrategyV4
    ```
2.  Start the Companion Bot (in a separate terminal):
    ```bash
    python -m src.telegram_bot.runner
    ```

---

## 📱 Telegram Bot Guide

The companion bot extends your control beyond standard Freqtrade notifications.

| Feature | Command | Description |
| :--- | :--- | :--- |
| **Main Menu** | `/start` | Open the interactive dashboard. |
| **Add Alert** | `/addalert` | Create a custom price or indicator alert. |
| **Watchlist** | `/watchlist` | View your tracked assets. |
| **Market Report** | `/report` | Get a comprehensive market summary. |
| **Signals** | `/signal` | Check technical indicators for a specific pair. |

**Quick Alerts:**
Simply type in the chat:
> `BTC > 95000`
> `ETH +3%`

---

## 📂 Project Structure

```
mft-algotrade-bot/
├── config/                 # Configuration templates
├── docs/                   # Detailed documentation
├── src/
│   ├── strategies/         # Freqtrade strategy logic
│   ├── risk/               # Risk management modules
│   ├── order_manager/      # Execution engine (AsyncIO)
│   ├── telegram_bot/       # Interactive Companion Bot
│   └── ml/                 # Machine Learning pipeline
├── tests/                  # Unit and integration tests
└── user_data/              # Local data (logs, db, results)
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
