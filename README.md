<div align="center">

```
   _____ _             _     _ _     _       _
  / ____| |           (_)   | (_)   | |     | |
 | (___ | |_ ___ _   _ _  __| |_  __| | __ _| |_
  \___ \| __/ _ \ | | | |/ _` | |/ _` |/ _` | __|
  ____) | ||  __/ |_| | | (_| | | (_| | (_| | |_
 |_____/ \__\___|\__,_|_|\__,_|_|\__,_|\__,_|\__|
```

**Institutional-Grade Mid-Frequency Trading (MFT) System**
*Engineered for Reliability, Speed, and Risk Management.*

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Hybrid%20Async-blueviolet?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/Risk-Institutional-red?style=for-the-badge" alt="Risk Management">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

[**Documentation**](docs/index.md) | [**Architecture**](docs/architecture/overview.md) | [**Report Bug**](https://github.com/kandibobe/mft-algotrade-bot/issues)

</div>

---

## **Introduction**

**Stoic Citadel** is a high-performance, hybrid trading framework designed to bridge the gap between retail algorithmic bots and institutional HFT systems. 

Unlike traditional bots that run on a single synchronous loop, Stoic Citadel employs a **Dual-Layer Architecture**:
1.  **Macro Layer (Strategy):** Uses Freqtrade for robust trend analysis, regime detection, and signal generation.
2.  **Micro Layer (Execution):** Uses a custom `AsyncIO` engine for millisecond-latency order management, smart execution algorithms (Iceberg, Chase), and real-time risk checks.

> **Status:** Soft Launch (Running V6 Strategy)

---

## **🚀 Quick Start**

Get up and running in less than 5 minutes.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and [Git](https://git-scm.com/downloads).

```bash
# 1. Clone the repository
git clone https://github.com/kandibobe/mft-algotrade-bot.git
cd mft-algotrade-bot

# 2. Configure environment (Copy default settings)
# Windows (PowerShell):
copy .env.example .env
# Mac/Linux:
cp .env.example .env

# 3. Launch the system
docker-compose up -d --build
```

**Access Points:**
*   📊 **Dashboard:** [http://localhost:3000](http://localhost:3000)
*   ⚙️ **API:** [http://localhost:8080](http://localhost:8080)
*   🔐 **Credentials:** Default user is `stoic_admin`. See `.env` file for the generated password.

---

## **🌟 Key Differentiators**

### **1. Hybrid "Sync-Async" Core**
Most bots are slow because they process everything in one loop. Stoic Citadel decouples **Thinking** (Strategy) from **Acting** (Execution).
*   *Strategy* runs every 1m/5m/1h to find setups.
*   *Execution* runs in microseconds to fill orders at the best price.

### **2. Institutional Risk Engine**
We don't just use Stop Loss. We use:
*   **HRP (Hierarchical Risk Parity):** Mathematically optimal portfolio rebalancing.
*   **Volatility Targeting:** Adjusts size based on market turbulence.
*   **Circuit Breakers:** Hard stops that trigger if daily drawdown exceeds limits.

### **3. Smart Execution Router**
Never market buy blindly. The `SmartOrderExecutor` uses:
*   **ChaseLimit:** Places limit orders and updates them dynamically to avoid spread costs.
*   **Iceberg:** Splits large orders to hide intent.
*   **Pegged:** Floats orders relative to the spread.

---

## **🏗 Architecture**

The system is built on a modular Event-Driven Architecture.

```mermaid
graph TD
    subgraph "Micro Layer (AsyncIO / Fast)"
    WS[WebSocket Aggregator] -->|Real-time Ticks| EX{Smart Executor}
    EX -->|Orders| API[Exchange API]
    end

    subgraph "Macro Layer (Sync / Robust)"
    FT[Freqtrade Strategy] -->|Signals| HC[Hybrid Connector]
    ML[Feature Store] -->|Predicitons| FT
    end

    subgraph "Safety & Governance"
    RM[Risk Manager] -->|Pre-Trade Check| EX
    RM -->|Portfolio Check| HC
    end

    HC <-->|Bridge| WS
    HC -->|Order Request| EX
```

---

## **🛠 Development & Management**

The project includes a robust CLI for management.

**Requirements:** Python 3.10+, Docker.

### **Initial Setup (Dev)**
```bash
# Linux/macOS
make dev-install

# Windows (PowerShell)
.\manage.ps1 dev-install
```

### **Common Commands**

| Task | Command (Unix) | Command (Windows) | Description |
| :--- | :--- | :--- | :--- |
| **Run Tests** | `make test` | `.\manage.ps1 test` | Run unit tests |
| **Lint Code** | `make lint` | `.\manage.ps1 lint` | Check code quality |
| **Backtest** | `make backtest` | `.\manage.ps1 backtest` | Run strategy simulation |
| **Update Ops** | `make update` | `.\manage.ps1 update` | Pull latest updates |

---

## **📂 Project Structure**

*   `src/strategies/` - Hybrid Strategies & Connector Logic.
*   `src/order_manager/` - Async Execution Engine.
*   `src/risk/` - HRP, Circuit Breakers, and Sizing.
*   `src/ml/` - Feature Store & Model Registry.
*   `src/websocket/` - Real-time Data Streamers.
*   `deploy/` - Docker & Kubernetes configurations.

---

## **🤝 Contributing**

We welcome contributions from the community!
1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## **🛡 Security**

Security is paramount.
*   API Keys are managed via `.env` (never committed).
*   No external calls are made without explicit configuration.
*   See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

---

## **📜 License**

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <sub>Built with ❤️ by the Stoic Citadel Team</sub>
</div>