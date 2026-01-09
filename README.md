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
*Soft Launch Ready - V6 Strategy Enabled*

---

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Hybrid%20Async-blueviolet" alt="Architecture">
  <img src="https://img.shields.io/badge/Risk-Institutional-red" alt="Risk Management">
  <img src="https://img.shields.io/badge/Execution-MFT-blue" alt="Execution">
  <img src="https://img.shields.io/badge/Status-Soft%20Launch-success" alt="Status">
</p>

</div>

---

### **Overview**

Stoic Citadel is a next-generation crypto trading framework designed to bridge the gap between traditional Algo-Trading bots (like Freqtrade) and HFT/MFT institutional systems.

It utilizes a **Hybrid Architecture**:
1.  **Macro Layer (Strategy):** Freqtrade-based synchronous logic for trend analysis and regime detection.
2.  **Micro Layer (Execution):** AsyncIO-based WebSocket Aggregator and Smart Order Executor for low-latency market interaction.

### **Current Status: Soft Launch**
The system is currently running on the **V6 Strategy (`StoicEnsembleStrategyV6`)**, which features:
*   **Fully Integrated HRP:** Hierarchical Risk Parity for dynamic portfolio rebalancing.
*   **ML Integration:** In-process ML confidence scoring.
*   **Safety Gates:** Comprehensive pre-trade checks via `RiskManager`.

### **Key Features**

| Feature | Description |
| :--- | :--- |
| **🚀 Hybrid Connector** | Seamless bridge between Strategy loop and real-time Websocket data. |
| **🧠 Feature Store** | Production-grade Feature Store (Feast/Redis) for online/offline consistency. |
| **🛡️ Unified Risk Engine** | Centralized `RiskManager` implementing Circuit Breakers, Drawdown protection, and HRP sizing. |
| **⚡ Smart Execution** | `ChaseLimit`, `Pegged`, and `Iceberg` orders to minimize slippage and impact. |
| **📊 Regime Detection** | Dynamic strategy adaptation based on Hurst Exponent and Volatility Z-Score. |

---

### **Architecture**

```mermaid
graph TD
    A[Market Data (WS)] -->|Async| B(Data Aggregator)
    B -->|Real-time Ticker| C{Smart Executor}
    B -->|Metrics| D[Hybrid Connector]
    
    E[Freqtrade Strategy] -->|Signal| D
    D -->|Order Request| C
    
    C -->|Order Placement| F[Exchange API]
    
    subgraph Risk Gate
    G[Risk Manager] -->|Check| C
    G -->|Check| D
    end
```

---

### **Getting Started**

#### 1. Configuration
The system uses a **Unified Configuration** system (`src/config/manager.py`).
To start with the recommended Soft Launch configuration:

```bash
cp .env.example .env
# Edit .env with your API Keys and Settings
```

#### 2. Deployment (Docker)
Run the system using Docker Compose:

```bash
docker-compose up -d --build
```

This will launch:
*   **Stoic Bot:** The main trading application.
*   **Redis:** For the Feature Store and Async messaging.
*   **PostgreSQL:** For persistent storage (trades, signals).

#### 3. Development & Testing
Install dependencies:

```bash
pip install -r requirements.txt
pre-commit install
```

Run the validation suite:
```bash
python scripts/maintenance/validate_config.py
pytest tests/integration/test_v6_logic.py
```

---

### **Project Structure**

*   `src/strategies/`: Trading strategies and Hybrid Connector.
*   `src/order_manager/`: Async Smart Order Execution logic.
*   `src/websocket/`: Real-time data aggregation.
*   `src/risk/`: Centralized risk management.
*   `src/ml/`: Machine Learning pipeline and Feature Store.
*   `src/config/`: Unified configuration system.
*   `scripts/`: Maintenance, analysis, and setup scripts.
*   `docs/`: Detailed documentation and architecture guides.

---

### **Troubleshooting**

*   **"Permission denied" in Database:**
    If you see `mkdir: can't create directory` in PostgreSQL logs, the volume permissions might be corrupted.
    Fix: Run `docker-compose down -v` to reset volumes and restart with `docker-compose up -d`.

*   **"Conflict" errors:**
    If you see container name conflicts (e.g., `stoic_postgres`), remove old containers manually:
    ```bash
    docker rm -f stoic_redis stoic_postgres stoic_freqtrade stoic_frequi
    ```

*   **Containers not starting:**
    Ensure you are running the command from the project root where the updated `docker-compose.yml` is located.

---

### **Documentation**
*   [Architecture Overview](docs/architecture/overview.md)
*   [Latest Audit Report](docs/reports/archive/final_audit_fix_report.md)
*   [Installation Guide](docs/getting_started/installation.md)

### **License**

MIT License. See [LICENSE](LICENSE) for details.
