_____ _             _     _ _     _       _
  / ____| |           (_)   | (_)   | |     | |
 | (___ | |_ ___ _   _ _  __| |_  __| | __ _| |_
  \___ \| __/ _ \ | | | |/ _` | |/ _` |/ _` | __|
  ____) | ||  __/ |_| | | (_| | | (_| | (_| | |_
 |_____/ \__\___|\__,_|_|\__,_|_|\__,_|\__,_|\__|
```

**An Institutional-Grade, Hybrid MFT Algorithmic Trading System**

---

[![CI](https://github.com/kandibobe/mft-algotrade-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/kandibobe/mft-algotrade-bot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Stoic Citadel is a high-performance, hybrid Mid-Frequency Trading (MFT) system designed for robustness, speed, and intelligence. It combines the strategic depth of Freqtrade with a custom-built, low-latency execution layer based on Python's AsyncIO.

---

### **Visual Showcase**

<details>
<summary><strong>Click to see the system in action</strong></summary>

| Grafana Dashboard | Telegram Bot | Backtest Results |
| :---: | :---: | :---: |
| *Your Grafana screenshot here* | *Your Telegram GIF here* | *Your backtest plot here* |
| An overview of the system's performance and health. | Control and monitor the bot on the go. | Example equity curve from a recent backtest. |

</details>

---

### **Technology Stack**

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/docker-24.0-blue.svg)
![Freqtrade](https://img.shields.io/badge/freqtrade-latest-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-2.0-blue.svg)
![aiohttp](https://img.shields.io/badge/aiohttp-3.8-blue.svg)

---

### **For Different Audiences**

| **For Investors & Users** | **For Developers** | **For Quants & Researchers** |
| :--- | :--- | :--- |
| Interested in running the bot and its performance. | Focused on contributing to the codebase and system architecture. | Interested in the models, feature engineering, and alpha generation. |
| ➡️ **[Quick Start Guide](https://kandibobe.github.io/mft-algotrade-bot/getting_started/installation/)** | ➡️ **[Architecture Deep-Dive](https://kandibobe.github.io/mft-algotrade-bot/architecture/overview/)** | ➡️ **[ML Pipeline Guide](https://kandibobe.github.io/mft-algotrade-bot/guides/ml_pipeline/)** |
| ➡️ **[Telegram Bot Setup](https://kandibobe.github.io/mft-algotrade-bot/getting_started/telegram/)** | ➡️ **[Contribution Guide](CONTRIBUTING.md)** | ➡️ **[Strategy Development Guide](https://kandibobe.github.io/mft-algotrade-bot/guides/strategy_development/)** |

---

### **Core Features**

-   **Intelligent Alpha Generation:** A meta-learning ensemble of ML models (XGBoost, LightGBM) built on a triple-barrier labeling system to generate high-quality trading signals.
-   **Quantitative Risk Management:** An institutional-grade risk core featuring Hierarchical Risk Parity (HRP) for portfolio allocation, the Fractional Kelly Criterion for position sizing, and multi-level Circuit Breakers for system safety.
-   **High-Speed Execution:** A fully asynchronous Micro-Execution Layer with a custom `ChaseLimit` order type to minimize slippage and maximize maker-fee rebates.

---

### **Getting Started**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kandibobe/mft-algotrade-bot.git
    cd mft-algotrade-bot
    ```

2.  **Follow the documentation:**
    All setup and usage instructions are available in the **[official documentation](https://kandibobe.github.io/mft-algotrade-bot/getting_started/installation/)**.

### **Community & Support**

-   **Found a bug?** [Open a bug report](https://github.com/kandibobe/mft-algotrade-bot/issues/new?template=bug_report.md).
-   **Have a feature idea?** [Suggest a new feature](https://github.com/kandibobe/mft-algotrade-bot/issues/new?template=feature_request.md).
-   **Want to contribute?** Read our [**Contributing Guide**](CONTRIBUTING.md).

### **Sponsor This Project**

If you find Stoic Citadel valuable, please consider supporting its development.

*(Your funding/sponsorship details here)*

---

*Stoic Citadel - Built for stability, optimized for speed, driven by data.*
