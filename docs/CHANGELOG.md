# Changelog - Stoic Citadel Trading System

All notable changes to the **Stoic Citadel** project will be documented in this file.

## [2.1.0] - 2026-01-20

### 🚀 Major Feature Release: V7 Core
- **Hybrid Online Learning:** Integrated `River` library for real-time model adaptation. The system now learns from live trade outcomes (Profit/Loss, slippage) to adjust confidence scores dynamically.
- **Explainable AI (XAI):** Added `SHAP` and `Lime` integration. Every trade signal now carries an explanation vector (e.g., "RSI < 30 contributed +0.4 to confidence").
- **Multi-Exchange Execution:** Added `MT5Backend` to support Forex and Indices via MetaTrader 5, alongside the existing crypto `CCXT` backend.
- **News Filter:** New `NewsFilter` component that parses ForexFactory/Investing.com calendars and automatically halts trading before high-impact events.

### 📱 User Experience
- **Telegram Bot Overhaul:** Complete rewrite of the bot interface with inline keyboards, pagination, and a new "Business Metrics" dashboard.
- **Performance Dashboard:** Real-time visualization of Sharpe Ratio, Sortino Ratio, and Max Drawdown in the web interface.

### 🛡️ Risk & Stability
- **Enhanced HRP:** Hierarchical Risk Parity now supports clustering constraints to prevent over-concentration in correlated assets.
- **Liquidation Engine:** Improved `LiquidationService` with "Chase-to-Close" logic for faster position exiting during stress events.

## [2.0.1] - 2026-01-09

### 🔧 Critical Fixes & Stability
- **V6 ML Integration:** Fixed Feature Store connectivity; Strategy now uses `StrategyMLAdapter` with robust fallback to `ml_confidence` derivation.
- **Hierarchical Risk Parity (HRP):** Fully activated HRP sizing with hourly weight updates and safety clamps (`hrp_min_weight`, `hrp_max_weight`).
- **Config Hardening:** Fixed schema validation errors and added missing `stoploss` parameter for Paper Trading stability.
- **Test Coverage:** Added `tests/integration/test_v6_logic.py` to verify HRP and ML flows before launch.

## [2.0.0] - 2026-01-05

### 🔥 AI & Machine Learning
- **Meta-Labeling Integration:** Implemented a secondary ML layer (De Prado's methodology) to filter signals and reduce false positives.
- **Walk-Forward Optimization (WFO):** Added automated sliding-window optimization to prevent overfitting and adapt to market regimes.
- **SHAP Feature Selection:** Integrated mathematical feature importance analysis to clean noise from the model.
- **Probability Calibration:** Added Platt Scaling and Isotonic proxies to make model confidence scores statistically valid.

### 🛡️ Risk Management & Stability
- **Hierarchical Risk Parity (HRP):** Implemented ML-based portfolio allocation for superior diversification.
- **Fractional Kelly Criterion:** Added dynamic position sizing based on model win-probability.
- **Drift Analysis:** Implemented automated daily reporting to detect model performance degradation.
- **Circuit Breakers:** Strengthened multi-level safety guards.

### ⚡ Execution & MFT Layer
- **Advanced Order Types:** Added support for **Iceberg** and **Post-Only** orders.
- **Self-Healing Engine:** Automated recovery logic for WebSocket streams and async services.
- **MFT Latency Tracking:** Full Signal-to-Fill latency logging in the database.
- **Panic Stop 2.0:** High-speed market liquidation logic triggered via Telegram.

### 🔒 Security & Operations
- **Secret Encryption:** AES-256 encryption for API keys with master-key management.
- **Hot Reload:** Configuration updates now applied without stopping the trading bot.
- **DB Migrations:** Integrated **Alembic** for professional database schema versioning.
- **CI/CD Pipeline:** Hardened GitHub Actions with security scanning (Bandit) and performance regression tests.

### 📊 Monitoring & Documentation
- **Real-time Dashboard:** Streamlit-based cockpit with Monte Carlo simulations and execution quality metrics.
- **Interactive Telegram Bot:** Full system control via mobile (Status, Balance, Reload, Panic Stop).
- **Auto-generated Docs:** Integrated MkDocs with code-to-docs automation (mkdocstrings).

---
*Stoic Citadel v2.0 - Institutional Performance, Retail Accessibility.*