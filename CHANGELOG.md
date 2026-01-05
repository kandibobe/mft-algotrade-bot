# Changelog - Stoic Citadel Trading System

All notable changes to the **Stoic Citadel** project will be documented in this file.

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
