# 🗺️ Project Roadmap

This document outlines the development roadmap for Stoic Citadel. Our goal is to be transparent about our priorities and upcoming features.

## Guiding Principles
*   **Stability First:** Core infrastructure must be robust and reliable.
*   **Quant-Driven:** Features should be based on sound quantitative finance principles.
*   **Performance:** Latency and throughput are critical metrics.
*   **Pragmatism:** Avoid overengineering. Focus on features with high ROI (Return on Investment).

---

## ⛔ **Phase 1-4: Hybrid MFT Core (Completed)**

-   [x] **Foundation:** Robust ML pipeline, Risk Mixins, Unified Config.
-   [x] **Hybrid Connector:** Websocket Aggregator & Strategy Bridge.
-   [x] **Smart Execution:** Async Order Executor & Chase Limit Logic.
-   [x] **Optimization:** Latency reduction and safety checks.
-   [x] **V2.0 Release:** HRP, Meta-Learning, Feature Store.

---

## ✅ **Phase 5: Alpha Generation & Execution (Current Focus)**

### Q1 2026
*   **[Target: ✅ Completed] Advanced Order Types:**
    *   Implemented Iceberg Orders for reducing market impact.
    *   Developed "Pegged Order" type that tracks the best bid/ask dynamically.
    *   Added Post-Only execution safety.
*   **[Target: In Progress] Alternative Data Integration:**
    *   Integrate sentiment analysis (e.g., via Twitter/News API).
    *   Add support for key on-chain metrics.
*   **[Target: Planned] Performance Hardening:**
    *   Regular Hyperparameter Optimization (Nightly Hyperopt).
    *   Refining Meta-Labeling filters based on execution data.

---

## ➡️ **Phase 6: Practical AI & Stability (Future Research)**

### Q2-Q3 2026
*   **[Target: Research] RL-Based Position Sizing:**
    *   Develop a Reinforcement Learning agent specifically for dynamic capital allocation.
    *   Agent will learn to scale in/out based on market regime and model confidence.
*   **[Target: Research] Advanced Slippage Control:**
    *   Use ML/RL to predict short-term order book pressure and choose optimal order types.
*   **[Target: Planned] Cross-Exchange Monitoring:**
    *   Develop low-latency price comparison to improve fill quality across multiple venues.

### Q4 2026 and Beyond
*   **[Target: Vision] High-Performance Core Enhancements:**
    *   Selective rewrite of bottleneck components in Rust (only if performance warrants).
*   **[Target: Vision] DeFi Connectivity:**
    *   Explore integration with high-liquidity DEXs (e.g., dYdX v4, Hyperliquid) for diversification.

---

## Contribution

We welcome contributions! If you have an idea for a feature, please open a [Feature Request](/.github/ISSUE_TEMPLATE/bug_report.md) to discuss it.
