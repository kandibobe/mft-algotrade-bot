# Welcome to Stoic Citadel

**Institutional-Grade Mid-Frequency Trading (MFT) System**

## Overview

Stoic Citadel is a next-generation crypto trading framework designed to bridge the gap between traditional Algo-Trading bots (like Freqtrade) and HFT/MFT institutional systems. It leverages a **Hybrid Architecture** to combine strategic, synchronous analysis with high-speed, asynchronous execution.

## Current Status: Soft Launch

The system is currently in **Soft Launch** mode, running the V6 Strategy. This ensures:
*   **Safety:** Full integration with the `RiskManager`.
*   **Stability:** Use of proven ML models and HRP sizing.
*   **Performance:** Low-latency execution via the async order manager.

## Documentation Structure

*   **[Getting Started](getting_started/installation.md):** Installation, configuration, and first launch.
*   **[Architecture](architecture/overview.md):** Deep dive into the Hybrid Async/Sync design.
*   **[Risk Management](architecture/risk_layer.md):** Explanation of Circuit Breakers, HRP, and Drawdown protection.
*   **[ML Pipeline](guides/ml_pipeline.md):** How the Feature Store and Model Registry work.

## Quick Links

*   [Source Code](https://github.com/kandibobe/mft-algotrade-bot)
*   [Latest Audit Report](reports/archive/final_audit_fix_report.md)
