# AUDIT REPORT 2025: Stoic Citadel Codebase Analysis

**Date:** December 24, 2025
**Scope:** `src/`, `scripts/`, `user_data/strategies/`
**Auditor:** AI Architect

## Executive Summary
The `Stoic Citadel` codebase contains robust ML infrastructure but suffers from critical integration issues (blocking calls in async code) and logical misalignment between Model Output and Strategy Thresholds. The repository also contains significant "dead code" (legacy strategies) that creates confusion.

---

## 🔴 CRITICAL: Blocking Bugs & Logic Errors

### 1. Blocking Calls in Async Context
**Risk:** High. Can freeze the bot during live trading, causing missed heartbeats or exchange timeouts.
*   **Location:** `src/order_manager/smart_limit_executor.py`
    *   `time.sleep(self.config.chase_interval_seconds)`
*   **Location:** `src/order_manager/order_executor.py`
    *   `time.sleep(delay_seconds)`
*   **Location:** `src/monitoring/metrics_exporter.py`
    *   `time.sleep(60)` (Inside main loop, blocks export)
*   **Location:** `src/main.py`
    *   `time.sleep(0.1)` (Inside main loop)

**Recommendation:** Replace all `time.sleep()` with `await asyncio.sleep()` in async methods, or run in separate threads.

### 2. Strategy Logic Mismatch (The "Silent Killer")
**Risk:** High. Bot may never enter trades.
*   **Issue:** `StoicEnsembleStrategyV4.py` implements a dynamic threshold clipped to `[0.50, 0.75]`.
*   **Reality:** User reports Model outputs max probability `0.40`.
*   **Result:** `if prediction > 0.50` is ALWAYS FALSE. The bot is effectively disabled.
*   **Fix:** Implement **Relative Percentile Thresholding** (e.g., enter if prediction is in top 10% of last 1000 candles), regardless of absolute value.

---

## 🟡 WARNING: Potential Issues

### 1. Feature Engineering Aggressive Cleaning
**Risk:** Medium. Data misalignment.
*   **Location:** `src/ml/training/feature_engineering.py` (`_apply_aggressive_cleaning`)
*   **Issue:** Drops ALL rows with `NaN`.
*   **Impact:** If rolling windows create NaNs at the start, these rows are deleted. If the Strategy passes 1000 candles and receives back 900 predictions, the indices must be perfectly aligned or signals will be shifted (trading on old data). `StoicEnsembleStrategyV4.py` attempts to handle this with index matching, but it is fragile.

### 2. Hardcoded Values
**Risk:** Low/Medium. Maintenance debt.
*   `StoicEnsembleStrategyV4.py`: Hardcoded `0.55` fallback for threshold.
*   `StoicEnsembleStrategyV4.py`: Hardcoded logic weights (`0.5`).

---

## 🔵 CLEANUP: Dead Code & Clutter

The following files appear to be legacy or unused and should be archived or deleted:

*   `user_data/strategies/StoicStrategyV1.py`
*   `user_data/strategies/StoicEnsembleStrategyV2.py`
*   `user_data/strategies/StoicEnsembleStrategyV3.py`
*   `user_data/strategies/StoicEnsembleStrategyV4_backtest.py` (Duplicate?)
*   `user_data/strategies/StoicEnsembleStrategyV4_hyperopt.py` (Duplicate?)
*   `user_data/strategies/DummyTestStrategy.py`

---

## ✅ RECOMMENDED ACTION PLAN

1.  **Immediate Fix:** Replace `time.sleep` with `await asyncio.sleep` in `src/order_manager/`.
2.  **Logic Repair:** Rewrite `populate_entry_trend` in `StoicEnsembleStrategyV4.py` to use `rolling_rank` or `quantile` for thresholds.
3.  **Cleanup:** Delete legacy strategies to reduce noise.
