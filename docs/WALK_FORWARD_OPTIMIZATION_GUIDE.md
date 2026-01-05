# 🚶 Walk-Forward Optimization (WFO) Guide

This cookbook explains how to perform robust strategy validation using Walk-Forward Optimization. WFO is superior to standard backtesting because it tests the strategy on "unseen" data, simulating real-world performance more accurately.

## 1. Concept

**Standard Optimization (In-Sample):**
- Train on 2023 data -> Test on 2023 data.
- **Risk:** Overfitting. The model "memorizes" the past.

**Walk-Forward Optimization:**
- **Step 1:** Train on Jan-Mar (In-Sample).
- **Step 2:** Test on Apr (Out-of-Sample).
- **Step 3:** Train on Feb-Apr.
- **Step 4:** Test on May.
- ... Repeat.

## 2. Prerequisites

Ensure you have historical data for the timeframe you want to test.

```bash
# Download 1 year of data for top 20 pairs
docker-compose run --rm freqtrade download-data \
    --exchange binance \
    --days 365 \
    --timeframe 5m 1h
```

## 3. Running WFO

We utilize the `src/backtesting/wfo_engine.py` via our analysis scripts.

### Basic Command

```bash
python scripts/analysis/walk_forward_optimization.py \
    --strategy StoicEnsembleStrategyV6 \
    --config user_data/config/config_production.json \
    --timerange 20230101-20240101 \
    --train-period 90 \
    --test-period 30
```

### Parameters explained:
- `--train-period 90`: Use 90 days of data to find best parameters.
- `--test-period 30`: Test those parameters on the *next* 30 days.
- `--timerange`: The total period to cover.

## 4. Analyzing Results

The script generates a report in `user_data/walk_forward_results/`. Look for:

1.  **Stability**: Do parameters change drastically between windows?
2.  **Performance Drop**: Is OOS (Out-of-Sample) performance significantly worse than IS (In-Sample)?
    - Drop < 20% is Good.
    - Drop > 50% indicates Overfitting.

### Example Output

| Window | Train Date | Test Date | IS Sharpe | OOS Sharpe | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Jan-Mar | Apr | 2.5 | 2.1 | ✅ Robust |
| 2 | Feb-Apr | May | 2.8 | 0.5 | ❌ Overfit |
| 3 | Mar-May | Jun | 2.4 | 2.2 | ✅ Robust |

## 5. Automated Recalibration

For live trading, you can automate this process.

1.  **Schedule:** Run WFO every month.
2.  **Update:** If the new parameters are stable, update your strategy config.

```python
# Example of auto-update logic in your maintenance script
if oos_sharpe > 1.5 and drawdown < 0.2:
    update_live_config(new_params)
    notify_telegram("Strategy Recalibrated Successfully")
else:
    notify_telegram("Recalibration Failed: Keeping old parameters")
```

## 6. Common Pitfalls

-   **Too Short Train Period:** The optimizer doesn't have enough regimes to learn from. (Min: 60 days).
-   **Too Long Test Period:** Market regime shifts before the next recalibration. (Max: 30 days).
-   **Over-optimization:** Optimizing too many parameters at once. Stick to 3-5 key parameters (e.g., RSI threshold, Stoploss, ROI).
