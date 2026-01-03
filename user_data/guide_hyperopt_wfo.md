# Stoic Citadel - Hyperoptimization & Walk-Forward Guide

## 1. Hyperoptimization Upgrade

We have introduced a custom **Sortino Ratio Loss Function** (`SortinoHyperOptLoss`) and expanded the search space in `StoicEnsembleStrategyV5`.

### Why Sortino Ratio?
Standard profit maximization often leads to "lucky" strategies that take huge risks. The Sortino Ratio divides return by **downside deviation** (volatility of negative returns only), effectively penalizing losses while ignoring upside volatility (which is good). This creates a smoother, safer equity curve.

### How to Run
Run the following command in your terminal:

```bash
freqtrade hyperopt \
    --hyperopt-loss SortinoHyperOptLoss \
    --strategy StoicEnsembleStrategyV5 \
    --spaces buy sell roi stoploss trailing \
    --timerange 20240101- \
    --epochs 500
```

*   `--hyperopt-loss SortinoHyperOptLoss`: Uses our new custom class.
*   `--spaces ...`: Optimizes entry (buy), exit (sell), ROI table, stoploss, and trailing stop.

### Interpreting Results
Freqtrade will output a table. Look for:
*   **Objective**: This is the minimized loss value. Lower is better.
*   **Profit %**: Total profit.
*   **Drawdown**: Maximum drawdown.
*   **Sortino**: (If calculated/shown in logs, otherwise inferred from low Objective).

A lower "Objective" with SortinoLoss means a **higher Sortino Ratio**. You might see slightly lower total profit compared to `ShortTradeDurHyperOptLoss`, but the **Drawdown** should be significantly reduced.

---

## 2. Walk-Forward Optimization (WFO)

To prevent overfitting, you should not optimize on the entire year and then trade on it. Instead, use a "Rolling Window" approach.

### Manual WFO Method
1.  **Period 1 (Train)**: Optimize Jan-Mar.
2.  **Period 1 (Test)**: Backtest Apr using parameters from Step 1.
3.  **Period 2 (Train)**: Optimize Feb-Apr.
4.  **Period 2 (Test)**: Backtest May using parameters from Step 3.
5.  **Repeat**.

### Automated Script (Bash)
Create a file named `wfo.sh`:

```bash
#!/bin/bash

# Configuration
STRATEGY="StoicEnsembleStrategyV5"
LOSS="SortinoHyperOptLoss"
TIMEFRAME="5m"

# Sliding Window: 90 days train, 30 days test
# Example Dates (adjust as needed)
declare -a TRAIN_STARTS=("20240101" "20240201" "20240301")
declare -a TRAIN_ENDS=("20240331" "20240430" "20240531")
declare -a TEST_STARTS=("20240401" "20240501" "20240601")
declare -a TEST_ENDS=("20240430" "20240531" "20240630")

for i in "${!TRAIN_STARTS[@]}"; do
    echo "------------------------------------------------"
    echo "Window $i: Train ${TRAIN_STARTS[$i]}-${TRAIN_ENDS[$i]} -> Test ${TEST_STARTS[$i]}-${TEST_ENDS[$i]}"
    
    # 1. Hyperopt (Train)
    freqtrade hyperopt \
        --strategy $STRATEGY \
        --hyperopt-loss $LOSS \
        --timerange "${TRAIN_STARTS[$i]}-${TRAIN_ENDS[$i]}" \
        --epochs 100 \
        --min-trades 20 \
        --spaces buy sell roi stoploss \
        --print-json > "user_data/wfo_params_$i.json"
        
    # Note: You would need to parse the JSON and update the strategy or config 
    # automatically to fully automate this, or manually update for the backtest step.
    
    # 2. Backtest (Test) - This requires manually loading the best params from step 1
    # freqtrade backtesting ...
done
```

---

## 3. Configuration Improvements (`config.json`)

To make backtests realistic:

1.  **Fees**: Set to `0.0006` (0.06%) for Binance Futures (VIP 0 Taker is 0.05%, plus some cushion).
    ```json
    "fee": 0.0006,
    ```
2.  **Slippage**: Simulate price movement during order execution.
    ```json
    "slippage": {
        "market": 0.001,  // 0.1% slippage for market orders
        "limit": 0.0      // Limit orders usually don't slip if filled
    },
    ```
3.  **Volume Filter**: Ensure we don't trade dead coins.
    In `pairlists`:
    ```json
    {
        "method": "VolumePairList",
        "number_assets": 40,
        "sort_key": "quoteVolume",
        "min_value": 0,
        "refresh_period": 1800,
        "lookback_days": 1
    }
    ```
    *(Note: You are currently using `StaticPairList`, which is fine for major pairs).*

4.  **Order Book**: Enable order book check to prevent buying into a wall.
    ```json
    "entry_pricing": {
        "use_order_book": true,
        "order_book_top": 1,
        "check_depth_of_market": {
            "enabled_in_dry_run": true,
            "bids_to_ask_delta": 1
        }
    }
    ```
