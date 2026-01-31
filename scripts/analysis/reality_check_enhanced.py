#!/usr/bin/env python
"""
Reality Check Validation - Enhanced
===================================

Validation script to verify system health on recent data with realistic slippage and fees.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.training import (
    FeatureConfig,
    FeatureEngineer,
    TripleBarrierConfig,
    TripleBarrierLabeler,
)
from src.utils.logger import log as logger

def run_enhanced_reality_check():
    logger.info("🕵️ Starting ENHANCED Reality Check on latest data...")

    # Configuration for realistic simulation
    SPREAD_PCT = 0.001  # 0.1%
    SLIPPAGE_PCT = 0.0005 # 0.05%
    FEE_PCT = 0.001 # 0.1%
    
    total_cost_entry = (SPREAD_PCT / 2) + SLIPPAGE_PCT + FEE_PCT
    total_cost_exit = (SPREAD_PCT / 2) + SLIPPAGE_PCT + FEE_PCT
    total_roundtrip_cost = total_cost_entry + total_cost_exit

    # 1. Load latest data
    data_path = Path("user_data/data/binance/BTC_USDT-5m.feather")
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        return

    df = pd.read_feather(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    df = df.sort_index()

    # Take last 14 days for more robust statistics
    latest_data = df.last("14D")
    logger.info(
        f"Analyzing data from {latest_data.index.min()} to {latest_data.index.max()} ({len(latest_data)} rows)"
    )

    # 2. Test Labeling (Triple Barrier)
    # We want to see if opportunities exist even with costs
    # target profit must be > roundtrip cost
    TARGET_PROFIT = 0.01 
    STOP_LOSS = 0.005
    
    labeler = TripleBarrierLabeler(
        TripleBarrierConfig(take_profit=TARGET_PROFIT, stop_loss=STOP_LOSS, max_holding_period=24)
    )

    labels = labeler.label(latest_data)
    dist = labels.value_counts(dropna=False).to_dict()
    logger.info(f"Label distribution (14 days): {dist}")

    # 3. Expectancy Calculation
    num_buys = dist.get(1.0, 0)
    num_sells = dist.get(-1.0, 0)
    num_neutral = dist.get(0.0, 0)
    total = len(labels)
    
    if num_buys > 0:
        # Simple theoretical profit assuming hits TP or SL exactly
        # In reality, neutral trades also exist
        win_rate = num_buys / (num_buys + num_sells + 1e-9)
        theoretical_expectancy = (win_rate * TARGET_PROFIT) - ((1 - win_rate) * STOP_LOSS)
        real_expectancy = theoretical_expectancy - total_roundtrip_cost
        
        logger.info(f"Theoretical Win Rate (Buy vs Sell): {win_rate:.2%}")
        logger.info(f"Theoretical Expectancy per trade: {theoretical_expectancy:.4%}")
        logger.info(f"Realistic Expectancy (after {total_roundtrip_cost:.4%} costs): {real_expectancy:.4%}")
        
        if real_expectancy > 0:
            logger.info("✅ SUCCESS: Strategy has positive expectancy after costs.")
        else:
            logger.warning("❌ FAILURE: Strategy expectancy is negative after realistic costs.")
            logger.warning("Suggestion: Increase confidence thresholds and improve entry quality.")

    # 4. Result
    if dist.get(1.0, 0) > 0:
        logger.info(f"Found {num_buys} Buy opportunities in the last 14 days.")
    else:
        logger.warning(
            "⚠️ WARNING: No Buy signals (Label=1) found with current thresholds."
        )

    # 5. Volatility Analysis (Step 1.1)
    # Check if high volatility correlate with losses
    # Using existing analysis object if possible, or recalculate
    if num_buys > 0:
        trades = latest_data[labels.isin([1.0, -1.0])].copy()
        trades["is_win"] = (labels == 1.0).astype(int)
        
        # Calculate Regime and Vol Z-Score
        from src.utils.regime_detection import calculate_regime
        regime_df = calculate_regime(trades["high"], trades["low"], trades["close"], trades["volume"])
        trades["vol_zscore"] = regime_df["vol_zscore"]
        
        # Bin and check
        trades["z_bin"] = pd.cut(trades["vol_zscore"], bins=np.arange(-2, 4, 1))
        vol_perf = trades.groupby("z_bin", observed=True)["is_win"].mean()
        logger.info(f"Win Rate by Vol Z-Score:\n{vol_perf}")

    logger.info("🚀 Reality Check Complete!")


if __name__ == "__main__":
    run_enhanced_reality_check()
