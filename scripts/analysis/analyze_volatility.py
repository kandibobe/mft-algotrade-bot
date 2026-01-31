#!/usr/bin/env python
"""
Volatility Z-Score Analysis Script
==================================

Analyzes win rate and expectancy relative to market volatility (vol_zscore).
Helps identify optimal hard-cap for entries.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.regime_detection import calculate_regime
from src.ml.training import TripleBarrierConfig, TripleBarrierLabeler
from src.utils.logger import log as logger

def analyze_volatility_impact():
    logger.info("📊 Starting Volatility Impact Analysis...")

    # 1. Load data
    data_path = Path("user_data/data/binance/BTC_USDT-5m.feather")
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        return

    df = pd.read_feather(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    df = df.sort_index()

    # 2. Calculate Volatility Z-Score
    regime_df = calculate_regime(df["high"], df["low"], df["close"], df["volume"])
    df["vol_zscore"] = regime_df["vol_zscore"]

    # 3. Label data (Triple Barrier) for win/loss analysis
    labeler = TripleBarrierLabeler(
        TripleBarrierConfig(take_profit=0.01, stop_loss=0.005, max_holding_period=24)
    )
    df["target"] = labeler.label(df)

    # 4. Filter for only potential trades (Ignore 0.0/Neutral)
    trades = df[df["target"].isin([1.0, -1.0])].copy()
    trades["is_win"] = (trades["target"] == 1.0).astype(int)

    # 5. Bin Z-Score and calculate Win Rate
    # Bin by 0.5 steps
    bins = np.arange(-3, 5, 0.5)
    trades["z_bin"] = pd.cut(trades["vol_zscore"], bins)
    
    analysis = trades.groupby("z_bin", observed=True).agg(
        count=("is_win", "count"),
        win_rate=("is_win", "mean")
    )
    
    # Calculate expectancy (assuming 1% TP, 0.5% SL, and 0.4% costs)
    COSTS = 0.004
    analysis["expectancy"] = (analysis["win_rate"] * 0.01) - ((1 - analysis["win_rate"]) * 0.005) - COSTS

    print("\n--- Volatility Z-Score Analysis ---")
    print(analysis)

    # 6. Recommendation
    profitable_bins = analysis[analysis["expectancy"] > 0]
    if not profitable_bins.empty:
        max_z = profitable_bins.index.max().right
        min_z = profitable_bins.index.min().left
        logger.info(f"✅ Recommendation: Only trade when vol_zscore is between {min_z} and {max_z}")
    else:
        logger.warning("❌ No profitable vol_zscore bins found with current labeling.")
        
    # Plotting
    try:
        plt.figure(figsize=(10, 6))
        analysis["win_rate"].plot(kind="bar", color="skyblue")
        plt.axhline(y=0.5, color="red", linestyle="--", label="50% Win Rate")
        plt.title("Win Rate by Volatility Z-Score")
        plt.ylabel("Win Rate")
        plt.xlabel("Vol Z-Score Bin")
        plt.savefig("user_data/vol_analysis.png")
        logger.info("📈 Plot saved to user_data/vol_analysis.png")
    except Exception as e:
        logger.error(f"Could not save plot: {e}")

if __name__ == "__main__":
    analyze_volatility_impact()
