#!/usr/bin/env python3
"""
Check ML predictions and strategy conditions.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def analyze_predictions():
    """Analyze ML predictions and entry conditions."""
    
    # Load the most recent backtest results
    backtest_dir = "user_data/backtest_results"
    if not os.path.exists(backtest_dir):
        print(f"Backtest directory not found: {backtest_dir}")
        return
    
    # Find most recent backtest file
    backtest_files = [f for f in os.listdir(backtest_dir) if f.endswith('.meta.json')]
    if not backtest_files:
        print("No backtest results found")
        return
    
    latest_file = max(backtest_files, key=lambda x: os.path.getmtime(os.path.join(backtest_dir, x)))
    latest_path = os.path.join(backtest_dir, latest_file)
    
    print(f"Loading backtest results from: {latest_file}")
    
    with open(latest_path, 'r') as f:
        results = json.load(f)
    
    # Check strategy conditions
    print("\n=== Strategy Configuration ===")
    print(f"Strategy: {results.get('strategy', 'Unknown')}")
    print(f"Timeframe: {results.get('timeframe', 'Unknown')}")
    print(f"Timerange: {results.get('timerange', 'Unknown')}")
    print(f"Pairs: {results.get('pairs', [])}")
    
    # Check trade statistics
    print("\n=== Trade Statistics ===")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    
    if results.get('total_trades', 0) == 0:
        print("\n=== No Trades Analysis ===")
        print("Possible reasons:")
        print("1. Data availability (data starts from 2024-12-22)")
        print("2. Strict entry conditions in strategy")
        print("3. ML model predictions below threshold")
        print("4. Trend conditions not met (EMA filters)")
        print("5. Volume conditions not met")
    
    # Check the strategy file for conditions
    print("\n=== Strategy Entry Conditions (from StoicEnsembleStrategyV4) ===")
    print("ALL of these must be true:")
    print("1. ml_prediction > dynamic_threshold (calculated from recent predictions)")
    print("2. ml_confidence > 0.55")
    print("3. close > ema_200 (uptrend)")
    print("4. ema_50 > ema_100 (medium-term uptrend)")
    print("5. volume_ratio > 0.7")
    print("6. rsi < 70")
    print("7. bb_width > buy_bb_width_min.value (default 0.02)")
    print("8. bb_width < 0.2")
    print("\nAdditional regime-specific conditions may apply.")
    
    # Check data availability
    print("\n=== Data Availability ===")
    data_dir = "user_data/data/binance"
    if os.path.exists(data_dir):
        pairs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Available pairs: {pairs}")
        
        # Check BTC/USDT data
        btc_dir = os.path.join(data_dir, "BTC_USDT")
        if os.path.exists(btc_dir):
            files = [f for f in os.listdir(btc_dir) if f.endswith('.json')]
            if files:
                # Load first file to check dates
                sample_file = os.path.join(btc_dir, files[0])
                with open(sample_file, 'r') as f:
                    data = json.load(f)
                if data and len(data) > 0:
                    first_date = data[0][0] / 1000  # Convert from ms
                    last_date = data[-1][0] / 1000
                    print(f"BTC/USDT data range: {datetime.fromtimestamp(first_date)} to {datetime.fromtimestamp(last_date)}")
                    print(f"BTC/USDT data points: {len(data)}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    print("1. Download more historical data (full 12 months)")
    print("2. Test with simpler strategy (StoicEnsembleStrategyV2 or V3)")
    print("3. Adjust entry conditions to be less strict")
    print("4. Test with different timerange where data is available")
    print("5. Check if ML model is generating reasonable predictions")

if __name__ == "__main__":
    analyze_predictions()
