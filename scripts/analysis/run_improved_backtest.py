#!/usr/bin/env python3
"""
Stoic Citadel - Automated Backtest & WFO Cycle
==============================================

Runs a multi-stage validation cycle:
1. Standard Backtest (Long Period)
2. Walk-Forward Analysis (WFO)
3. Strategy Performance Summary
"""

import os
import sys
import subprocess
from datetime import datetime, timedelta

# Configuration
STRATEGY = "StoicEnsembleStrategyV6"
CONFIG = "user_data/config/config_production.json"
TIMEFRAME = "1h"
PAIRS = "BTC/USDT"
TIMERANGE = "20230101-" # From 2023 to present

def run_command(cmd):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def main():
    print(f"--- Starting Improved Backtest Cycle for {STRATEGY} ---")
    
    # 1. Standard Backtest
    print("\n[Step 1/3] Running Standard Backtest...")
    bt_cmd = [
        "freqtrade", "backtesting",
        "--config", CONFIG,
        "--strategy", STRATEGY,
        "--timeframe", TIMEFRAME,
        "--timerange", TIMERANGE,
        "--cache", "none"
    ]
    bt_output = run_command(bt_cmd)
    print(bt_output)
    
    # 2. Walk-Forward Analysis (Simulation)
    # Note: Using simple script if full WFO is not configured
    print("\n[Step 2/3] Running Walk-Forward Validation...")
    # Example: Check if we have enough data for 3-month windows
    wfo_cmd = [
        "python", "scripts/analysis/walk_forward_backtest.py",
        "--strategy", STRATEGY,
        "--config", CONFIG
    ]
    # Check if script exists, else use standard backtest with different ranges
    if os.path.exists("scripts/analysis/walk_forward_backtest.py"):
        wfo_output = run_command(wfo_cmd)
        print(wfo_output)
    else:
        print("WFO script not found, skipping or using manual ranges...")

    # 3. Performance Summary
    print("\n[Step 3/3] Generating Reality Check...")
    rc_cmd = [
        "python", "scripts/analysis/reality_check.py"
    ]
    if os.path.exists("scripts/analysis/reality_check.py"):
        rc_output = run_command(rc_cmd)
        print(rc_output)

    print("\n--- Backtest Cycle Complete ---")

if __name__ == "__main__":
    main()
