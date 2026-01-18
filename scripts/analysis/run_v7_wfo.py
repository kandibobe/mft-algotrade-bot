#!/usr/bin/env python3
"""
Walk-Forward Optimization (WFO) Script
======================================

Runs a full Walk-Forward Optimization for StoicEnsembleStrategyV7.
Tests ML model robustness by training on past data and testing on subsequent future data.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.backtesting.wfo_engine import WFOConfig, WFOEngine
from src.backtesting.vectorized_backtester import BacktestConfig
from src.config import config
from src.utils.logger import setup_structured_logging

logger = logging.getLogger("stoic_wfo")

def run_v7_wfo(pair: str = "BTC/USDT", timeframe: str = "5m", train_days: int = 180, test_days: int = 30):
    """Run WFO for V7 strategy."""
    setup_structured_logging(json_output=False)
    cfg = config()
    
    # 1. Load Data
    pair_slug = pair.replace("/", "_")
    data_path = cfg.paths.data_dir / "binance" / f"{pair_slug}-{timeframe}.feather"
    
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}. Please download data first.")
        return
        
    df = pd.read_feather(data_path)
    df = df.set_index('date')
    
    logger.info(f"Loaded {len(df)} rows of data for {pair}")

    # 2. Configure WFO
    wfo_cfg = WFOConfig(
        train_days=train_days,
        test_days=test_days,
        step_days=test_days, # Move forward by test period
        optimize_hyperparams=False, # Set to True for full optimization
        quick_mode=True
    )
    
    bt_cfg = BacktestConfig(
        fee_rate=0.001,
        slippage_rate=0.0005,
        initial_capital=10000.0
    )

    # 3. Run WFO Engine
    engine = WFOEngine(wfo_cfg, bt_cfg)
    results = engine.run(df, pair)

    # 4. Save Results
    results_dir = Path("user_data/walk_forward_results/v7_wfo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"wfo_report_{pair_slug}_{timestamp}.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# WFO Report: {pair} ({timeframe})\n\n")
        f.write(f"Executed at: {datetime.now()}\n\n")
        f.write("## Summary Metrics\n\n")
        for key, value in results["summary"].items():
            f.write(f"- **{key}**: {value}\n")
            
        f.write("\n## Fold Results\n\n")
        f.write("| Fold | Num Trades | Return |\n")
        f.write("|------|------------|--------|\n")
        for fold in results["fold_results"]:
            f.write(f"| {fold['fold']} | {fold['num_trades']} | {fold['return']:.2%} |\n")

    logger.info(f"WFO Report saved to {report_path}")
    print(f"\n🚀 WFO Completed for {pair}")
    print(f"Total Return: {results['summary']['Total Return']}")
    print(f"Sharpe Ratio: {results['summary']['Sharpe Ratio']}")
    print(f"Total Trades: {results['summary']['Total Trades']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run V7 Walk-Forward Optimization")
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--train-days", type=int, default=180, help="Days for training")
    parser.add_argument("--test-days", type=int, default=30, help="Days for testing")
    
    args = parser.parse_args()
    run_v7_wfo(pair=args.pair, train_days=args.train_days, test_days=args.test_days)
