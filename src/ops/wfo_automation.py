"""
WFO Automation Manager
=======================

Automates the full Walk-Forward Optimization cycle for Stoic Citadel.
Orchestrates training, validation, and parameter export.
"""

import logging
import argparse
import json
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Fix python path to allow importing from scripts
sys.path.append(os.getcwd())

from src.backtesting.wfo_engine import WFOEngine, WFOConfig
from src.backtesting.vectorized_backtester import BacktestConfig
from scripts.analysis.walk_forward_analysis import WalkForwardAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WFOAutomation:
    def __init__(self, pair="BTC/USDT", timeframe="5m"):
        self.pair = pair
        self.timeframe = timeframe
        self.results_dir = Path("user_data/walk_forward_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_automated_cycle(self):
        """Run a full WFO cycle and export results."""
        logger.info(f"Starting Automated WFO Cycle for {self.pair}...")
        
        # 1. Setup Configuration
        wfo_cfg = WFOConfig(
            train_days=180,  # 6 months training
            test_days=30,    # 1 month validation
            step_days=30,    # Sliding step
            optimize_hyperparams=True,
            n_trials=30
        )
        
        bt_cfg = BacktestConfig(
            initial_capital=1000,
            fee_rate=0.001
        )
        
        # 2. Initialize WFO Analysis
        # We use the existing WalkForwardAnalysis script logic as it's more comprehensive
        wfa = WalkForwardAnalysis(results_dir=str(self.results_dir))
        
        # 3. Execute Analysis
        results = wfa.run(
            pair=self.pair,
            timeframe=self.timeframe,
            train_days=wfo_cfg.train_days,
            test_days=wfo_cfg.test_days,
            step_days=wfo_cfg.step_days,
            model_type="xgboost"
        )
        
        # 4. Export Best Parameters
        self._export_best_params(results)
        
        logger.info("Automated WFO Cycle Completed successfully.")
        return results

    def _export_best_params(self, results):
        """Extract and save best parameters from WFO results."""
        if not results or 'window_results' not in results:
            return
            
        # Get the latest successful window
        latest_window = None
        for res in reversed(results['window_results']):
            if res.get('success'):
                latest_window = res
                break
        
        if latest_window:
            export_data = {
                "pair": self.pair,
                "last_updated": datetime.now().isoformat(),
                "metrics": {
                    "accuracy": latest_window.get("model_accuracy"),
                    "profit_factor": latest_window.get("profit_factor"),
                    "win_rate": latest_window.get("win_rate")
                },
                "model_path": latest_window.get("model_path")
            }
            
            output_file = Path("config/model_best_params.json")
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=4)
            logger.info(f"Best parameters exported to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Stoic Citadel - WFO Automation")
    parser.add_argument("--pair", default="BTC/USDT", help="Pair to optimize")
    args = parser.parse_args()
    
    automation = WFOAutomation(pair=args.pair)
    automation.run_automated_cycle()

if __name__ == "__main__":
    main()
