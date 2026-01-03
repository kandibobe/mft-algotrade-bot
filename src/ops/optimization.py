"""
Nightly Optimization Module
===========================

Handles automated hyperparameter optimization and model retraining.
"""

import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from src.ml.pipeline import MLTrainingPipeline

logger = logging.getLogger(__name__)

class NightlyOptimizer:
    def __init__(self, data_dir: str = "user_data/data/binance", results_dir: str = "user_data/hyperopt_results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_hyperopt(self, strategy: str, pairs: List[str], epochs: int = 100, config_path: str = None) -> bool:
        """Run Freqtrade Hyperopt."""
        logger.info(f"🚀 Starting Hyperopt for {strategy} ({epochs} epochs)")
        
        # Build command
        cmd = [
            "freqtrade", "hyperopt",
            "--strategy", strategy,
            "--hyperopt-loss", "SharpeHyperOptLoss",
            "--spaces", "buy", "sell", "roi", "stoploss",
            "-e", str(epochs),
            "--config", config_path or "user_data/config/config_backtest.json"
        ]
        
        # Note: We assume config handles pairs, or we patch config. 
        # For simplicity, we assume config is set up.
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("✅ Hyperopt completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Hyperopt failed: {e}")
            return False

    def run_ml_training(self, pairs: List[str], timeframe: str = "5m", n_trials: int = 50) -> bool:
        """Run ML Model Training."""
        logger.info(f"🚀 Starting ML Training for {pairs}")
        
        try:
            pipeline = MLTrainingPipeline(quick_mode=False)
            results = pipeline.run(
                pairs=pairs,
                timeframe=timeframe,
                optimize=True,
                n_trials=n_trials
            )
            
            # Check success
            success = all(r['success'] for r in results.values())
            if success:
                logger.info("✅ ML Training completed successfully")
                return True
            else:
                logger.error("❌ Some models failed to train")
                return False
                
        except Exception as e:
            logger.error(f"❌ ML Training crashed: {e}")
            return False

    def execute_nightly_cycle(self, strategy: str, pairs: List[str], epochs: int = 500, ml_trials: int = 100, config_path: str = None):
        """Execute the full nightly cycle."""
        start_time = datetime.now()
        report = []
        
        report.append(f"# Nightly Optimization Report - {start_time.strftime('%Y-%m-%d')}")
        
        # 1. Hyperopt
        if self.run_hyperopt(strategy, pairs, epochs, config_path):
            report.append(f"- [x] Hyperopt ({epochs} epochs): Success")
        else:
            report.append(f"- [ ] Hyperopt ({epochs} epochs): Failed")
            
        # 2. ML Training
        if self.run_ml_training(pairs, n_trials=ml_trials):
             report.append(f"- [x] ML Training ({ml_trials} trials): Success")
        else:
             report.append(f"- [ ] ML Training ({ml_trials} trials): Failed")
             
        # Write Report
        report_path = self.results_dir / f"nightly_report_{start_time.strftime('%Y%m%d')}.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report))
            
        logger.info(f"Nightly cycle finished. Report saved to {report_path}")
