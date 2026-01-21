"""
Retrain Production Model Script
===============================

Automated script to:
1. Load real data (Parquet)
2. Train models for BTC/USDT and ETH/USDT
3. Evaluate performance
4. Update the model registry
"""

import sys
import logging
from pathlib import Path

# Add src to python path
sys.path.append(str(Path.cwd()))

from src.ml.pipeline import MLPipeline
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    logger.info("=€ Starting Production Model Retraining...")

    # Initialize Pipeline
    # Data is in user_data/data/parquet (converted from feather)
    pipeline = MLPipeline(
        exchange_name="binance",
        data_dir="user_data/data/parquet", 
        models_dir="user_data/models",
        quick_mode=True # Use quick mode for this demonstration/test run
    )

    pairs = ["BTC/USDT", "ETH/USDT"]
    timeframes = ["5m", "1h"]

    results = {}

    for timeframe in timeframes:
        logger.info(f"Training models for timeframe: {timeframe}")
        trained_models = pipeline.run(pairs, timeframe=timeframe)
        results.update(trained_models)

    # Report
    logger.info("\n" + "="*40)
    logger.info("TRAINING REPORT")
    logger.info("="*40)
    
    success_count = 0
    for pair, path in results.items():
        if path:
            logger.info(f" {pair}: Saved to {path}")
            success_count += 1
        else:
            logger.error(f"L {pair}: Training Failed")

    if success_count == len(results):
        logger.info("<‰ All models trained successfully.")
    else:
        logger.warning(f"   Completed {success_count}/{len(results)} models.")

if __name__ == "__main__":
    main()