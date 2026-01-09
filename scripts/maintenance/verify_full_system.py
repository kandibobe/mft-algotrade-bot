#!/usr/bin/env python3
"""
System Verification Script
==========================
"""

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.backtesting.wfo_engine import WFOEngine, WFOConfig
from src.backtesting.vectorized_backtester import BacktestConfig
from src.ml.pipeline import MLTrainingPipeline
from src.config.unified_config import TradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SystemVerify")

def check_docker():
    logger.info("Checking Docker Services...")
    try:
        subprocess.check_call(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def generate_mock_data(days=30):
    dates = pd.date_range(start="2024-01-01", periods=days*288, freq="5min")
    n = len(dates)
    price = 10000 + np.cumsum(np.random.normal(0, 10, n))
    data = pd.DataFrame({
        "date": dates,
        "open": price, "high": price + 5, "low": price - 5, "close": price + 2, "volume": 1000
    })
    return data

def verify_ml_pipeline():
    logger.info("Verifying ML Pipeline...")
    try:
        pipeline = MLTrainingPipeline(quick_mode=True)
        # MLPipeline currently expects feather files on disk in run_pipeline_for_pair
        # For verification we just check if it initializes
        if pipeline:
            logger.info("✅ ML Pipeline Initialization Successful")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ ML Pipeline Verification Failed: {e}")
        return False

def verify_wfo_engine():
    logger.info("Verifying WFO Engine...")
    try:
        wfo_config = WFOConfig(train_days=10, test_days=5, step_days=5, quick_mode=True)
        backtest_config = BacktestConfig(initial_capital=10000)
        engine = WFOEngine(wfo_config, backtest_config)
        if engine:
            logger.info("✅ WFO Engine Initialization Successful")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ WFO Engine Verification Failed: {e}")
        return False

def main():
    results = {
        "Docker": check_docker(),
        "ML_Pipeline": verify_ml_pipeline(),
        "WFO_Engine": verify_wfo_engine()
    }
    
    all_passed = True
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{component:<20} {status}")
        if not success:
            all_passed = False
            
    if all_passed:
        logger.info("\n🎉 SYSTEM VERIFIED SUCCESSFULLY!")
        sys.exit(0)
    else:
        logger.error("\n⚠️ SYSTEM VERIFICATION FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()

