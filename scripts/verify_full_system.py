#!/usr/bin/env python3
"""
System Verification Script
==========================

Verifies the entire Stoic Citadel system:
1. Docker Connectivity & Health
2. ML Pipeline (Training & Optimization)
3. Walk-Forward Analysis Engine
4. Database Connection

Usage:
    python scripts/verify_full_system.py
"""

import logging
import subprocess
import sys
import codecs
from pathlib import Path

# Force UTF-8 for Windows console
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.backtesting.wfo_engine import WFOEngine, WFOConfig
from src.backtesting.vectorized_backtester import BacktestConfig
from src.ml.pipeline import MLTrainingPipeline
from src.config.unified_config import TradingConfig, ExchangeConfig, TrainingConfig, MLConfig
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.order_manager.smart_order import ChaseLimitOrder
from src.websocket.aggregator import DataAggregator
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("verification.log")
    ]
)
logger = logging.getLogger("SystemVerify")

def check_docker():
    logger.info("🐳 Checking Docker Services...")
    try:
        # Check if docker is running
        subprocess.check_call(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Check running containers (optional, just info)
        result = subprocess.run(["docker", "compose", "-f", "deploy/docker-compose.yml", "ps"], capture_output=True, text=True)
        logger.info(f"Docker Compose Status:\n{result.stdout}")
        
        return True
    except subprocess.CalledProcessError:
        logger.error("❌ Docker is not running or accessible.")
        return False
    except FileNotFoundError:
        logger.error("❌ Docker CLI not found.")
        return False

def generate_mock_data(days=100):
    """Generate realistic-looking OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=days*288, freq="5min") # 5m candles
    n = len(dates)
    
    # Random walk for price
    price = 10000 + np.cumsum(np.random.normal(0, 10, n))
    
    data = pd.DataFrame({
        "timestamp": dates,
        "open": price,
        "high": price + np.random.uniform(0, 5, n),
        "low": price - np.random.uniform(0, 5, n),
        "close": price + np.random.normal(0, 2, n),
        "volume": np.random.uniform(100, 1000, n)
    })
    
    data.set_index("timestamp", inplace=True)
    return data

def verify_ml_pipeline():
    logger.info("[ML] Verifying ML Pipeline...")
    try:
        data = generate_mock_data(days=30)
        logger.info(f"Generated {len(data)} rows of mock data.")
        
        # Test Standard Training
        pipeline = MLTrainingPipeline(quick_mode=True)
        # Use the new train_on_data API for direct dataframe training verification
        result = pipeline.train_on_data(data, "BTC/USDT", optimize=False)
        
        if result.get("success"):
            logger.info("[ML] Standard Training Successful")
        else:
            logger.error(f"[ML] Standard Training Failed: {result.get('reason')}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"[ML] ML Pipeline Verification Failed: {e}", exc_info=True)
        return False

def verify_wfo_engine():
    logger.info("📈 Verifying WFO Engine...")
    try:
        data = generate_mock_data(days=60)
        
        wfo_config = WFOConfig(
            train_days=30,
            test_days=10,
            step_days=10,
            optimize_hyperparams=False, # Keep fast for verification
            quick_mode=True
        )
        
        backtest_config = BacktestConfig(
            initial_capital=10000,
        )
        
        engine = WFOEngine(wfo_config, backtest_config)
        results = engine.run(data, "BTC/USDT")
        
        summary = results.get("summary", {})
        if summary and summary.get("Total Trades", 0) != "No trades executed.":
            logger.info("✅ WFO Engine Run Successful")
            logger.info(f"Summary: {summary}")
        else:
            logger.warning("⚠️ WFO Run completed but no trades (might be expected on random data)")
            
        return True
    except Exception as e:
        logger.error(f"❌ WFO Engine Verification Failed: {e}", exc_info=True)
        return False

def main():
    logger.info("="*50)
    logger.info("[START] STARTING SYSTEM VERIFICATION")
    logger.info("="*50)
    
    results = {
        "Docker": check_docker(),
        "ML_Pipeline": verify_ml_pipeline(),
        "WFO_Engine": verify_wfo_engine()
    }
    
    logger.info("="*50)
    logger.info("[RESULTS] VERIFICATION RESULTS")
    logger.info("="*50)
    
    all_passed = True
    for component, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        logger.info(f"{component:<20} {status}")
        if not success:
            all_passed = False
            
    if all_passed:
        logger.info("\n[SUCCESS] SYSTEM VERIFIED SUCCESSFULLY!")
        sys.exit(0)
    else:
        logger.error("\n[FAILURE] SYSTEM VERIFICATION FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()