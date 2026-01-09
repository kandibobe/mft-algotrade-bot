#!/usr/bin/env python3
"""
Stoic Citadel - Risk Management Chaos Test
==========================================

Simulates extreme market conditions to verify Risk Manager defenses.
Scenarios:
1. Flash Crash (Series of rapid losses)
2. Max Drawdown Breach
3. Emergency Stop

Usage:
    python scripts/risk/chaos_test.py
"""

import sys
from pathlib import Path
from decimal import Decimal
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.risk.risk_manager import RiskManager, RiskMetrics
from src.risk.circuit_breaker import CircuitBreakerConfig
from src.risk.position_sizing import PositionSizingConfig
from src.risk.liquidation import LiquidationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChaosTest")

def run_chaos_test():
    logger.info("🔥 STARTING CHAOS TEST 🔥")
    
    # 1. Initialize Risk Manager with strict limits
    circuit_config = CircuitBreakerConfig(
        max_drawdown_pct=0.10,      # 10% Max Drawdown
        daily_loss_limit_pct=0.05,  # 5% Daily Loss Limit
        cooldown_minutes=60
    )
    
    rm = RiskManager(
        circuit_config=circuit_config,
        sizing_config=PositionSizingConfig(),
        liquidation_config=LiquidationConfig(),
        enable_notifications=False
    )
    
    initial_balance = 10000.0
    rm.initialize(account_balance=initial_balance)
    
    logger.info(f"Initial Balance: ${initial_balance}")
    
    # 2. Simulate Normal Trading (Win)
    logger.info("--- Phase 1: Normal Trading ---")
    trade_1 = rm.evaluate_trade("BTC/USDT", entry_price=50000, stop_loss_price=49000)
    if not trade_1["allowed"]:
        logger.error("❌ Trade 1 rejected unexpectedly!")
        sys.exit(1)
        
    size = float(trade_1["position_size"])
    rm.record_entry("BTC/USDT", 50000, size, 49000)
    
    # Exit with profit
    rm.record_exit("BTC/USDT", 51000) # 2% profit
    logger.info("✅ Trade 1: Win. System Healthy.")
    
    # 3. Simulate Flash Crash (Series of Losses)
    logger.info("--- Phase 2: Flash Crash Simulation ---")
    
    # Loss 1: -3%
    rm.record_entry("ETH/USDT", 3000, size, 2900)
    rm.record_exit("ETH/USDT", 2910) # -3%
    
    # Loss 2: -4% (Should trigger Daily Loss Limit of 5% total)
    rm.record_entry("SOL/USDT", 100, size, 90)
    rm.record_exit("SOL/USDT", 96) # -4%
    
    # Check Status
    status = rm.get_status()
    metrics = status["metrics"]
    cb_status = status["circuit_breaker"]
    
    logger.info(f"Current Daily PnL: {metrics['daily_pnl_pct']}%")
    logger.info(f"Circuit Breaker State: {cb_status['state']}")
    
    if cb_status['state'] == 'open':
        logger.info("✅ Circuit Breaker TRIPPED as expected (Daily Loss Limit).")
    else:
        logger.error(f"❌ Circuit Breaker failed to trip! State: {cb_status['state']}")
        sys.exit(1)
        
    # 4. Verify lockout
    logger.info("--- Phase 3: Lockout Verification ---")
    trade_rejected = rm.evaluate_trade("ADA/USDT", 1.0, 0.9)
    if not trade_rejected["allowed"]:
        logger.info(f"✅ Trade correctly rejected: {trade_rejected['rejection_reason']}")
    else:
        logger.error("❌ Risk Manager allowed trade during Circuit Breaker trip!")
        sys.exit(1)

    logger.info("🎉 CHAOS TEST PASSED: Risk Manager successfully protected the account.")

if __name__ == "__main__":
    run_chaos_test()
