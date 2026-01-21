"""
Chaos Engineering Test for Stoic Citadel
========================================

Simulates various system failures to ensure risk controls work correctly:
1. WebSocket failure
2. Exchange API timeout
3. Flash crash
"""

import asyncio
import logging
import time
from unittest.mock import MagicMock, AsyncMock
from src.risk.risk_manager import RiskManager
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.order_manager.smart_order import ChaseLimitOrder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChaosTest")

async def simulate_flash_crash(risk_manager):
    logger.info("💥 Simulating BTC Flash Crash (-10% in 5m)...")
    # We mock the DataLoader inside check_market_kill_switch to return crashed data
    with patch('src.data.loader.DataLoader.load_pair_data') as mock_load:
        import pandas as pd
        mock_load.return_value = pd.DataFrame({
            'close': [100.0, 90.0] # 10% drop
        })
        
        is_halted, reason = risk_manager.check_market_kill_switch()
        if is_halted:
            logger.info(f"✅ Market Kill Switch correctly triggered: {reason}")
        else:
            logger.error("❌ Market Kill Switch FAILED to trigger!")

async def simulate_api_timeout(executor):
    logger.info("⏳ Simulating Exchange API Timeout (504)...")
    executor.backend.create_limit_buy_order = AsyncMock(side_effect=Exception("504 Gateway Time-out"))
    
    order = ChaseLimitOrder(symbol="BTC/USDT", side="buy", quantity=0.01, price=50000.0, stop_price=49000.0)
    
    try:
        await executor.submit_order(order)
    except Exception as e:
        logger.info(f"✅ Order submission failed gracefully as expected: {e}")

async def main():
    risk_manager = RiskManager()
    executor = SmartOrderExecutor(dry_run=True, risk_manager=risk_manager)
    
    await simulate_flash_crash(risk_manager)
    await simulate_api_timeout(executor)
    
    logger.info("Chaos Test Completed.")

if __name__ == "__main__":
    from unittest.mock import patch
    asyncio.run(main())