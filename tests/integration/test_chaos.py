"""
Chaos Monkey Test for Smart Order Executor
===========================================

Verifies the robustness of the SmartOrderExecutor against network failures and delays.
"""

import asyncio
import random
import logging
from unittest.mock import patch, AsyncMock

import pytest

from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.order_manager.smart_order import ChaseLimitOrder
from src.websocket.aggregator import DataAggregator
from src.risk.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@pytest.fixture
def event_loop():
    """Fixture to create a new event loop for each test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def setup_executor():
    """Set up the SmartOrderExecutor with a mock backend."""
    aggregator = DataAggregator()
    risk_manager = RiskManager()
    
    # Using dry_run=True will use a MockExchangeBackend
    executor = SmartOrderExecutor(
        aggregator=aggregator,
        dry_run=True,
        risk_manager=risk_manager
    )
    
    # Start the executor
    asyncio.create_task(executor.start())
    yield executor
    await executor.stop()

@pytest.mark.asyncio
async def test_chaos_network_delay(setup_executor):
    """Test graceful handling of network delays during order management."""
    executor = setup_executor
    
    original_fetch = executor.backend.fetch_order

    async def delayed_fetch(*args, **kwargs):
        delay = random.uniform(0.5, 1.0)
        logging.info(f"CHAOS: Injecting {delay:.2f}s delay into fetch_order")
        await asyncio.sleep(delay)
        return await original_fetch(*args, **kwargs)

    with patch.object(executor.backend, 'fetch_order', side_effect=delayed_fetch):
        order = ChaseLimitOrder(symbol="BTC/USDT", side="buy", quantity=0.01, price=50000)
        order_id = await executor.submit_order(order)
        
        await asyncio.sleep(2) # Allow time for chaos to take effect
        
        status = executor.get_order_status(order_id)
        assert status is not None # Should not crash

@pytest.mark.asyncio
async def test_chaos_connection_drop(setup_executor):
    """Test graceful handling of connection drops."""
    executor = setup_executor

    async def connection_error_fetch(*args, **kwargs):
        logging.info("CHAOS: Simulating connection drop")
        raise ConnectionError("CHAOS: Simulated connection error")

    with patch.object(executor.backend, 'fetch_order', side_effect=connection_error_fetch):
        order = ChaseLimitOrder(symbol="ETH/USDT", side="sell", quantity=0.1, price=3000)
        order_id = await executor.submit_order(order)
        
        await asyncio.sleep(2)
        
        # Executor should still be running and order should be in a recoverable state
        status = executor.get_order_status(order_id)
        assert status is not None
        assert executor._running

