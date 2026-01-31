import pytest
import asyncio
from unittest.mock import MagicMock
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.order_manager.smart_order import ChaseLimitOrder
from src.order_manager.order_types import OrderStatus
from src.websocket.aggregator import DataAggregator, TickerData
from src.risk.risk_manager import RiskManager

@pytest.mark.asyncio
async def test_full_trade_cycle_simulation():
    """
    End-to-End test simulating a full trade cycle using Mock Exchange.
    
    Flow:
    1. Initialize System (Executor, Risk, Aggregator)
    2. Simulate Market Data (Ticker)
    3. Submit Buy Order
    4. Simulate Price Movement to Fill Order
    5. Verify Execution & Logs
    """
    
    # 1. Setup
    aggregator = DataAggregator()
    risk_manager = RiskManager()
    
    # Mock Risk Manager to allow everything (focus on execution logic)
    risk_manager.circuit_breaker.can_trade = MagicMock(return_value=True)
    risk_manager.evaluate_trade = MagicMock(return_value={"allowed": True})
    
    # Configure Executor in Dry-Run Mode (uses MockExchangeBackend)
    exchange_config = {"name": "binance", "key": "test", "secret": "test"}
    executor = SmartOrderExecutor(
        aggregator=aggregator,
        exchange_config=exchange_config,
        dry_run=True,
        risk_manager=risk_manager
    )
    
    await executor.start()
    
    try:
        # 2. Simulate Initial Market Data
        symbol = "BTC/USDT"
        initial_price = 50000.0
        
        # Inject ticker
        ticker = TickerData(
            exchange="binance",
            symbol=symbol,
            bid=49990.0,
            ask=50010.0,
            last=50000.0,
            volume_24h=1000.0,
            change_24h=0.0,
            timestamp=1234567890.0
        )
        await aggregator._process_ticker(ticker)
        
        # 3. Submit Order
        order = ChaseLimitOrder(
            symbol=symbol,
            side="buy",
            quantity=0.1,
            price=50005.0 # Aggressive buy
        )
        
        order_id = await executor.submit_order(order)
        assert order_id is not None
        # Status can be SUBMITTED or OPEN depending on race condition with background task
        assert order.status in [OrderStatus.SUBMITTED, OrderStatus.OPEN]
        
        # Wait for internal loop to pick it up
        await asyncio.sleep(0.5)
        
        # 4. Simulate Fill Logic (MockBackend logic needs update or manual fill simulation)
        # Since MockExchangeBackend is basic, we can simulate fill by updating order status
        # via the backend if we had a handle to it, or by simulating market movement if using shadow mode logic.
        
        # In this test setup, we are in DRY-RUN mode.
        # SmartOrderExecutor loop calls fetch_order.
        # MockExchangeBackend.fetch_order returns "open".
        
        # To simulate a fill in this E2E test, we can manually update the mock backend's state
        mock_backend = executor.backend
        mock_order = mock_backend.orders.get(order.exchange_order_id)
        assert mock_order is not None
        
        # Simulate fill on the "exchange"
        mock_order["status"] = "closed" # CCXT mapped status
        mock_order["filled"] = 0.1
        mock_order["remaining"] = 0.0
        
        # Wait for Executor to poll and update
        await asyncio.sleep(1.5)
        
        # 5. Verification
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 0.1
        
        print(f"✅ Test Passed: Order {order_id} filled successfully.")
        
    finally:
        await executor.stop()
