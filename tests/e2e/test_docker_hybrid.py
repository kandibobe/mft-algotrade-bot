
import pytest
import asyncio
import aiohttp
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.order_manager.smart_order import ChaseLimitOrder
from src.order_manager.order_types import OrderStatus

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_smart_order_executor_remote_mock():
    """
    Test SmartOrderExecutor against a running Mock Exchange container.
    This test verifies the network interaction layer.
    """
    mock_url = "http://mock_exchange:8888"
    
    # Check if mock exchange is reachable
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{mock_url}/health", timeout=2) as resp:
                if resp.status != 200:
                    pytest.skip("Mock exchange not healthy")
    except Exception:
        pytest.skip("Mock exchange not reachable (requires docker-compose environment)")

    # Configure Executor to use Remote Mock
    config = {
        "name": "binance",
        "use_remote_mock": True,
        "url": mock_url
    }
    
    executor = SmartOrderExecutor(
        exchange_config=config,
        dry_run=True # Keeps other safety checks happy
    )
    
    await executor.start()
    
    try:
        # Create and submit an order
        order = ChaseLimitOrder(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            price=50000.0
        )
        
        order_id = await executor.submit_order(order)
        
        # Allow some time for submission
        await asyncio.sleep(1)
        
        assert order.status == OrderStatus.SUBMITTED
        assert order.exchange_order_id is not None
        
        print(f"Order submitted: {order_id}, Exchange ID: {order.exchange_order_id}")
        
        # Verify order exists on the remote mock exchange
        async with aiohttp.ClientSession() as session:
            # We use the list endpoint as implemented in RemoteMockBackend
            async with session.get(f"{mock_url}/api/v1/orders?symbol=BTC/USDT") as resp:
                assert resp.status == 200
                data = await resp.json()
                orders = data.get("orders", [])
                
                # Find our order
                found = False
                for o in orders:
                    if o["order_id"] == order.exchange_order_id:
                        found = True
                        assert o["symbol"] == "BTC/USDT"
                        assert o["side"] == "buy"
                        assert o["quantity"] == 0.01
                        break
                
                assert found, "Order not found on remote mock exchange"
                
    finally:
        await executor.stop()