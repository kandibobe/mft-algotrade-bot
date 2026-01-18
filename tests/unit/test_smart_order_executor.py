import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.order_manager.smart_order import ChaseLimitOrder
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.risk.risk_manager import RiskManager


@pytest.mark.asyncio
async def test_multi_backend_routing():
    """Test that orders are routed to the correct backend based on exchange name."""

    # 1. Setup
    primary_config = {"name": "binance", "key": "k", "secret": "s"}
    secondary_config = [{"name": "kraken", "key": "k2", "secret": "s2"}]

    risk_manager = MagicMock(spec=RiskManager)
    risk_manager.circuit_breaker = MagicMock()
    risk_manager.circuit_breaker.can_trade.return_value = True
    risk_manager.evaluate_trade.return_value = {"allowed": True}

    executor = SmartOrderExecutor(
        exchange_config=primary_config,
        additional_exchanges=secondary_config,
        dry_run=True, # Forces MockBackend
        risk_manager=risk_manager
    )

    # Initialize (creates backends)
    await executor.start()

    assert "binance" in executor.backends
    assert "kraken" in executor.backends

    # 2. Test Primary Routing
    order_primary = ChaseLimitOrder(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        price=50000.0
    )

    # We spy on the mock backends
    binance_backend = executor.backends["binance"]
    kraken_backend = executor.backends["kraken"]

    # Mock create methods
    binance_backend.create_limit_buy_order = AsyncMock(return_value={"id": "binance_1"})
    kraken_backend.create_limit_buy_order = AsyncMock(return_value={"id": "kraken_1"})

    await executor.submit_order(order_primary, exchange="binance")
    await asyncio.sleep(0.1) # Allow background task to run

    binance_backend.create_limit_buy_order.assert_called_once()
    kraken_backend.create_limit_buy_order.assert_not_called()

    # 3. Test Secondary Routing
    order_secondary = ChaseLimitOrder(
        symbol="ETH/USD",
        side="buy",
        quantity=10.0,
        price=3000.0
    )

    # Reset mocks
    binance_backend.create_limit_buy_order.reset_mock()
    kraken_backend.create_limit_buy_order.reset_mock()

    await executor.submit_order(order_secondary, exchange="kraken")
    await asyncio.sleep(0.1) # Allow background task to run

    kraken_backend.create_limit_buy_order.assert_called_once()
    binance_backend.create_limit_buy_order.assert_not_called()

    await executor.stop()

@pytest.mark.asyncio
async def test_arbitrage_execution_flow():
    """Test that arbitrage execution submits two orders to correct exchanges."""

    # Setup
    primary_config = {"name": "binance", "key": "k", "secret": "s"}
    secondary_config = [{"name": "bybit", "key": "k2", "secret": "s2"}]

    risk_manager = MagicMock(spec=RiskManager)
    # Manually attach circuit_breaker mock since spec might not pick it up if it's dynamic
    risk_manager.circuit_breaker = MagicMock()
    risk_manager.circuit_breaker.can_trade.return_value = True
    risk_manager.evaluate_trade.return_value = {"allowed": True}

    executor = SmartOrderExecutor(
        exchange_config=primary_config,
        additional_exchanges=secondary_config,
        dry_run=True,
        risk_manager=risk_manager
    )
    await executor.start()

    # Mock backends
    executor.backends["binance"].create_limit_buy_order = AsyncMock(return_value={"id": "buy_1"})
    executor.backends["bybit"].create_limit_sell_order = AsyncMock(return_value={"id": "sell_1"})

    # Mock Ticker
    ticker = MagicMock()
    ticker.arbitrage_opportunity = True
    ticker.symbol = "BTC/USDT"
    ticker.best_ask = 50000.0
    ticker.best_ask_exchange = "binance"
    ticker.best_bid = 50100.0
    ticker.best_bid_exchange = "bybit"

    # Execute
    await executor.execute_arbitrage_trade(ticker, 0.1)
    await asyncio.sleep(0.1) # Allow background task to run

    # Verify
    executor.backends["binance"].create_limit_buy_order.assert_called_once()
    executor.backends["bybit"].create_limit_sell_order.assert_called_once()

    await executor.stop()
