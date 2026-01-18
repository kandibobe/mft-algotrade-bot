import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

from src.order_manager.order_types import OrderSide
from src.order_manager.smart_order import PeggedOrder
from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.websocket.aggregator import AggregatedTicker


class TestPeggedOrderExecution(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.aggregator = MagicMock()
        # Mock get_aggregated_ticker to return None by default to avoid Slippage Check failures
        self.aggregator.get_aggregated_ticker.return_value = None
        self.executor = SmartOrderExecutor(
            aggregator=self.aggregator,
            dry_run=True
        )
        await self.executor.start()

    async def asyncTearDown(self):
        await self.executor.stop()

    @patch('src.order_manager.exchange_backend.MockExchangeBackend.create_limit_buy_order')
    @patch('src.order_manager.exchange_backend.MockExchangeBackend.fetch_order')
    @patch('src.order_manager.exchange_backend.MockExchangeBackend.cancel_order')
    async def test_pegged_order_price_update(self, mock_cancel, mock_fetch, mock_create):
        # Setup mocks
        mock_create.return_value = {"id": "exch_123", "status": "open"}
        mock_fetch.return_value = {"id": "exch_123", "status": "open", "filled": 0, "price": 100.0}

        # Create a pegged buy order: Primary peg (Best Bid) + 0 offset
        order = PeggedOrder(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=100.0,
            offset=0.0,
            peg_side="primary"
        )

        # Submit order
        # Need to wait a bit because submit_order is async and starts a task
        order_id = await self.executor.submit_order(order)
        await asyncio.sleep(0.1) # Give time for _manage_order to start

        # Verify initial placement
        mock_create.assert_called_with("BTC/USDT", 1.0, 100.0, {})

        # Simulate ticker update: Best Bid moves from 100 to 101
        ticker = AggregatedTicker(
            symbol="BTC/USDT",
            best_bid=101.0,
            best_bid_exchange="binance",
            best_ask=102.0,
            best_ask_exchange="binance",
            spread=1.0,
            spread_pct=1.0,
            exchanges={},
            vwap=101.5,
            total_volume_24h=1000.0,
            timestamp=time.time()
        )

        # Trigger process_ticker_update
        await self.executor._process_ticker_update(ticker)

        # Verify order replacement
        mock_cancel.assert_called_with("exch_123", "BTC/USDT")
        # Should be called again with new price 101.0
        self.assertEqual(order.price, 101.0)
        mock_create.assert_called_with("BTC/USDT", 1.0, 101.0, {})

    @patch('src.order_manager.exchange_backend.MockExchangeBackend.create_limit_buy_order')
    @patch('src.order_manager.exchange_backend.MockExchangeBackend.fetch_order')
    async def test_pegged_order_opposite_side(self, mock_fetch, mock_create):
        mock_create.return_value = {"id": "exch_124", "status": "open"}
        mock_fetch.return_value = {"id": "exch_124", "status": "open", "filled": 0, "price": 2000.0}

        # Create a pegged buy order: Opposite peg (Best Ask) - 0.5 offset (Aggressive)
        order = PeggedOrder(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            quantity=10.0,
            price=2000.0,
            offset=-0.5,
            peg_side="opposite"
        )

        await self.executor.submit_order(order)
        await asyncio.sleep(0.1)

        # Simulate ticker update: Best Ask is 2010.0
        ticker = AggregatedTicker(
            symbol="ETH/USDT",
            best_bid=2005.0,
            best_bid_exchange="binance",
            best_ask=2010.0,
            best_ask_exchange="binance",
            spread=5.0,
            spread_pct=0.25,
            exchanges={},
            vwap=2007.5,
            total_volume_24h=5000.0,
            timestamp=time.time()
        )

        await self.executor._process_ticker_update(ticker)

        # New price should be 2010.0 - 0.5 = 2009.5
        self.assertEqual(order.price, 2009.5)

if __name__ == '__main__':
    unittest.main()
