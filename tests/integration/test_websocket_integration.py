"""
Integration Tests for Websocket Data Flow
=========================================

Tests the actual flow of data from DataAggregator to HybridConnector.
Unlike unit tests, this does NOT mock get_realtime_metrics.
It verifies that pushing data to the aggregator updates the strategy's state.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add user_data to path
sys.path.append(str(Path(__file__).parent.parent.parent / "user_data"))

from strategies.StoicEnsembleStrategyV5 import StoicEnsembleStrategyV5

from src.websocket.aggregator import DataAggregator
from src.websocket.data_stream import TickerData


@pytest.mark.asyncio
class TestWebsocketIntegration:

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance with a real DataAggregator."""
        # Mock Config Loading
        with patch('src.config.manager.ConfigurationManager.initialize') as mock_init, \
             patch('src.config.manager.ConfigurationManager.get_config') as mock_get_config:

            mock_config = MagicMock()
            mock_config.pairs = ["BTC/USDT"]
            mock_config.exchange.name = "binance"
            mock_config.exchange.api_key = "test_key"
            mock_config.exchange.api_secret = "test_secret"
            mock_config.dry_run = True
            mock_get_config.return_value = mock_config

            # Initialize Strategy
            with patch('freqtrade.strategy.IStrategy.__init__', return_value=None):
                strat = StoicEnsembleStrategyV5()
                strat.config = mock_config
                strat.dp = MagicMock()
                strat.dp.runmode.value = 'live'

                # Mock SmartOrderExecutor to avoid starting background tasks
                with patch('src.strategies.hybrid_connector.SmartOrderExecutor'):
                    strat.initialize_hybrid_connector(pairs=["BTC/USDT"])

                return strat

    async def test_data_flow_to_strategy(self, strategy):
        """Test that data pushed to aggregator is accessible via strategy."""
        aggregator = strategy._aggregator
        assert aggregator is not None
        assert isinstance(aggregator, DataAggregator)

        # 1. Simulate incoming data
        ticker = TickerData(
            symbol="BTC/USDT",
            bid=50000.0,
            ask=50010.0,
            last=50005.0,
            volume_24h=100.0,
            change_24h=0.0,
            timestamp=datetime.now().timestamp() * 1000,
            exchange="binance"
        )

        # Manually trigger the callback on the aggregator
        # In a real app, this comes from the websocket loop
        # We simulate the normalization and aggregation step

        # We can directly invoke the logic that the aggregator uses when it receives data
        # Or easier: directly trigger the on_aggregated_ticker event if we want to skip normalization logic
        # But let's try to be as close to reality as possible.

        # The aggregator usually subscribes to data_stream.
        # Here we can manually construct an AggregatedTicker and emit it.

        # Emit the event by simulating internal processing
        # 1. Process the raw ticker
        # We need a TickerData object (from data_stream.py)
        # But for this test, let's just bypass _process_ticker and inject directly into storage
        # to avoid needing a full TickerData object if Ticker is different.
        # Actually Ticker is usually for CCXT, TickerData is internal.

        ticker_data = TickerData(
            symbol="BTC/USDT",
            exchange="binance",
            bid=50000.0,
            ask=50010.0,
            last=50005.0,
            volume_24h=100.0,
            change_24h=0.0,
            timestamp=datetime.now().timestamp()
        )

        # Inject
        await aggregator._process_ticker(ticker_data)

        # 2. Trigger aggregation
        await aggregator._aggregate_and_emit()

        # 3. Verify Strategy State
        # The strategy should have cached the metrics
        cached = strategy.get_realtime_metrics("BTC/USDT")

        assert cached is not None
        assert cached.symbol == "BTC/USDT"
        assert cached.best_bid == 50000.0
        assert cached.best_ask == 50010.0
        assert cached.spread == 10.0
        assert cached.spread_pct == pytest.approx(0.02)

        # 4. Verify Market Safety Logic
        # Case A: Safe Spread (0.02% < 0.5%)
        assert strategy.check_market_safety("BTC/USDT", "long") is True

        # Case B: Unsafe Spread
        bad_ticker_data = TickerData(
            symbol="BTC/USDT",
            exchange="binance",
            bid=50000.0,
            ask=52000.0, # 4% spread
            last=51000.0,
            volume_24h=100.0,
            change_24h=0.0,
            timestamp=datetime.now().timestamp()
        )
        await aggregator._process_ticker(bad_ticker_data)
        await aggregator._aggregate_and_emit()

        cached_bad = strategy.get_realtime_metrics("BTC/USDT")
        assert cached_bad.spread_pct == pytest.approx(4.0)

        # Mocking logger to prevent clutter
        with patch('src.strategies.hybrid_connector.logger'):
            assert strategy.check_market_safety("BTC/USDT", "long") is False
