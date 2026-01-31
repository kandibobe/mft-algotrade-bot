"""
Integration Tests for Hybrid Connector Strategy Flow
====================================================

Tests the integration of HybridConnector into StoicEnsembleStrategyV7.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add user_data/strategies to path so we can import strategies
sys.path.append(str(Path(__file__).parent.parent.parent / "user_data" / "strategies"))

from StoicEnsembleStrategyV7 import StoicEnsembleStrategyV7

from src.websocket.aggregator import AggregatedTicker


class TestHybridConnectorIntegration:

    @pytest.fixture
    def strategy(self):
        """Create a mocked strategy instance."""
        # Mock Config Loading
        with patch('src.config.manager.ConfigurationManager.initialize') as mock_init, \
             patch('src.config.manager.ConfigurationManager.get_config') as mock_get_config:

            mock_config = MagicMock()
            mock_config.pairs = ["BTC/USDT", "ETH/USDT"]
            mock_get_config.return_value = mock_config

            # Initialize Strategy
            # We mock the Freqtrade IStrategy init to avoid deep framework dependencies
            with patch('freqtrade.strategy.IStrategy.__init__', return_value=None):
                strat = StoicEnsembleStrategyV7(mock_config)
                strat.config = mock_config # Manually set config
                strat.dp = MagicMock()
                strat.dp.runmode.value = 'live' # Simulate live mode

                # Manually init Mixins if needed, but bot_start should handle it
                return strat

    def test_bot_start_initializes_connector(self, strategy):
        """Test that bot_start initializes the Hybrid Connector."""
        with patch.object(strategy, 'initialize_hybrid_connector') as mock_init_conn:
            strategy.bot_start()

            mock_init_conn.assert_called_once()
            # Verify called with correct pairs from config
            args, kwargs = mock_init_conn.call_args
            assert "BTC/USDT" in kwargs['pairs'] or "BTC/USDT" in args[0]

    def test_confirm_trade_entry_checks_safety(self, strategy):
        """Test that confirm_trade_entry calls check_market_safety."""
        # Mock check_market_safety
        strategy.check_market_safety = MagicMock(return_value=True)

        # Mock super().confirm_trade_entry to return True
        with patch('src.strategies.risk_mixin.StoicRiskMixin.confirm_trade_entry', return_value=True):
            allowed = strategy.confirm_trade_entry(
                pair="BTC/USDT",
                order_type="limit",
                amount=1.0,
                rate=50000.0,
                time_in_force="gtc",
                current_time=datetime.now(),
                entry_tag="test",
                side="long"
            )

            assert allowed is True
            strategy.check_market_safety.assert_called_once_with("BTC/USDT", "long")

    def test_unsafe_market_rejects_trade(self, strategy):
        """Test that unsafe market condition rejects trade."""
        # Mock check_market_safety to return False
        strategy.check_market_safety = MagicMock(return_value=False)

        allowed = strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=50000.0,
            time_in_force="gtc",
            current_time=datetime.now(),
            entry_tag="test",
            side="long"
        )

        assert allowed is False

    def test_check_market_safety_logic(self, strategy):
        """Test the actual logic of check_market_safety."""
        # We need to test the HybridConnectorMixin method directly
        # Mock get_realtime_metrics

        # Case 1: No data -> Safe (warning logged)
        with patch.object(strategy, 'get_realtime_metrics', return_value=None):
            assert strategy.check_market_safety("BTC/USDT", "long") is True

        # Case 2: High Spread -> Unsafe
        # Need to provide all required fields for dataclass
        bad_ticker = AggregatedTicker(
            symbol="BTC/USDT",
            best_bid=49000.0,
            best_bid_exchange="binance",
            best_ask=51000.0,
            best_ask_exchange="binance",
            spread=2000.0,
            spread_pct=4.0, # Spread = 4% > 0.5% limit
            exchanges={},
            vwap=50000.0,
            total_volume_24h=1000.0,
            timestamp=1234567890.0
        )
        with patch.object(strategy, 'get_realtime_metrics', return_value=bad_ticker):
            assert strategy.check_market_safety("BTC/USDT", "long") is False

        # Case 3: Low Spread -> Safe
        good_ticker = AggregatedTicker(
            symbol="BTC/USDT",
            best_bid=49990.0,
            best_bid_exchange="binance",
            best_ask=50010.0,
            best_ask_exchange="binance",
            spread=20.0,
            spread_pct=0.04, # Spread = 0.04% < 0.5% limit
            exchanges={},
            vwap=50000.0,
            total_volume_24h=1000.0,
            timestamp=1234567890.0
        )
        with patch.object(strategy, 'get_realtime_metrics', return_value=good_ticker):
            assert strategy.check_market_safety("BTC/USDT", "long") is True
