"""
Integration Tests for Stoic Ensemble Strategy V7
===============================================

Tests the integration of HybridConnector, RiskManager, and ML components 
into the V7 strategy implementation.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add user_data/strategies to path so we can import strategies
sys.path.append(str(Path(__file__).parent.parent.parent / "user_data" / "strategies"))

from StoicEnsembleStrategyV7 import StoicEnsembleStrategyV7

from src.websocket.aggregator import AggregatedTicker


class TestStoicV7Integration:

    @pytest.fixture
    def strategy(self):
        """Create a mocked V7 strategy instance."""
        # Mock Config
        mock_config = {
            "exchange": {
                "name": "binance",
                "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
                "sandbox": True
            },
            "dry_run": True,
            "runmode": "live"
        }

        # Mock Freqtrade Strategy Init
        with patch('freqtrade.strategy.IStrategy.__init__', return_value=None):
            strat = StoicEnsembleStrategyV7(config=mock_config)
            strat.config = mock_config
            strat.dp = MagicMock()
            strat.dp.runmode.value = 'live'

            return strat

    def test_v7_initialization(self, strategy):
        """Test that V7 initializes with expected parameters."""
        assert strategy.INTERFACE_VERSION >= 3
        assert hasattr(strategy, 'ml_confidence_threshold')
        assert strategy.trailing_stop is True

    def test_populate_indicators_ml_integration(self, strategy):
        """Test that populate_indicators attempts to fetch ML features."""
        df = pd.DataFrame({'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]})
        metadata = {'pair': 'BTC/USDT'}

        # Mock Feature Store
        mock_fs = MagicMock()
        # Mock get_online_features to return a DataFrame with predictions
        mock_fs.get_online_features.return_value = pd.DataFrame({
            'ensemble_prediction': [0.8],
            'prediction_confidence': [0.9]
        })

        with patch('StoicEnsembleStrategyV7.create_feature_store', return_value=mock_fs):
            # We need to ensure populate_indicators is called
            # and it handles the feature store initialization
            df_out = strategy.populate_indicators(df, metadata)

            assert 'ml_prediction' in df_out.columns

    def test_market_safety_gate(self, strategy):
        """Test the HybridConnector safety gate integration."""
        # Mock check_market_safety (from HybridConnectorMixin)
        with patch.object(strategy, 'check_market_safety', return_value=False) as mock_safety:
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
            mock_safety.assert_called_once()

    def test_custom_entry_price_mft(self, strategy):
        """Test that custom_entry_price uses MFT metrics."""
        # Mock get_realtime_metrics
        mock_ticker = AggregatedTicker(
            symbol="BTC/USDT",
            best_bid=49950.0,
            best_bid_exchange="binance",
            best_ask=50050.0,
            best_ask_exchange="binance",
            spread=100.0,
            spread_pct=0.2,
            exchanges={},
            vwap=50000.0,
            total_volume_24h=1000.0,
            timestamp=datetime.now().timestamp()
        )

        # Mock Trade object
        mock_trade = MagicMock()
        mock_trade.is_short = False

        if hasattr(strategy, 'custom_entry_price'):
            with patch.object(strategy, 'get_realtime_metrics', return_value=mock_ticker):
                price = strategy.custom_entry_price(
                    pair="BTC/USDT",
                    trade=mock_trade,
                    current_time=datetime.now(),
                    proposed_rate=50000.0,
                    entry_tag="test",
                    side="long"
                )
                assert price == 49950.0
