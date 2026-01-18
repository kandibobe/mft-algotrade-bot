"""
Burn-in Test for Stoic Ensemble Strategy V7
===========================================

Verifies that the strategy correctly interacts with the micro-layer (MFT hooks)
and uses real-time metrics for price calculation.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add user_data and src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "user_data"))

from user_data.strategies.StoicEnsembleStrategyV7 import StoicEnsembleStrategyV7


class TestV7BurnIn:

    @pytest.fixture
    def strategy(self):
        """Create a mocked V7 strategy instance."""
        mock_config = {
            "exchange": {"name": "binance", "pair_whitelist": ["BTC/USDT"], "sandbox": True},
            "dry_run": True,
            "runmode": "live",
            "timeframe": "5m"
        }

        # Mock IStrategy init and other freqtrade components
        with patch('freqtrade.strategy.IStrategy.__init__', return_value=None):
            strat = StoicEnsembleStrategyV7(config=mock_config)
            strat.config = mock_config
            strat.dp = MagicMock()
            strat.timeframe = '5m'

            # Initialize internal state
            strat.hrp_weights = {}
            strat.last_hrp_update = datetime.min
            strat.feature_store = None

            return strat

    def test_custom_entry_price_hook(self, strategy):
        """Test that custom_entry_price uses real-time Best Bid/Ask."""
        pair = "BTC/USDT"

        # 1. Create a mock ticker with specific Best Bid/Ask
        mock_ticker = MagicMock()
        mock_ticker.best_bid = 50000.0
        mock_ticker.best_ask = 50010.0

        # 2. Mock get_realtime_metrics to return our mock ticker
        strategy.get_realtime_metrics = MagicMock(return_value=mock_ticker)

        # 3. Call the hook for LONG entry
        price_long = strategy.custom_entry_price(
            pair=pair, trade=None, current_time=datetime.now(),
            proposed_rate=50005.0, entry_tag=None, side='long'
        )

        # Should return best_bid for long entry (ChaseLimit logic)
        assert price_long == 50000.0
        strategy.get_realtime_metrics.assert_called_with(pair)

        # 4. Call the hook for SHORT entry (if supported by strategy, though V7 is mostly long)
        price_short = strategy.custom_entry_price(
            pair=pair, trade=None, current_time=datetime.now(),
            proposed_rate=50005.0, entry_tag=None, side='short'
        )

        assert price_short == 50010.0

    def test_custom_exit_price_hook(self, strategy):
        """Test that custom_exit_price uses real-time Best Bid/Ask."""
        pair = "BTC/USDT"
        mock_trade = MagicMock()
        mock_trade.is_short = False

        mock_ticker = MagicMock()
        mock_ticker.best_bid = 49990.0
        mock_ticker.best_ask = 50000.0

        strategy.get_realtime_metrics = MagicMock(return_value=mock_ticker)

        # Exit Long trade -> Sell at Best Ask (or Best Bid depending on logic, V7 uses best_ask for exit long)
        # Note: In V7: price = ticker.best_ask if not trade.is_short else ticker.best_bid
        price_exit = strategy.custom_exit_price(
            pair=pair, trade=mock_trade, current_time=datetime.now(),
            proposed_rate=49995.0, current_profit=0.01, exit_tag="roi"
        )

        assert price_exit == 50000.0

    def test_populate_entry_trend_v7_logic(self, strategy):
        """Test the refined V7 entry logic including dynamic threshold."""

        # Create a mock dataframe with all required columns for V7
        # We need at least 13 rows to satisfy the persistence_window=3 and other potential rolling ops
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=15, freq='5m'),
            'open': [100.0]*15,
            'high': [101.0]*15,
            'low': [99.0]*15,
            'close': [100.0]*15,
            'volume': [1000.0]*15,
            'rsi': [30.0]*15,
            'ema_200': [90.0]*15,
            'bb_lower': [98.0]*15,
            'hurst': [0.6]*15, # Trending
            'regime': [1]*15,
            'vol_zscore': [0.5]*15,
            'ml_prediction': [0.8]*15,
            'ml_confidence': [0.6]*15
        })

        # Mock StoicLogic.populate_entry_exit_signals
        mock_signals = pd.DataFrame({
            'enter_long': [1]*15,
            'exit_long': [0]*15
        }, index=df.index)

        # Mocking load_config to return proper strategy parameters
        mock_config_obj = MagicMock()
        mock_config_obj.strategy.ml_confidence_base = 0.50
        mock_config_obj.strategy.ml_confidence_max = 0.90
        mock_config_obj.strategy.hurst_trending_threshold = 0.50

        with patch('src.strategies.core_logic.StoicLogic.populate_entry_exit_signals', return_value=mock_signals), \
             patch('src.strategies.base_strategy.BaseStoicStrategy.populate_indicators', return_value=df), \
             patch('src.config.unified_config.load_config', return_value=mock_config_obj):

                # Setup hyperparameters for the strategy instance itself
                strategy.buy_rsi = MagicMock()
                strategy.buy_rsi.value = 35
                strategy.sell_rsi = MagicMock()
                strategy.sell_rsi.value = 70

                df_out = strategy.populate_entry_trend(df, {'pair': 'BTC/USDT'})

                assert 'dynamic_threshold' in df_out.columns
                assert 'enter_long' in df_out.columns

                # Check if it's truthy (Freqtrade uses 1 or True)
                assert bool(df_out['enter_long'].iloc[-1]) == True
