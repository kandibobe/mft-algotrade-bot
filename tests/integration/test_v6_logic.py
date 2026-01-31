"""
Integration Tests for Stoic Ensemble Strategy V7 Logic
======================================================

Tests the specific V7 logic improvements:
1. HRP Weight Calculation and Application
2. ML Prediction and Confidence Derivation
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add user_data/strategies to path
sys.path.append(str(Path(__file__).parent.parent.parent / "user_data" / "strategies"))

from StoicEnsembleStrategyV7 import StoicEnsembleStrategyV7


class TestV7Logic:

    @pytest.fixture
    def strategy(self):
        """Create a mocked V7 strategy instance."""
        mock_config = {
            "exchange": {"name": "binance", "pair_whitelist": ["BTC/USDT", "ETH/USDT"], "sandbox": True},
            "dry_run": True,
            "runmode": "live",
            "timeframe": "5m"
        }

        with patch('freqtrade.strategy.IStrategy.__init__', return_value=None):
            strat = StoicEnsembleStrategyV7(config=mock_config)
            strat.config = mock_config
            strat.dp = MagicMock()
            strat.dp.runmode.value = 'live'
            strat.timeframe = '5m'

            # Initialize internal state manually since we mocked __init__
            strat.hrp_weights = {}
            strat.last_hrp_update = datetime.min
            strat.feature_store = None

            # Mock base method return values
            strat.custom_stake_amount = MagicMock(return_value=100.0) # Base returns proposed stake

            return strat

    def test_hrp_weight_update(self, strategy):
        """Test that HRP weights are updated correctly."""

        # Mock DP data
        strategy.dp.current_whitelist.return_value = ["BTC/USDT", "ETH/USDT"]

        # Mock DataFrames for pairs
        btc_df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=100, freq='h'), 'close': np.random.randn(100).cumsum() + 1000})
        eth_df = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=100, freq='h'), 'close': np.random.randn(100).cumsum() + 100})

        def get_pair_dataframe(pair, timeframe):
            if pair == "BTC/USDT": return btc_df
            if pair == "ETH/USDT": return eth_df
            return pd.DataFrame()

        strategy.dp.get_pair_dataframe.side_effect = get_pair_dataframe

        # Mock get_hrp_weights to return known weights
        mock_weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}

        # Patch where it is used (imported)
        with patch('StoicEnsembleStrategyV7.get_hrp_weights', return_value=mock_weights):
            strategy._update_hrp_weights()

            assert strategy.hrp_weights == mock_weights
            assert strategy.last_hrp_update > datetime.min

    def test_custom_stake_amount_with_hrp(self, strategy):
        """Test that stake amount is adjusted by HRP weights."""

        # Setup HRP weights
        strategy.hrp_weights = {"BTC/USDT": 0.8} # Weight > 0.5 (min default)

        # Setup Hyperparameters (mocking property access)
        strategy.hrp_min_weight = MagicMock()
        strategy.hrp_min_weight.value = 0.5
        strategy.hrp_max_weight = MagicMock()
        strategy.hrp_max_weight.value = 1.5

        # Restore the real custom_stake_amount method on the instance
        strategy.custom_stake_amount = StoicEnsembleStrategyV7.custom_stake_amount.__get__(strategy, StoicEnsembleStrategyV7)

        # Mock super().custom_stake_amount to return base stake
        with patch('src.strategies.base_strategy.BaseStoicStrategy.custom_stake_amount', return_value=100.0):

            # Case 1: HRP applied
            stake = strategy.custom_stake_amount(
                pair="BTC/USDT", current_time=datetime.now(), current_rate=50000.0,
                proposed_stake=100.0, min_stake=10.0, max_stake=1000.0,
                leverage=1.0, entry_tag=None, side="long"
            )

            # Calculation: weight 0.8 * num_assets (1) = 0.8
            # scaling_factor = max(0.5, min(0.8, 1.5)) = 0.8
            # adjusted = 100 * 0.8 = 80.0
            assert stake == 80.0

    def test_feature_store_init_only_live(self, strategy):
        """Test that feature store is initialized in live mode."""
        strategy.config['runmode'] = 'live'
        strategy.feature_store = None

        # Patch where it is used (imported)
        with patch('StoicEnsembleStrategyV7.create_feature_store') as mock_create:
            mock_store = MagicMock()
            mock_create.return_value = mock_store

            # Calling bot_start should trigger init
            with patch('src.strategies.base_strategy.BaseStoicStrategy.bot_start'):
                strategy.bot_start()
                mock_create.assert_called_once()
                mock_store.initialize.assert_called_once()
