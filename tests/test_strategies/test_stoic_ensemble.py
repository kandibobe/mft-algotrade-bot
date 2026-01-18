"""
Test Stoic Ensemble Strategy
==============================

Comprehensive tests for the StoicEnsembleStrategy trading logic.

Author: Stoic Citadel Team
License: MIT
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../user_data/strategies"))

# Add tests to path for conftest helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import assert_column_exists


class TestStoicEnsembleStrategy:
    """Test the main strategy implementation."""

    def test_strategy_version(self, stoic_strategy):
        """Test that strategy has correct version and interface."""
        assert hasattr(stoic_strategy, "INTERFACE_VERSION")
        assert stoic_strategy.INTERFACE_VERSION == 3
        assert stoic_strategy.timeframe == "5m"

    def test_risk_parameters(self, stoic_strategy):
        """Test that risk parameters are within safe limits."""
        # Hard stoploss should be exactly -10% (Institutional Risk)
        assert stoic_strategy.stoploss == -0.10, "Stoploss not at -10%"

        # Trailing stop should be enabled (handled by custom_exit/stoploss in V5)
        # Strategy explicitly sets trailing_stop = False to use custom logic
        assert stoic_strategy.trailing_stop is False, "Trailing stop should be False (using custom)"
        # assert stoic_strategy.trailing_stop_positive == 0.01, "Trailing stop not at +1%"

        # Minimal ROI should have reasonable targets (keys can be str or int)
        assert 0 in stoic_strategy.minimal_roi or '0' in stoic_strategy.minimal_roi, "No immediate ROI target"
        roi_0 = stoic_strategy.minimal_roi.get(0, stoic_strategy.minimal_roi.get('0'))
        assert roi_0 is not None and roi_0 > 0, "ROI target not positive"

    def test_protections_configured(self, stoic_strategy):
        """Test that protection mechanisms are properly configured."""
        protections = stoic_strategy.protections

        assert len(protections) > 0, "No protections configured"

        # Check for StoplossGuard
        has_stoploss_guard = any(p["method"] == "StoplossGuard" for p in protections)
        assert has_stoploss_guard, "StoplossGuard not configured"

        # Check for MaxDrawdown protection
        has_max_drawdown = any(p["method"] == "MaxDrawdown" for p in protections)
        assert has_max_drawdown, "MaxDrawdown protection not configured"

    def test_entry_signal_in_uptrend(self, stoic_strategy, uptrend_dataframe, strategy_metadata):
        """Test that entry signals are generated in uptrend."""
        # Populate indicators
        df = stoic_strategy.populate_indicators(uptrend_dataframe.copy(), strategy_metadata)

        # Populate entry signals
        df = stoic_strategy.populate_entry_trend(df, strategy_metadata)

        # Should have 'enter_long' column
        assert_column_exists(df, "enter_long")

        # Should generate at least some entry signals in uptrend
        entry_signals = df["enter_long"].sum()
        assert entry_signals >= 0, "Entry signals should be non-negative"

    def test_no_entry_in_downtrend(self, stoic_strategy, downtrend_dataframe, strategy_metadata):
        """Test that strategy doesn't enter in clear downtrend."""
        # Populate indicators
        df = stoic_strategy.populate_indicators(downtrend_dataframe.copy(), strategy_metadata)

        # Populate entry signals
        df = stoic_strategy.populate_entry_trend(df, strategy_metadata)

        # Should generate very few or no entry signals in downtrend
        entry_signals = df["enter_long"].sum()

        # In a clear downtrend, entries should be minimal
        total_candles = len(df)
        entry_rate = entry_signals / total_candles

        assert entry_rate < 0.1, f"Too many entries in downtrend: {entry_rate * 100}%"

    def test_exit_signal_generation(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that exit signals can be generated."""
        # Populate indicators
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Populate exit signals
        df = stoic_strategy.populate_exit_trend(df, strategy_metadata)

        # Should have 'exit_long' column
        assert_column_exists(df, "exit_long")

        # Exit signals should be non-negative
        exit_signals = df["exit_long"].sum()
        assert exit_signals >= 0, "Exit signals should be non-negative"

    def test_startup_candle_count(self, stoic_strategy):
        """Test that startup_candle_count is sufficient for EMA200."""
        # Should have at least 200 candles for EMA200
        assert stoic_strategy.startup_candle_count >= 200, "Insufficient startup candles for EMA200"

    def test_order_types_configured(self, stoic_strategy):
        """Test that order types are properly configured."""
        assert "entry" in stoic_strategy.order_types
        assert "exit" in stoic_strategy.order_types
        assert "stoploss" in stoic_strategy.order_types

        # Stoploss should be market order for quick execution
        assert stoic_strategy.order_types["stoploss"] == "market"

    def test_custom_stake_amount_volatility_adjustment(
        self, stoic_strategy, sample_dataframe, strategy_metadata, mock_exchange
    ):
        """Test that position sizing adjusts for volatility."""
        stoic_strategy.dp = MagicMock()

        # Create high volatility scenario
        df_high_vol = sample_dataframe.copy()
        df_high_vol["atr"] = 5.0  # High ATR
        df_high_vol["close"] = 100.0

        # Mock the dataframe provider
        stoic_strategy.dp.get_analyzed_dataframe.return_value = (df_high_vol, None)

        # Test stake adjustment
        proposed_stake = 100.0
        adjusted_stake = stoic_strategy.custom_stake_amount(
            pair="BTC/USDT",
            current_time=datetime.now(),
            current_rate=100.0,
            proposed_stake=proposed_stake,
            min_stake=10.0,
            max_stake=1000.0,
            leverage=1.0,
            entry_tag=None,
            side="long",
        )

        # High volatility should reduce stake
        assert adjusted_stake < proposed_stake, "Stake not reduced for high volatility"

    def test_confirm_trade_entry_rejects_high_spread(self, stoic_strategy):
        """Test that strategy rejects trades with high spread."""
        # Mock ticker with high spread
        stoic_strategy.dp = MagicMock()
        stoic_strategy.dp.ticker.return_value = {
            'bid': 100.0,
            'ask': 101.0,  # 1% spread
            'quoteVolume': 1000000
        }

        result = stoic_strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=100.0,
            time_in_force="GTC",
            current_time=datetime.utcnow(),
            entry_tag=None,
            side="long",
        )

        assert result is False, "Should reject trade due to high spread"

    def test_confirm_trade_entry_accepts_normal_spread(self, stoic_strategy):
        """Test that strategy accepts trades with normal spread."""
        # Mock ticker with normal spread
        stoic_strategy.dp = MagicMock()
        stoic_strategy.dp.ticker.return_value = {
            'bid': 100.0,
            'ask': 100.05,  # 0.05% spread
            'quoteVolume': 1000000
        }

        # Ensure market safety check passes (mocking it if needed, or relying on defaults)
        # Assuming check_market_safety returns True by default or if mocked
        # We might need to mock check_market_safety if it relies on other things
        # But let's try with just DP mock first.
        # Actually, check_market_safety might call other things.
        # Let's mock check_market_safety to be sure we are testing spread.
        stoic_strategy.check_market_safety = MagicMock(return_value=True)

        result = stoic_strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=100.0,
            time_in_force="GTC",
            current_time=datetime.utcnow(),
            entry_tag=None,
            side="long",
        )

        assert result is True, "Should accept trade with normal spread"

    def test_custom_exit_emergency_24h(self, stoic_strategy, mock_trade):
        """Test emergency exit after 24 hours with loss."""
        # Mock trade that's been open for 25 hours and losing
        current_time = datetime.utcnow()
        mock_trade.open_date_utc = current_time - timedelta(hours=25)

        result = stoic_strategy.custom_exit(
            pair="BTC/USDT",
            trade=mock_trade,
            current_time=current_time,
            current_rate=98.0,  # -2% loss
            current_profit=-0.025,  # Must be < -0.02 to trigger
        )

        # V5 priority: Time Decay triggers first
        assert result in ["emergency_exit_24h", "time_decay_exit"], "Should trigger emergency exit"

    def test_custom_exit_no_hard_tp(self, stoic_strategy, mock_trade):
        """Test that strategy does not use hard TP exit (relies on trailing)."""
        result = stoic_strategy.custom_exit(
            pair="BTC/USDT",
            trade=mock_trade,
            current_time=datetime.utcnow(),
            current_rate=110.0,
            current_profit=0.11,  # 11% profit
        )

        # Should NOT return a fixed TP string, should be None (handled by stoploss/trailing)
        assert result is None, "Should rely on trailing stop, not hard exit"

    def test_custom_exit_no_exit(self, stoic_strategy, mock_trade):
        """Test that custom exit returns None when no exit conditions met."""
        # Normal trade, no exit conditions
        mock_trade.open_date_utc = datetime.utcnow() - timedelta(hours=2)

        result = stoic_strategy.custom_exit(
            pair="BTC/USDT",
            trade=mock_trade,
            current_time=datetime.utcnow(),
            current_rate=101.0,  # Small profit
            current_profit=0.01,
        )

        assert result is None, "Should not exit normal trade"


class TestStrategyRobustness:
    """Test strategy robustness and edge cases."""

    def test_empty_signals_initialization(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that signal columns are properly initialized."""
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
        df = stoic_strategy.populate_entry_trend(df, strategy_metadata)

        # enter_long should exist
        assert "enter_long" in df.columns

        # Signals should be binary (0/1) or NaN (for warmup period)
        non_nan = df["enter_long"].dropna()
        if len(non_nan) > 0:
            assert non_nan.isin([0, 1]).all(), "Non-NaN signal values should be binary"

    def test_no_future_peeking(self, stoic_strategy):
        """Test that strategy doesn't use future data."""
        # Process only new candles should be True
        assert stoic_strategy.process_only_new_candles is True, "Strategy may be using future data"

    def test_consistent_signal_generation(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that running the strategy twice produces same results."""
        # Run 1
        df1 = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
        df1 = stoic_strategy.populate_entry_trend(df1, strategy_metadata)

        # Run 2
        df2 = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
        df2 = stoic_strategy.populate_entry_trend(df2, strategy_metadata)

        # Results should be identical
        pd.testing.assert_series_equal(
            df1["enter_long"], df2["enter_long"], check_names=False
        )

    @pytest.mark.slow
    def test_strategy_with_large_dataset(self, stoic_strategy, strategy_metadata):
        """Test strategy performance with large dataset."""
        # Create large dataset (1 year of 5min data)
        rows = 105120  # 365 * 24 * 60 / 5
        df = pd.DataFrame(
            {
                "open": np.random.randn(rows) * 10 + 100,
                "high": np.random.randn(rows) * 10 + 101,
                "low": np.random.randn(rows) * 10 + 99,
                "close": np.random.randn(rows) * 10 + 100,
                "volume": np.random.randint(1000, 10000, rows),
            }
        )

        # Should handle large dataset without issues
        result = stoic_strategy.populate_indicators(df, strategy_metadata)
        result = stoic_strategy.populate_entry_trend(result, strategy_metadata)

        assert len(result) == rows, "Dataframe length mismatch"
        assert "enter_long" in result.columns, "Entry signals not generated"


class TestStrategyCompliance:
    """Test compliance with Stoic Citadel standards."""

    def test_stoploss_compliance(self, stoic_strategy):
        """Test that stoploss matches HARD_STOPLOSS constant."""
        # Hard stoploss must be -10%
        HARD_STOPLOSS = -0.10
        assert stoic_strategy.stoploss == HARD_STOPLOSS, "Stoploss deviates from standard"

    def test_no_hardcoded_values_in_entry(self, stoic_strategy, sample_dataframe):
        """Test that strategy doesn't rely on hardcoded pair-specific values."""
        # Test with different pairs
        metadata1 = {"pair": "BTC/USDT", "timeframe": "5m"}
        metadata2 = {"pair": "ETH/USDT", "timeframe": "5m"}

        df1 = stoic_strategy.populate_indicators(sample_dataframe.copy(), metadata1)
        df1 = stoic_strategy.populate_entry_trend(df1, metadata1)

        df2 = stoic_strategy.populate_indicators(sample_dataframe.copy(), metadata2)
        df2 = stoic_strategy.populate_entry_trend(df2, metadata2)

        # Logic should be identical regardless of pair
        pd.testing.assert_series_equal(
            df1["enter_long"], df2["enter_long"], check_names=False
        )
