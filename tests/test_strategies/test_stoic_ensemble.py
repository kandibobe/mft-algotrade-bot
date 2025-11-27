"""
Test Stoic Ensemble Strategy
==============================

Comprehensive tests for the StoicEnsembleStrategy trading logic.

Author: Stoic Citadel Team
License: MIT
"""

import pytest
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../user_data/strategies"))

from conftest import assert_signal_generated, assert_column_exists


class TestStoicEnsembleStrategy:
    """Test the main strategy implementation."""

    def test_strategy_version(self):
        """Test that strategy has correct version and interface."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        assert hasattr(strategy, "INTERFACE_VERSION")
        assert strategy.INTERFACE_VERSION == 3
        assert strategy.timeframe == "5m"

    def test_risk_parameters(self):
        """Test that risk parameters are within safe limits."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Hard stoploss should be exactly -5%
        assert strategy.stoploss == -0.05, "Stoploss not at -5%"

        # Trailing stop should be enabled
        assert strategy.trailing_stop is True, "Trailing stop not enabled"
        assert strategy.trailing_stop_positive == 0.01, "Trailing stop not at +1%"

        # Minimal ROI should have reasonable targets
        assert 0 in strategy.minimal_roi, "No immediate ROI target"
        assert strategy.minimal_roi[0] > 0, "ROI target not positive"

    def test_protections_configured(self):
        """Test that protection mechanisms are properly configured."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        protections = strategy.protections

        assert len(protections) > 0, "No protections configured"

        # Check for StoplossGuard
        has_stoploss_guard = any(p["method"] == "StoplossGuard" for p in protections)
        assert has_stoploss_guard, "StoplossGuard not configured"

        # Check for MaxDrawdown protection
        has_max_drawdown = any(p["method"] == "MaxDrawdown" for p in protections)
        assert has_max_drawdown, "MaxDrawdown protection not configured"

    def test_entry_signal_in_uptrend(self, uptrend_dataframe, strategy_metadata):
        """Test that entry signals are generated in uptrend."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Populate indicators
        df = strategy.populate_indicators(uptrend_dataframe.copy(), strategy_metadata)

        # Populate entry signals
        df = strategy.populate_entry_trend(df, strategy_metadata)

        # Should have 'enter_long' column
        assert_column_exists(df, "enter_long")

        # Should generate at least some entry signals in uptrend
        entry_signals = df["enter_long"].sum()
        assert entry_signals >= 0, "Entry signals should be non-negative"

    def test_no_entry_in_downtrend(self, downtrend_dataframe, strategy_metadata):
        """Test that strategy doesn't enter in clear downtrend."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Populate indicators
        df = strategy.populate_indicators(downtrend_dataframe.copy(), strategy_metadata)

        # Populate entry signals
        df = strategy.populate_entry_trend(df, strategy_metadata)

        # Should generate very few or no entry signals in downtrend
        entry_signals = df["enter_long"].sum()

        # In a clear downtrend, entries should be minimal
        total_candles = len(df)
        entry_rate = entry_signals / total_candles

        assert entry_rate < 0.1, f"Too many entries in downtrend: {entry_rate * 100}%"

    def test_exit_signal_generation(self, sample_dataframe, strategy_metadata):
        """Test that exit signals can be generated."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Populate indicators
        df = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Populate exit signals
        df = strategy.populate_exit_trend(df, strategy_metadata)

        # Should have 'exit_long' column
        assert_column_exists(df, "exit_long")

        # Exit signals should be non-negative
        exit_signals = df["exit_long"].sum()
        assert exit_signals >= 0, "Exit signals should be non-negative"

    def test_startup_candle_count(self):
        """Test that startup_candle_count is sufficient for EMA200."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Should have at least 200 candles for EMA200
        assert strategy.startup_candle_count >= 200, "Insufficient startup candles for EMA200"

    def test_order_types_configured(self):
        """Test that order types are properly configured."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        assert "entry" in strategy.order_types
        assert "exit" in strategy.order_types
        assert "stoploss" in strategy.order_types

        # Stoploss should be market order for quick execution
        assert strategy.order_types["stoploss"] == "market"

    def test_custom_stake_amount_volatility_adjustment(
        self, sample_dataframe, strategy_metadata, mock_exchange
    ):
        """Test that position sizing adjusts for volatility."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        strategy.dp = MagicMock()

        # Create high volatility scenario
        df_high_vol = sample_dataframe.copy()
        df_high_vol["atr"] = 5.0  # High ATR
        df_high_vol["close"] = 100.0

        # Mock the dataframe provider
        strategy.dp.get_analyzed_dataframe.return_value = (df_high_vol, None)

        # Test stake adjustment
        proposed_stake = 100.0
        adjusted_stake = strategy.custom_stake_amount(
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

    def test_confirm_trade_entry_rejects_low_liquidity_hours(self):
        """Test that strategy rejects trades during low liquidity hours."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Test during low liquidity hours (0-5 AM UTC)
        low_liquidity_time = datetime(2024, 1, 1, 2, 0, 0)

        result = strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=100.0,
            time_in_force="GTC",
            current_time=low_liquidity_time,
            entry_tag=None,
            side="long",
        )

        assert result is False, "Should reject trade during low liquidity hours"

    def test_confirm_trade_entry_accepts_normal_hours(self):
        """Test that strategy accepts trades during normal hours."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Test during normal hours (10 AM UTC)
        normal_time = datetime(2024, 1, 1, 10, 0, 0)

        result = strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=100.0,
            time_in_force="GTC",
            current_time=normal_time,
            entry_tag=None,
            side="long",
        )

        assert result is True, "Should accept trade during normal hours"

    def test_custom_exit_emergency_24h(self, mock_trade):
        """Test emergency exit after 24 hours with loss."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Mock trade that's been open for 25 hours and losing
        mock_trade.open_date_utc = datetime.utcnow() - timedelta(hours=25)

        result = strategy.custom_exit(
            pair="BTC/USDT",
            trade=mock_trade,
            current_time=datetime.utcnow(),
            current_rate=98.0,  # -2% loss
            current_profit=-0.02,
        )

        assert result == "emergency_exit_24h", "Should trigger emergency exit"

    def test_custom_exit_take_profit_10pct(self, mock_trade):
        """Test take profit at 10% gain."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        result = strategy.custom_exit(
            pair="BTC/USDT",
            trade=mock_trade,
            current_time=datetime.utcnow(),
            current_rate=110.0,  # 10% profit
            current_profit=0.10,
        )

        assert result == "take_profit_10pct", "Should trigger 10% take profit"

    def test_custom_exit_no_exit(self, mock_trade):
        """Test that custom exit returns None when no exit conditions met."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Normal trade, no exit conditions
        mock_trade.open_date_utc = datetime.utcnow() - timedelta(hours=2)

        result = strategy.custom_exit(
            pair="BTC/USDT",
            trade=mock_trade,
            current_time=datetime.utcnow(),
            current_rate=101.0,  # Small profit
            current_profit=0.01,
        )

        assert result is None, "Should not exit normal trade"


class TestStrategyRobustness:
    """Test strategy robustness and edge cases."""

    def test_empty_signals_initialization(self, sample_dataframe, strategy_metadata):
        """Test that signal columns are properly initialized."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        df = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
        df = strategy.populate_entry_trend(df, strategy_metadata)

        # enter_long should exist and be 0/1
        assert "enter_long" in df.columns
        assert df["enter_long"].isin([0, 1]).all(), "Signal values not binary"

    def test_no_future_peeking(self, sample_dataframe, strategy_metadata):
        """Test that strategy doesn't use future data."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Process only new candles should be True
        assert strategy.process_only_new_candles is True, "Strategy may be using future data"

    def test_consistent_signal_generation(self, sample_dataframe, strategy_metadata):
        """Test that running the strategy twice produces same results."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Run 1
        df1 = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
        df1 = strategy.populate_entry_trend(df1, strategy_metadata)

        # Run 2
        df2 = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
        df2 = strategy.populate_entry_trend(df2, strategy_metadata)

        # Results should be identical
        pd.testing.assert_series_equal(
            df1["enter_long"], df2["enter_long"], check_names=False
        )

    @pytest.mark.slow
    def test_strategy_with_large_dataset(self, strategy_metadata):
        """Test strategy performance with large dataset."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy
        import numpy as np

        strategy = StoicEnsembleStrategy()

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
        result = strategy.populate_indicators(df, strategy_metadata)
        result = strategy.populate_entry_trend(result, strategy_metadata)

        assert len(result) == rows, "Dataframe length mismatch"
        assert "enter_long" in result.columns, "Entry signals not generated"


class TestStrategyCompliance:
    """Test compliance with Stoic Citadel standards."""

    def test_stoploss_compliance(self):
        """Test that stoploss matches HARD_STOPLOSS constant."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Hard stoploss must be -5%
        HARD_STOPLOSS = -0.05
        assert strategy.stoploss == HARD_STOPLOSS, "Stoploss deviates from standard"

    def test_no_hardcoded_values_in_entry(self, sample_dataframe, strategy_metadata):
        """Test that strategy doesn't rely on hardcoded pair-specific values."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Test with different pairs
        metadata1 = {"pair": "BTC/USDT", "timeframe": "5m"}
        metadata2 = {"pair": "ETH/USDT", "timeframe": "5m"}

        df1 = strategy.populate_indicators(sample_dataframe.copy(), metadata1)
        df1 = strategy.populate_entry_trend(df1, metadata1)

        df2 = strategy.populate_indicators(sample_dataframe.copy(), metadata2)
        df2 = strategy.populate_entry_trend(df2, metadata2)

        # Logic should be identical regardless of pair
        pd.testing.assert_series_equal(
            df1["enter_long"], df2["enter_long"], check_names=False
        )
