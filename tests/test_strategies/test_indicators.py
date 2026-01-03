"""
Test Indicator Calculation
===========================

Tests for technical indicators used in trading strategies.

Author: Stoic Citadel Team
License: MIT
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add strategies to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../user_data/strategies"))

# Add tests to path for conftest helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conftest import (
    assert_column_exists,
    assert_no_nan_in_column,
)


class TestIndicatorCalculation:
    """Test indicator calculation in strategies."""

    def test_basic_indicators_exist(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that basic indicators are calculated correctly."""
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Check that all required indicators exist
        required_indicators = [
            "ema_50",
            "ema_100",
            "ema_200",
            "rsi",
            "adx",
            "slowk",
            "slowd",
            "macd",
            "macd_signal",
            "bb_lower",
            "bb_upper",
            "atr",
            "volume_mean",
        ]

        for indicator in required_indicators:
            assert_column_exists(df, indicator, f"Missing indicator: {indicator}")

    def test_no_nan_in_indicators(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that indicators have minimal NaN values after warmup."""
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # After EMA200 warmup, these indicators should have no NaN
        indicators_to_check = ["ema_50", "ema_100", "rsi", "adx", "atr"]

        # Skip first 250 rows for warmup
        df_after_warmup = df.iloc[250:]

        for indicator in indicators_to_check:
            if indicator in df.columns:
                nan_count = df_after_warmup[indicator].isna().sum()
                assert nan_count == 0, f"Indicator {indicator} has {nan_count} NaN values after warmup"

    def test_rsi_bounds(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that RSI is within valid bounds (0-100)."""
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Skip NaN values at beginning
        rsi_valid = df["rsi"].dropna()

        assert rsi_valid.min() >= 0, "RSI below 0"
        assert rsi_valid.max() <= 100, "RSI above 100"

    def test_bollinger_bands_order(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that Bollinger Bands have correct order: lower < middle < upper."""
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Skip warmup period
        df_valid = df.iloc[250:].dropna(subset=["bb_lower", "bb_upper", "bb_middle"])

        assert (
            df_valid["bb_lower"] <= df_valid["bb_middle"]
        ).all(), "Lower band above middle"
        assert (
            df_valid["bb_middle"] <= df_valid["bb_upper"]
        ).all(), "Middle band above upper"

    def test_ema_trend_logic(self, stoic_strategy, uptrend_dataframe, strategy_metadata):
        """Test that EMAs reflect trend correctly."""
        df = stoic_strategy.populate_indicators(uptrend_dataframe.copy(), strategy_metadata)

        # In strong uptrend, shorter EMA should be above longer EMA
        df_valid = df.iloc[-100:]  # Last 100 candles of uptrend

        # EMA50 should be above EMA200 in uptrend most of the time
        ema_bullish = (df_valid["ema_50"] > df_valid["ema_200"]).sum()
        total = len(df_valid)

        assert ema_bullish / total > 0.7, "EMAs not reflecting uptrend"

    def test_volume_mean_calculation(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that volume mean is calculated correctly."""
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Volume mean should be positive
        volume_mean_valid = df["volume_mean"].dropna()
        assert (volume_mean_valid > 0).all(), "Volume mean should be positive"

    def test_trend_score_calculation(self, stoic_strategy, uptrend_dataframe, strategy_metadata):
        """Test that trend score is calculated when present."""
        df = stoic_strategy.populate_indicators(uptrend_dataframe.copy(), strategy_metadata)

        if "trend_score" in df.columns:
            # Trend score should be reasonable
            trend_valid = df["trend_score"].dropna()
            assert len(trend_valid) > 0, "No valid trend scores"

    def test_atr_positive_values(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that ATR is always positive."""
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        atr_valid = df["atr"].dropna()
        assert (atr_valid >= 0).all(), "ATR should never be negative"


class TestIndicatorEdgeCases:
    """Test edge cases in indicator calculation."""

    def test_minimum_candles(self, stoic_strategy, strategy_metadata):
        """Test behavior with minimum number of candles."""
        # Create minimal dataset
        n = 250  # Just above EMA200 requirement
        np.random.seed(42)
        
        df = pd.DataFrame({
            "open": np.random.randn(n) * 10 + 100,
            "high": np.random.randn(n) * 10 + 101,
            "low": np.random.randn(n) * 10 + 99,
            "close": np.random.randn(n) * 10 + 100,
            "volume": np.random.randint(1000, 10000, n),
        })

        # Should not raise exception
        result = stoic_strategy.populate_indicators(df, strategy_metadata)
        assert len(result) == n

    def test_zero_volume_handling(self, stoic_strategy, strategy_metadata):
        """Test handling of zero volume candles."""
        np.random.seed(42)
        n = 300
        
        df = pd.DataFrame({
            "open": np.random.randn(n) * 10 + 100,
            "high": np.random.randn(n) * 10 + 101,
            "low": np.random.randn(n) * 10 + 99,
            "close": np.random.randn(n) * 10 + 100,
            "volume": np.random.randint(0, 10000, n),  # Some zero volumes
        })

        # Should handle gracefully
        result = stoic_strategy.populate_indicators(df, strategy_metadata)
        assert len(result) == n

    def test_flat_price_handling(self, stoic_strategy, strategy_metadata):
        """Test handling of flat price periods."""
        n = 300
        
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.5] * n,
            "low": [99.5] * n,
            "close": [100.0] * n,
            "volume": [1000] * n,
        })

        # Should handle gracefully without div by zero
        result = stoic_strategy.populate_indicators(df, strategy_metadata)
        assert len(result) == n
