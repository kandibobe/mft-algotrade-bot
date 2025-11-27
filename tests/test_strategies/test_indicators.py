"""
Test Indicator Calculation
===========================

Tests for technical indicators used in trading strategies.

Author: Stoic Citadel Team
License: MIT
"""

import pytest
import pandas as pd
import sys
import os

# Add strategies to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../user_data/strategies"))

from conftest import (
    assert_column_exists,
    assert_no_nan_in_column,
)


class TestIndicatorCalculation:
    """Test indicator calculation in strategies."""

    def test_basic_indicators_exist(self, sample_dataframe, strategy_metadata):
        """Test that basic indicators are calculated correctly."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

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
            "macdsignal",
            "bb_lowerband",
            "bb_upperband",
            "atr",
            "volume_mean",
        ]

        for indicator in required_indicators:
            assert_column_exists(df, indicator)

    def test_no_nan_in_indicators(self, sample_dataframe, strategy_metadata):
        """Test that indicators don't have NaN values after warmup period."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # After 200 candles, EMA200 and other indicators should have valid values
        # We skip the first 200 rows for warmup
        skip_rows = strategy.startup_candle_count

        # These indicators should not have NaN after warmup
        indicators_to_check = [
            "ema_50",
            "ema_100",
            "rsi",
            "adx",
            "atr",
        ]

        for indicator in indicators_to_check:
            if len(df) > skip_rows:
                assert_no_nan_in_column(df, indicator, skip_first=skip_rows)

    def test_rsi_bounds(self, sample_dataframe, strategy_metadata):
        """Test that RSI values are within valid bounds [0, 100]."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # RSI should be between 0 and 100
        rsi_values = df["rsi"].dropna()
        assert (rsi_values >= 0).all(), "RSI has values below 0"
        assert (rsi_values <= 100).all(), "RSI has values above 100"

    def test_bollinger_bands_order(self, sample_dataframe, strategy_metadata):
        """Test that Bollinger Bands maintain correct order (lower < middle < upper)."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Remove NaN values
        df_valid = df.dropna(subset=["bb_lowerband", "bb_middleband", "bb_upperband"])

        # Check order: lower < middle < upper
        assert (
            df_valid["bb_lowerband"] <= df_valid["bb_middleband"]
        ).all(), "Lower band above middle band"
        assert (
            df_valid["bb_middleband"] <= df_valid["bb_upperband"]
        ).all(), "Middle band above upper band"

    def test_ema_trend_logic(self, uptrend_dataframe, strategy_metadata):
        """Test that EMAs follow trend direction in uptrend."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = strategy.populate_indicators(uptrend_dataframe.copy(), strategy_metadata)

        # In a strong uptrend, price should be above EMA50 most of the time
        df_valid = df.iloc[150:]  # Look at recent data
        above_ema50 = (df_valid["close"] > df_valid["ema_50"]).sum()
        total_candles = len(df_valid)

        # At least 60% of candles should be above EMA50 in uptrend
        assert above_ema50 / total_candles > 0.6, "Price not consistently above EMA50 in uptrend"

    def test_volume_mean_calculation(self, sample_dataframe, strategy_metadata):
        """Test that volume mean is calculated correctly."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Volume mean should be positive
        volume_mean = df["volume_mean"].dropna()
        assert (volume_mean > 0).all(), "Volume mean has non-positive values"

    def test_trend_score_calculation(self, uptrend_dataframe, strategy_metadata):
        """Test custom trend score indicator."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = strategy.populate_indicators(uptrend_dataframe.copy(), strategy_metadata)

        # Trend score should be between 0 and 3
        trend_scores = df["trend_score"].dropna()
        assert (trend_scores >= 0).all(), "Trend score below 0"
        assert (trend_scores <= 3).all(), "Trend score above 3"

    def test_atr_positive_values(self, sample_dataframe, strategy_metadata):
        """Test that ATR (volatility) is always positive."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        atr_values = df["atr"].dropna()
        assert (atr_values > 0).all(), "ATR has non-positive values"

    @pytest.mark.slow
    def test_indicator_performance(self, sample_dataframe, strategy_metadata, benchmark):
        """Benchmark indicator calculation performance."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Benchmark the populate_indicators method
        def run_indicators():
            return strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # This should complete in reasonable time
        if "benchmark" in dir():
            result = benchmark(run_indicators)
        else:
            # If pytest-benchmark not installed, just run it once
            result = run_indicators()

        assert len(result) == len(sample_dataframe), "Dataframe length changed"


class TestIndicatorEdgeCases:
    """Test edge cases in indicator calculation."""

    def test_minimum_candles(self, strategy_metadata):
        """Test behavior with minimum required candles."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Create minimal dataframe (exactly startup_candle_count)
        rows = strategy.startup_candle_count
        df = pd.DataFrame(
            {
                "open": [100.0] * rows,
                "high": [101.0] * rows,
                "low": [99.0] * rows,
                "close": [100.0] * rows,
                "volume": [1000] * rows,
            }
        )

        # Should not raise an error
        result = strategy.populate_indicators(df, strategy_metadata)
        assert len(result) == rows

    def test_zero_volume_handling(self, sample_dataframe, strategy_metadata):
        """Test handling of zero volume candles."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()
        df = sample_dataframe.copy()

        # Set some volume values to zero
        df.loc[df.index[10:15], "volume"] = 0

        # Should not crash
        result = strategy.populate_indicators(df, strategy_metadata)
        assert len(result) == len(df)

    def test_flat_price_handling(self, strategy_metadata):
        """Test handling of completely flat price action."""
        from StoicEnsembleStrategy import StoicEnsembleStrategy

        strategy = StoicEnsembleStrategy()

        # Create flat price data
        df = pd.DataFrame(
            {
                "open": [100.0] * 200,
                "high": [100.0] * 200,
                "low": [100.0] * 200,
                "close": [100.0] * 200,
                "volume": [1000] * 200,
            }
        )

        # Should handle flat prices gracefully
        result = strategy.populate_indicators(df, strategy_metadata)

        # ADX should be low (no trend)
        adx_mean = result["adx"].dropna().mean()
        assert adx_mean < 15, "ADX too high for flat market"
