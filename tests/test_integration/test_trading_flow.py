"""
Integration Tests for Trading Flow
====================================

End-to-end tests for the complete trading workflow.

Author: Stoic Citadel Team
License: MIT
"""

import pytest
import sys
import os
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../user_data/strategies"))


@pytest.mark.integration
class TestTradingFlow:
    """Integration tests for complete trading workflow."""

    def test_complete_strategy_workflow(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test complete workflow: indicators -> entry -> exit."""
        # Step 1: Populate indicators
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
        assert len(df) > 0, "Indicators failed to populate"

        # Step 2: Generate entry signals
        df = stoic_strategy.populate_entry_trend(df, strategy_metadata)
        assert "enter_long" in df.columns, "Entry signals not generated"

        # Step 3: Generate exit signals
        df = stoic_strategy.populate_exit_trend(df, strategy_metadata)
        assert "exit_long" in df.columns, "Exit signals not generated"

        # Verify dataframe integrity
        assert not df.empty, "Dataframe is empty"
        assert len(df) == len(sample_dataframe), "Dataframe length changed"

    def test_strategy_with_minimal_config(self, stoic_strategy, sample_dataframe):
        """Test strategy works with minimal configuration."""
        metadata = {"pair": "BTC/USDT", "timeframe": "5m"}

        # Should work without errors
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), metadata)
        df = stoic_strategy.populate_entry_trend(df, metadata)
        df = stoic_strategy.populate_exit_trend(df, metadata)

        assert len(df) > 0, "Strategy failed with minimal config"

    def test_multiple_pairs_processing(self, stoic_strategy, sample_dataframe):
        """Test processing multiple pairs sequentially."""
        pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

        results = {}
        for pair in pairs:
            metadata = {"pair": pair, "timeframe": "5m"}
            df = stoic_strategy.populate_indicators(sample_dataframe.copy(), metadata)
            df = stoic_strategy.populate_entry_trend(df, metadata)
            results[pair] = df

        # All pairs should be processed successfully
        assert len(results) == len(pairs), "Not all pairs processed"
        for pair, df in results.items():
            assert "enter_long" in df.columns, f"Signals missing for {pair}"

    def test_protection_mechanisms_integration(self, stoic_strategy):
        """Test that protection mechanisms are properly integrated."""
        protections = stoic_strategy.protections

        # Verify all protections have required fields
        for protection in protections:
            assert "method" in protection, "Protection missing method"
            assert "lookback_period_candles" in protection, "Missing lookback period"
            assert "stop_duration_candles" in protection, "Missing stop duration"


@pytest.mark.integration
class TestDataIntegrity:
    """Test data integrity throughout the pipeline."""

    def test_no_data_corruption(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that original data is not corrupted during processing."""
        original_df = sample_dataframe.copy()

        # Process data
        result_df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)

        # Original OHLCV columns should be unchanged
        for col in ["open", "high", "low", "close", "volume"]:
            pd.testing.assert_series_equal(
                original_df[col],
                result_df[col],
                check_names=False,
                obj=f"Column {col} was modified",
            )

    def test_signal_consistency_across_runs(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test that signals are consistent across multiple runs."""
        # Run 3 times
        runs = []
        for _ in range(3):
            df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
            df = stoic_strategy.populate_entry_trend(df, strategy_metadata)
            runs.append(df["enter_long"].copy())

        # All runs should produce identical signals
        for i in range(1, len(runs)):
            pd.testing.assert_series_equal(
                runs[0], runs[i], check_names=False, obj=f"Run {i} differs from run 0"
            )


@pytest.mark.integration
class TestRiskManagement:
    """Integration tests for risk management features."""

    def test_stoploss_enforcement(self, stoic_strategy, mock_trade):
        """Test that stoploss is properly enforced."""
        # Verify stoploss is set
        assert stoic_strategy.stoploss == -0.05, "Stoploss not at -5%"

        # Mock trade should respect stoploss
        mock_trade.stop_loss = mock_trade.open_rate * (1 + stoic_strategy.stoploss)
        expected_stop = 50000.0 * 0.95  # -5%

        assert abs(mock_trade.stop_loss - expected_stop) < 1.0, "Stoploss not enforced"

    def test_position_sizing_integration(self, stoic_strategy, sample_dataframe, strategy_metadata):
        """Test position sizing with real data."""
        stoic_strategy.dp = MagicMock()

        # Populate indicators to get ATR
        df = stoic_strategy.populate_indicators(sample_dataframe.copy(), strategy_metadata)
        stoic_strategy.dp.get_analyzed_dataframe.return_value = (df, None)

        # Test position sizing
        stake = stoic_strategy.custom_stake_amount(
            pair="BTC/USDT",
            current_time=datetime.now(),
            current_rate=100.0,
            proposed_stake=100.0,
            min_stake=10.0,
            max_stake=1000.0,
            leverage=1.0,
            entry_tag=None,
            side="long",
        )

        # Stake should be within bounds
        assert 10.0 <= stake <= 1000.0, "Stake outside allowed bounds"

    def test_trailing_stop_configuration(self, stoic_strategy):
        """Test trailing stop is properly configured."""
        assert stoic_strategy.trailing_stop is True, "Trailing stop not enabled"
        assert stoic_strategy.trailing_stop_positive == 0.01, "Trailing stop trigger incorrect"
        assert (
            stoic_strategy.trailing_stop_positive_offset == 0.015
        ), "Trailing stop offset incorrect"


@pytest.mark.integration
@pytest.mark.slow
class TestBacktestCompatibility:
    """Test compatibility with Freqtrade backtesting."""

    def test_strategy_has_required_methods(self, stoic_strategy):
        """Test that strategy implements all required Freqtrade methods."""
        # Required methods for Freqtrade
        required_methods = [
            "populate_indicators",
            "populate_entry_trend",
            "populate_exit_trend",
        ]

        for method in required_methods:
            assert hasattr(stoic_strategy, method), f"Missing required method: {method}"
            assert callable(
                getattr(stoic_strategy, method)
            ), f"Method {method} is not callable"

    def test_strategy_metadata_compliance(self, stoic_strategy):
        """Test that strategy metadata meets Freqtrade requirements."""
        # Required attributes
        assert hasattr(stoic_strategy, "INTERFACE_VERSION"), "Missing INTERFACE_VERSION"
        assert hasattr(stoic_strategy, "timeframe"), "Missing timeframe"
        assert hasattr(stoic_strategy, "stoploss"), "Missing stoploss"
        assert hasattr(stoic_strategy, "minimal_roi"), "Missing minimal_roi"

        # Type checks
        assert isinstance(stoic_strategy.INTERFACE_VERSION, int), "INTERFACE_VERSION not int"
        assert isinstance(stoic_strategy.timeframe, str), "timeframe not string"
        assert isinstance(stoic_strategy.stoploss, float), "stoploss not float"
        assert isinstance(stoic_strategy.minimal_roi, dict), "minimal_roi not dict"

    def test_order_types_compatibility(self, stoic_strategy):
        """Test that order types are Freqtrade-compatible."""
        assert hasattr(stoic_strategy, "order_types"), "Missing order_types"
        order_types = stoic_strategy.order_types

        # Valid order type values
        valid_types = ["limit", "market"]

        assert order_types["entry"] in valid_types, "Invalid entry order type"
        assert order_types["exit"] in valid_types, "Invalid exit order type"
        assert order_types["stoploss"] in valid_types, "Invalid stoploss order type"


@pytest.mark.integration
class TestEnvironmentIntegration:
    """Test integration with Docker environment."""

    def test_import_dependencies(self):
        """Test that all required dependencies can be imported."""
        try:
            import pandas as pd
            import numpy as np
            import talib
            import pandas_ta as pta

            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import required dependency: {e}")

    def test_strategy_can_be_imported(self, minimal_config):
        """Test that strategy can be imported without errors."""
        try:
            from StoicEnsembleStrategy import StoicEnsembleStrategy

            strategy = StoicEnsembleStrategy(minimal_config)
            assert strategy is not None
        except Exception as e:
            pytest.fail(f"Failed to import strategy: {e}")

    def test_talib_indicators_available(self):
        """Test that TA-Lib indicators are available."""
        import talib

        # Check key indicators used in strategy
        required_indicators = [
            "EMA",
            "RSI",
            "ADX",
            "STOCH",
            "MACD",
            "BBANDS",
            "ATR",
        ]

        for indicator in required_indicators:
            assert hasattr(talib, indicator), f"TA-Lib missing indicator: {indicator}"
