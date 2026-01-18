"""
Tests for StoicLogic
"""
import numpy as np
import pandas as pd
import pytest

from src.strategies.core_logic import MarketRegime, StoicLogic


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing indicators."""
    length = 300
    dates = pd.date_range(start='2024-01-01', periods=length, freq='5min')
    df = pd.DataFrame({
        'date': dates,
        'open': 100 + np.random.randn(length),
        'high': 105 + np.random.randn(length),
        'low': 95 + np.random.randn(length),
        'close': 100 + np.cumsum(np.random.randn(length)),
        'volume': 1000 + np.random.randn(length) * 100
    })
    return df

class TestStoicLogic:

    def test_populate_indicators(self, sample_dataframe):
        """Test if indicators are populated correctly."""
        df = StoicLogic.populate_indicators(sample_dataframe)

        # Check if new columns exist
        assert 'ema_50' in df.columns
        assert 'ema_200' in df.columns
        assert 'rsi' in df.columns
        assert 'bb_lower' in df.columns
        assert 'bb_upper' in df.columns
        assert 'bb_width' in df.columns

        # Check values are not all NaN (after warm-up period)
        assert not df['ema_50'].iloc[-1:].isna().all()
        assert not df['rsi'].iloc[-1:].isna().all()

    def test_populate_entry_exit_signals_trend(self):
        """Test trend signal generation."""
        # Create synthetic data where Trend condition matches
        # Condition: Price > EMA200 AND ML > Threshold AND Regime = PUMP_DUMP
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'ema_200': [90, 90, 90, 90, 90],
            'ml_prediction': [0.8, 0.8, 0.8, 0.8, 0.8],
            'rsi': [50, 50, 50, 50, 50], # Not used for trend
            'bb_lower': [80, 80, 80, 80, 80], # Not used for trend
            'regime': [MarketRegime.PUMP_DUMP.value] * 5,
            'vol_zscore': [2.0] * 5
        })

        result_df = StoicLogic.populate_entry_exit_signals(df, persistence_window=3)

        # Last candle should have entry signal
        assert result_df['enter_long'].iloc[-1] == 1

    def test_populate_entry_exit_signals_persistence(self):
        """Test that signal requires persistence."""
        # Condition met only for last 1 candle (persistence=3)
        df = pd.DataFrame({
            'close': [90, 90, 90, 90, 110], # Jump above EMA200 at end
            'ema_200': [100, 100, 100, 100, 100],
            'ml_prediction': [0.8] * 5,
            'regime': [MarketRegime.PUMP_DUMP.value] * 5,
            'rsi': [50] * 5,
            'bb_lower': [80] * 5,
             'vol_zscore': [2.0] * 5
        })

        result_df = StoicLogic.populate_entry_exit_signals(df, persistence_window=3)

        # Should NOT enter because persistence requirement not met
        assert result_df['enter_long'].iloc[-1] == 0

    def test_get_entry_decision_scalar(self):
        """Test scalar decision logic."""

        # 1. Quiet Chop -> Stay Flat
        decision = StoicLogic.get_entry_decision({}, MarketRegime.QUIET_CHOP)
        assert decision.should_enter_long is False

        # 2. Pump Dump -> Trend Entry
        candle = {
            'close': 100,
            'ema_200': 90,
            'ml_prediction': 0.8
        }
        decision = StoicLogic.get_entry_decision(candle, MarketRegime.PUMP_DUMP)
        assert decision.should_enter_long is True
        assert "Trend Follow" in decision.reason

        # 3. Violent Chop -> Mean Reversion
        candle_mr = {
            'rsi': 25,
            'close': 95,
            'bb_lower': 96
        }
        decision = StoicLogic.get_entry_decision(candle_mr, MarketRegime.VIOLENT_CHOP)
        assert decision.should_enter_long is True
        assert "Mean Reversion" in decision.reason
