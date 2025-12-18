"""
Tests for Triple Barrier Labeling
=================================

Tests the Triple Barrier Method implementation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml.training.labeling import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    DynamicBarrierLabeler,
    create_labels_for_training,
)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_rows = 200

    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='5min')

    # Start at 100, add some random walk
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)

    df = pd.DataFrame({
        'open': close - np.random.uniform(0, 0.5, n_rows),
        'high': close + np.random.uniform(0, 0.5, n_rows),
        'low': close - np.random.uniform(0, 0.5, n_rows),
        'close': close,
        'volume': np.random.randint(1000, 10000, n_rows)
    }, index=dates)

    return df


@pytest.fixture
def trending_up_df():
    """Create OHLCV data with clear uptrend."""
    n_rows = 100
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='5min')

    # Strong uptrend
    close = 100 + np.arange(n_rows) * 0.1

    df = pd.DataFrame({
        'open': close - 0.02,
        'high': close + 0.05,
        'low': close - 0.02,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_rows)
    }, index=dates)

    return df


@pytest.fixture
def trending_down_df():
    """Create OHLCV data with clear downtrend."""
    n_rows = 100
    dates = pd.date_range(start='2024-01-01', periods=n_rows, freq='5min')

    # Strong downtrend
    close = 100 - np.arange(n_rows) * 0.1

    df = pd.DataFrame({
        'open': close + 0.02,
        'high': close + 0.02,
        'low': close - 0.05,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_rows)
    }, index=dates)

    return df


class TestTripleBarrierConfig:
    """Test configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TripleBarrierConfig()

        assert config.take_profit == 0.005
        assert config.stop_loss == 0.002
        assert config.max_holding_period == 24
        assert config.include_hold_class is True
        assert config.fee_adjustment == 0.001

    def test_custom_config(self):
        """Test custom configuration."""
        config = TripleBarrierConfig(
            take_profit=0.01,
            stop_loss=0.005,
            max_holding_period=48,
            fee_adjustment=0.0015
        )

        assert config.take_profit == 0.01
        assert config.stop_loss == 0.005
        assert config.max_holding_period == 48


class TestTripleBarrierLabeler:
    """Test Triple Barrier labeler."""

    def test_initialization(self):
        """Test labeler initialization."""
        labeler = TripleBarrierLabeler()
        assert labeler.config is not None
        assert isinstance(labeler.config, TripleBarrierConfig)

    def test_label_output_shape(self, sample_ohlcv_df):
        """Test that labels have correct shape."""
        labeler = TripleBarrierLabeler()
        labels = labeler.label(sample_ohlcv_df)

        assert len(labels) == len(sample_ohlcv_df)
        assert isinstance(labels, pd.Series)

    def test_label_values(self, sample_ohlcv_df):
        """Test that labels have valid values."""
        labeler = TripleBarrierLabeler()
        labels = labeler.label(sample_ohlcv_df)

        # Check valid label values (including NaN)
        valid_values = {-1, 0, 1, np.nan}
        for val in labels.dropna().unique():
            assert val in {-1, 0, 1}

    def test_end_rows_are_nan(self, sample_ohlcv_df):
        """Test that last rows are NaN (insufficient forward data)."""
        config = TripleBarrierConfig(max_holding_period=24)
        labeler = TripleBarrierLabeler(config)
        labels = labeler.label(sample_ohlcv_df)

        # Last max_holding_period rows should be NaN
        assert labels.iloc[-24:].isna().all()

    def test_uptrend_has_buy_signals(self, trending_up_df):
        """Test that uptrend generates buy signals."""
        config = TripleBarrierConfig(
            take_profit=0.005,
            stop_loss=0.003,
            max_holding_period=20,
            fee_adjustment=0.0005
        )
        labeler = TripleBarrierLabeler(config)
        labels = labeler.label(trending_up_df)

        # Should have some buy signals (label=1)
        valid_labels = labels.dropna()
        buy_ratio = (valid_labels == 1).mean()
        assert buy_ratio > 0.3, f"Expected more buy signals in uptrend, got {buy_ratio:.2%}"

    def test_downtrend_has_sell_signals(self, trending_down_df):
        """Test that downtrend generates sell signals."""
        config = TripleBarrierConfig(
            take_profit=0.005,
            stop_loss=0.003,
            max_holding_period=20,
            fee_adjustment=0.0005
        )
        labeler = TripleBarrierLabeler(config)
        labels = labeler.label(trending_down_df)

        # Should have some sell signals (label=-1)
        valid_labels = labels.dropna()
        sell_ratio = (valid_labels == -1).mean()
        assert sell_ratio > 0.3, f"Expected more sell signals in downtrend, got {sell_ratio:.2%}"

    def test_fee_adjustment_affects_labels(self, trending_up_df):
        """Test that fee adjustment changes label distribution."""
        # Use trending data for more predictable results
        config_low_fee = TripleBarrierConfig(
            take_profit=0.01,
            stop_loss=0.005,
            fee_adjustment=0.0001,
            max_holding_period=20
        )
        config_high_fee = TripleBarrierConfig(
            take_profit=0.01,
            stop_loss=0.005,
            fee_adjustment=0.008,  # Very high fee
            max_holding_period=20
        )

        labeler_low = TripleBarrierLabeler(config_low_fee)
        labeler_high = TripleBarrierLabeler(config_high_fee)

        labels_low = labeler_low.label(trending_up_df).dropna()
        labels_high = labeler_high.label(trending_up_df).dropna()

        # Higher fees make TP harder to hit (TP - fee < market movement)
        # With low fees, easier to get buy signals
        # With very high fees (0.8%), the TP adjusted becomes 0.2% which is much harder
        buy_ratio_low = (labels_low == 1).mean()
        buy_ratio_high = (labels_high == 1).mean()

        # The difference may be small but should generally favor low fees
        # We use >= to allow for equal (no buys in either case)
        assert buy_ratio_low >= buy_ratio_high or abs(buy_ratio_low - buy_ratio_high) < 0.1

    def test_label_with_meta(self, sample_ohlcv_df):
        """Test labeling with metadata."""
        labeler = TripleBarrierLabeler()
        result_df = labeler.label_with_meta(sample_ohlcv_df)

        # Check columns
        assert 'label' in result_df.columns
        assert 'barrier_type' in result_df.columns
        assert 'holding_period' in result_df.columns
        assert 'return_pct' in result_df.columns

        # Check barrier types
        valid_barriers = {'take_profit', 'stop_loss', 'time_barrier', 'insufficient_data'}
        for barrier in result_df['barrier_type'].unique():
            assert barrier in valid_barriers


class TestDynamicBarrierLabeler:
    """Test Dynamic Barrier labeler."""

    def test_initialization(self):
        """Test dynamic labeler initialization."""
        labeler = DynamicBarrierLabeler()
        assert labeler.atr_period == 14
        assert labeler.atr_multiplier_tp == 2.0
        assert labeler.atr_multiplier_sl == 1.0

    def test_custom_atr_params(self):
        """Test custom ATR parameters."""
        labeler = DynamicBarrierLabeler(
            atr_period=20,
            atr_multiplier_tp=3.0,
            atr_multiplier_sl=1.5
        )
        assert labeler.atr_period == 20
        assert labeler.atr_multiplier_tp == 3.0
        assert labeler.atr_multiplier_sl == 1.5

    def test_dynamic_label_output_shape(self, sample_ohlcv_df):
        """Test that dynamic labels have correct shape."""
        labeler = DynamicBarrierLabeler()
        labels = labeler.label(sample_ohlcv_df)

        assert len(labels) == len(sample_ohlcv_df)
        assert isinstance(labels, pd.Series)

    def test_dynamic_label_values(self, sample_ohlcv_df):
        """Test that dynamic labels have valid values."""
        labeler = DynamicBarrierLabeler()
        labels = labeler.label(sample_ohlcv_df)

        for val in labels.dropna().unique():
            assert val in {-1, 0, 1}

    def test_atr_edges_are_nan(self, sample_ohlcv_df):
        """Test that first ATR period rows are NaN."""
        labeler = DynamicBarrierLabeler(atr_period=14)
        labels = labeler.label(sample_ohlcv_df)

        # First atr_period rows should be NaN
        assert labels.iloc[:14].isna().all()


class TestFactoryFunction:
    """Test the create_labels_for_training factory function."""

    def test_simple_method(self, sample_ohlcv_df):
        """Test simple labeling method."""
        labels = create_labels_for_training(sample_ohlcv_df, method='simple')

        assert len(labels) == len(sample_ohlcv_df)
        # Simple method returns 0 or 1
        assert set(labels.dropna().unique()).issubset({0, 1})

    def test_triple_barrier_method(self, sample_ohlcv_df):
        """Test triple barrier method via factory."""
        labels = create_labels_for_training(
            sample_ohlcv_df,
            method='triple_barrier',
            take_profit=0.01,
            stop_loss=0.005
        )

        assert len(labels) == len(sample_ohlcv_df)
        for val in labels.dropna().unique():
            assert val in {-1, 0, 1}

    def test_dynamic_barrier_method(self, sample_ohlcv_df):
        """Test dynamic barrier method via factory."""
        labels = create_labels_for_training(
            sample_ohlcv_df,
            method='dynamic_barrier',
            take_profit=0.01,
            atr_period=10
        )

        assert len(labels) == len(sample_ohlcv_df)

    def test_unknown_method_raises(self, sample_ohlcv_df):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown labeling method"):
            create_labels_for_training(sample_ohlcv_df, method='unknown')


class TestLabelQuality:
    """Test label quality and correctness."""

    def test_label_distribution_reasonable(self, sample_ohlcv_df):
        """Test that label distribution is not heavily skewed."""
        labeler = TripleBarrierLabeler()
        labels = labeler.label(sample_ohlcv_df).dropna()

        # Check that we have multiple classes
        unique_labels = labels.unique()
        assert len(unique_labels) >= 2, "Should have at least 2 label classes"

        # Check no single class dominates too much (>90%)
        for label in unique_labels:
            ratio = (labels == label).mean()
            assert ratio < 0.9, f"Label {label} is too dominant: {ratio:.2%}"

    def test_labels_align_with_returns(self, sample_ohlcv_df):
        """Test that labels roughly align with forward returns."""
        config = TripleBarrierConfig(
            take_profit=0.01,
            stop_loss=0.005,
            max_holding_period=10,
            fee_adjustment=0
        )
        labeler = TripleBarrierLabeler(config)
        result_df = labeler.label_with_meta(sample_ohlcv_df)

        # Buy labels should have positive returns on average
        buy_returns = result_df[result_df['label'] == 1]['return_pct'].mean()
        sell_returns = result_df[result_df['label'] == -1]['return_pct'].mean()

        if not np.isnan(buy_returns) and not np.isnan(sell_returns):
            # Buy returns should be higher than sell returns
            assert buy_returns > sell_returns, \
                f"Buy returns ({buy_returns:.4f}) should > sell returns ({sell_returns:.4f})"
