"""
Tests for Feature Engineering Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml.training.feature_engineering import (
    FeatureEngineer,
    FeatureConfig,
)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=300, freq="1H")

    # Generate realistic price data
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.randn(300) * 100)

    df = pd.DataFrame({
        "open": close_prices + np.random.randn(300) * 50,
        "high": close_prices + np.abs(np.random.randn(300) * 100),
        "low": close_prices - np.abs(np.random.randn(300) * 100),
        "close": close_prices,
        "volume": np.random.uniform(1000, 10000, 300),
    }, index=dates)

    return df


class TestFeatureConfig:
    """Test FeatureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = FeatureConfig()

        assert config.include_price_features is True
        assert config.include_volume_features is True
        assert config.include_momentum_features is True
        assert config.include_volatility_features is True
        assert config.include_trend_features is True
        assert config.scale_features is True
        assert config.scaling_method == "standard"
        assert config.short_period == 14
        assert config.medium_period == 50
        assert config.long_period == 200

    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureConfig(
            include_price_features=False,
            short_period=10,
            scaling_method="minmax",
        )

        assert config.include_price_features is False
        assert config.short_period == 10
        assert config.scaling_method == "minmax"


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_initialization(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer()

        assert engineer.config is not None
        assert isinstance(engineer.config, FeatureConfig)
        assert engineer.feature_names == []
        assert engineer.scaler is None

    def test_transform_basic(self, sample_ohlcv_data):
        """Test basic feature transformation."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_ohlcv_data)

        # Should have more columns than input
        assert len(result.columns) > len(sample_ohlcv_data.columns)

        # Should preserve original data
        assert "open" in result.columns
        assert "close" in result.columns

        # Should have generated features
        assert len(engineer.get_feature_names()) > 0

    def test_price_features(self, sample_ohlcv_data):
        """Test price feature generation."""
        config = FeatureConfig(
            include_price_features=True,
            include_volume_features=False,
            include_momentum_features=False,
            include_volatility_features=False,
            include_trend_features=False,
            include_time_features=False,
            scale_features=False,
        )

        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_ohlcv_data)

        # Check for price features
        assert "returns" in result.columns
        assert "returns_log" in result.columns
        assert "price_position" in result.columns
        assert "gap" in result.columns
        assert "intraday_return" in result.columns

    def test_volume_features(self, sample_ohlcv_data):
        """Test volume feature generation."""
        config = FeatureConfig(
            include_price_features=False,
            include_volume_features=True,
            include_momentum_features=False,
            include_volatility_features=False,
            include_trend_features=False,
            include_time_features=False,
            scale_features=False,
        )

        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_ohlcv_data)

        # Check for volume features
        assert "volume_change" in result.columns
        assert "volume_sma" in result.columns
        assert "volume_ratio" in result.columns
        assert "vwap" in result.columns

    def test_momentum_features(self, sample_ohlcv_data):
        """Test momentum indicator generation."""
        config = FeatureConfig(
            include_price_features=False,
            include_volume_features=False,
            include_momentum_features=True,
            include_volatility_features=False,
            include_trend_features=False,
            include_time_features=False,
            scale_features=False,
        )

        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_ohlcv_data)

        # Check for momentum indicators
        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_volatility_features(self, sample_ohlcv_data):
        """Test volatility indicator generation."""
        config = FeatureConfig(
            include_price_features=False,
            include_volume_features=False,
            include_momentum_features=False,
            include_volatility_features=True,
            include_trend_features=False,
            include_time_features=False,
            scale_features=False,
        )

        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_ohlcv_data)

        # Check for volatility indicators
        assert "atr" in result.columns
        assert "atr_percent" in result.columns
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns
        assert "bb_position" in result.columns
        assert "volatility" in result.columns

    def test_trend_features(self, sample_ohlcv_data):
        """Test trend indicator generation."""
        config = FeatureConfig(
            include_price_features=False,
            include_volume_features=False,
            include_momentum_features=False,
            include_volatility_features=False,
            include_trend_features=True,
            include_time_features=False,
            scale_features=False,
        )

        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_ohlcv_data)

        # Check for trend indicators
        assert "sma_short" in result.columns
        assert "sma_medium" in result.columns
        assert "sma_long" in result.columns
        assert "ema_short" in result.columns
        assert "ema_medium" in result.columns
        assert "price_vs_sma_short" in result.columns
        assert "ma_cross_short_medium" in result.columns
        assert "adx" in result.columns

    def test_time_features(self, sample_ohlcv_data):
        """Test time-based feature generation."""
        config = FeatureConfig(
            include_price_features=False,
            include_volume_features=False,
            include_momentum_features=False,
            include_volatility_features=False,
            include_trend_features=False,
            include_time_features=True,
            scale_features=False,
        )

        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_ohlcv_data)

        # Check for time features
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "day_sin" in result.columns
        assert "day_cos" in result.columns

    def test_feature_scaling(self, sample_ohlcv_data):
        """Test feature scaling."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_ohlcv_data)

        # Check that scaler was created
        assert engineer.scaler is not None

        # Check that features are scaled (not OHLCV columns)
        feature_cols = engineer.get_feature_names()
        if feature_cols:
            # Features should have reasonable range after scaling
            for col in feature_cols[:5]:  # Check first 5 features
                if col in result.columns:
                    values = result[col].dropna()
                    if len(values) > 0:
                        # Standard scaled features should be roughly centered around 0
                        assert -10 < values.mean() < 10

    def test_correlation_removal(self, sample_ohlcv_data):
        """Test highly correlated feature removal."""
        config = FeatureConfig(
            remove_correlated=True,
            correlation_threshold=0.95,
        )

        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_ohlcv_data)

        # Should have features
        assert len(engineer.get_feature_names()) > 0

        # Check that remaining features are not highly correlated
        feature_cols = [col for col in result.columns
                       if col not in ['open', 'high', 'low', 'close', 'volume']]

        if len(feature_cols) > 1:
            corr_matrix = result[feature_cols].corr().abs()
            # Remove diagonal
            np.fill_diagonal(corr_matrix.values, 0)
            # Check max correlation
            max_corr = corr_matrix.max().max()
            assert max_corr <= config.correlation_threshold

    def test_get_feature_names(self, sample_ohlcv_data):
        """Test feature names retrieval."""
        engineer = FeatureEngineer()
        engineer.transform(sample_ohlcv_data)

        feature_names = engineer.get_feature_names()

        # Should return list
        assert isinstance(feature_names, list)

        # Should not include OHLCV columns
        assert "open" not in feature_names
        assert "close" not in feature_names

        # Should have features
        assert len(feature_names) > 0

    def test_nan_handling(self, sample_ohlcv_data):
        """Test NaN handling in features."""
        engineer = FeatureEngineer()
        result = engineer.transform(sample_ohlcv_data)

        # Some features will have NaN at the beginning (due to rolling windows)
        # This is expected behavior

        # But should not be all NaN
        for col in result.columns:
            assert not result[col].isna().all()

    def test_different_scaling_methods(self, sample_ohlcv_data):
        """Test different scaling methods."""
        for method in ["standard", "minmax", "robust"]:
            config = FeatureConfig(scaling_method=method)
            engineer = FeatureEngineer(config)
            result = engineer.transform(sample_ohlcv_data)

            # Should complete without error
            assert len(result) == len(sample_ohlcv_data)
            assert engineer.scaler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
