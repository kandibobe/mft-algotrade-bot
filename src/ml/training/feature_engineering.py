"""
Feature Engineering Pipeline
=============================

Transform raw OHLCV data into ML-ready features.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Technical indicators
    include_price_features: bool = True
    include_volume_features: bool = True
    include_momentum_features: bool = True
    include_volatility_features: bool = True
    include_trend_features: bool = True

    # Lookback periods
    short_period: int = 14
    medium_period: int = 50
    long_period: int = 200

    # Feature scaling
    scale_features: bool = True
    scaling_method: str = "standard"  # standard, minmax, robust

    # Feature selection
    remove_correlated: bool = True
    correlation_threshold: float = 0.95

    # Time features
    include_time_features: bool = True

    # Custom features
    custom_features: List[str] = field(default_factory=list)


class FeatureEngineer:
    """
    Feature engineering pipeline for trading ML models.

    Transforms raw OHLCV data into ML-ready features with:
    - Technical indicators
    - Statistical features
    - Time-based features
    - Feature scaling and selection

    Usage:
        engineer = FeatureEngineer(config)
        features_df = engineer.transform(ohlcv_df)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.

        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
        self.scaler = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform OHLCV data to features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Engineering features from {len(df)} rows")

        result = df.copy()

        # Add technical indicators
        if self.config.include_price_features:
            result = self._add_price_features(result)

        if self.config.include_volume_features:
            result = self._add_volume_features(result)

        if self.config.include_momentum_features:
            result = self._add_momentum_features(result)

        if self.config.include_volatility_features:
            result = self._add_volatility_features(result)

        if self.config.include_trend_features:
            result = self._add_trend_features(result)

        # Add time features
        if self.config.include_time_features:
            result = self._add_time_features(result)

        # Remove highly correlated features
        if self.config.remove_correlated:
            result = self._remove_correlated_features(result)

        # Scale features
        if self.config.scale_features:
            result = self._scale_features(result)

        # Store feature names
        self.feature_names = [col for col in result.columns
                            if col not in ['open', 'high', 'low', 'close', 'volume']]

        logger.info(f"Generated {len(self.feature_names)} features")

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_log'] = np.log(df['close'] / df['close'].shift(1))

        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Gap features
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Price change from open
        df['intraday_return'] = (df['close'] - df['open']) / df['open']

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()

        # Volume moving averages
        df['volume_sma'] = df['volume'].rolling(self.config.short_period).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

        # Volume-price features
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.config.short_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.short_period).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_short = df['close'].ewm(span=12, adjust=False).mean()
        ema_long = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_short - ema_long
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Stochastic
        low_min = df['low'].rolling(self.config.short_period).min()
        high_max = df['high'].rolling(self.config.short_period).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(self.config.short_period).mean()
        df['atr_percent'] = df['atr'] / df['close']

        # Bollinger Bands
        sma = df['close'].rolling(self.config.medium_period).mean()
        std = df['close'].rolling(self.config.medium_period).std()
        df['bb_upper'] = sma + (2 * std)
        df['bb_lower'] = sma - (2 * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Historical volatility
        df['volatility'] = df['returns'].rolling(self.config.medium_period).std()

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Moving averages
        df['sma_short'] = df['close'].rolling(self.config.short_period).mean()
        df['sma_medium'] = df['close'].rolling(self.config.medium_period).mean()
        df['sma_long'] = df['close'].rolling(self.config.long_period).mean()

        # EMA
        df['ema_short'] = df['close'].ewm(span=self.config.short_period, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=self.config.medium_period, adjust=False).mean()

        # Price vs MA
        df['price_vs_sma_short'] = (df['close'] - df['sma_short']) / df['sma_short']
        df['price_vs_sma_medium'] = (df['close'] - df['sma_medium']) / df['sma_medium']

        # MA crossovers
        df['ma_cross_short_medium'] = (df['sma_short'] > df['sma_medium']).astype(int)
        df['ma_cross_medium_long'] = (df['sma_medium'] > df['sma_long']).astype(int)

        # ADX (trend strength)
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = true_range.rolling(self.config.short_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.config.short_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.config.short_period).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(self.config.short_period).mean()

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Hour, day of week, month
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        # Cyclical encoding (sin/cos for periodicity)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()

        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Drop features with high correlation
        to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > self.config.correlation_threshold)
        ]

        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            df = df.drop(columns=to_drop)

        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features to standard range."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        # Don't scale OHLCV columns
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in base_cols]

        if not feature_cols:
            return df

        # Select scaler
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:  # robust
            self.scaler = RobustScaler()

        # Fit and transform
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names
