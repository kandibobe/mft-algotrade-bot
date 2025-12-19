"""
Feature Engineering Pipeline
=============================

Transform raw OHLCV data into ML-ready features.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib

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

    IMPORTANT: To avoid data leakage, use fit_transform() on training data,
    then transform() on test/validation data. Never fit on test data!

    Usage:
        engineer = FeatureEngineer(config)

        # Training: fit scaler on train data only
        train_features = engineer.fit_transform(train_df)

        # Testing: apply same scaler (no fitting!)
        test_features = engineer.transform(test_df)

        # Save scaler for production
        engineer.save_scaler("models/scaler.joblib")
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
        self._is_fitted = False
        self._scaled_feature_cols: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler on data and transform. Use this for TRAINING data only.

        This method:
        1. Engineers features (indicators, time features, etc.)
        2. Removes correlated features
        3. FITS the scaler on this data (learns mean/std or min/max)
        4. Transforms the data using the fitted scaler

        IMPORTANT: Only call this on training data to avoid data leakage!

        Args:
            df: Training DataFrame with OHLCV data

        Returns:
            DataFrame with engineered and scaled features
        """
        logger.info(f"Fit-transforming features from {len(df)} rows (TRAINING MODE)")

        result = self._engineer_features(df)

        # Validate features before further processing
        is_valid, issues = self.validate_features(
            result,
            fix_issues=True,  # Auto-fix issues in training
            raise_on_error=False  # Don't fail, just warn
        )

        # Remove highly correlated features (learn which to remove from train)
        if self.config.remove_correlated:
            result = self._remove_correlated_features(result)

        # Fit scaler on training data and transform
        if self.config.scale_features:
            result = self._fit_scale_features(result)

        # Store feature names (after correlation removal)
        self.feature_names = [col for col in result.columns
                            if col not in ['open', 'high', 'low', 'close', 'volume']]

        self._is_fitted = True
        logger.info(f"Generated {len(self.feature_names)} features (scaler fitted)")

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using pre-fitted scaler. Use this for TEST/VALIDATION data.

        This method:
        1. Engineers features (same indicators as training)
        2. Applies the same scaler fitted on training data (NO refitting!)

        IMPORTANT: Must call fit_transform() first on training data!

        Args:
            df: Test/validation DataFrame with OHLCV data

        Returns:
            DataFrame with engineered and scaled features

        Raises:
            ValueError: If fit_transform() was not called first
        """
        if self.config.scale_features and not self._is_fitted:
            raise ValueError(
                "Scaler not fitted! Call fit_transform() on training data first. "
                "This prevents data leakage from test set into scaler parameters."
            )

        logger.info(f"Transforming features from {len(df)} rows (TEST MODE)")

        result = self._engineer_features(df)

        # Validate features (strict in test mode)
        is_valid, issues = self.validate_features(
            result,
            fix_issues=False,  # Don't auto-fix in test mode
            raise_on_error=True  # Fail if critical issues found
        )

        # Apply same correlation filter (use stored feature names)
        if self.config.remove_correlated and self.feature_names:
            # Keep only features that were kept during training
            cols_to_keep = ['open', 'high', 'low', 'close', 'volume'] + [
                col for col in self.feature_names if col in result.columns
            ]
            result = result[[col for col in cols_to_keep if col in result.columns]]

        # Apply pre-fitted scaler (NO fitting on test data!)
        if self.config.scale_features:
            result = self._apply_scale_features(result)

        logger.info(f"Transformed {len(self.feature_names)} features")

        return result

    def validate_features(
        self,
        df: pd.DataFrame,
        fix_issues: bool = False,
        raise_on_error: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate feature data quality.

        Checks for:
        - NaN values (missing data)
        - Inf values (division by zero, etc.)
        - Low variance features (constant or nearly constant)
        - Extreme outliers

        Args:
            df: DataFrame with features to validate
            fix_issues: If True, attempt to fix issues (fill NaN, clip outliers)
            raise_on_error: If True, raise ValueError on validation failure

        Returns:
            Tuple of (is_valid, issues_dict)

        Raises:
            ValueError: If raise_on_error=True and validation fails
        """
        issues = {
            'nan_columns': [],
            'inf_columns': [],
            'low_variance_columns': [],
            'outlier_columns': [],
            'warnings': []
        }

        # Check for NaN values
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            nan_counts = df[nan_cols].isnull().sum().to_dict()
            issues['nan_columns'] = nan_counts
            issues['warnings'].append(
                f"Found NaN values in {len(nan_cols)} columns: {nan_counts}"
            )

            if fix_issues:
                logger.warning(f"Filling NaN values in {nan_cols}")
                df[nan_cols] = df[nan_cols].fillna(method='ffill').fillna(0)

        # Check for Inf values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df[numeric_cols])
        inf_cols = numeric_cols[inf_mask.any()].tolist()

        if inf_cols:
            inf_counts = inf_mask[inf_cols].sum().to_dict()
            issues['inf_columns'] = inf_counts
            issues['warnings'].append(
                f"Found Inf values in {len(inf_cols)} columns: {inf_counts}"
            )

            if fix_issues:
                logger.warning(f"Replacing Inf values in {inf_cols}")
                df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
                df[inf_cols] = df[inf_cols].fillna(method='ffill').fillna(0)

        # Check for low variance features (potentially useless)
        if len(df) > 1:
            variance = df[numeric_cols].var()
            low_var_threshold = 1e-6
            low_var_cols = variance[variance < low_var_threshold].index.tolist()

            if low_var_cols:
                issues['low_variance_columns'] = low_var_cols
                issues['warnings'].append(
                    f"Found {len(low_var_cols)} low-variance features "
                    f"(var < {low_var_threshold}): {low_var_cols[:5]}..."
                )

        # Check for extreme outliers (beyond 5 std devs)
        for col in numeric_cols:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue  # Skip OHLCV columns

            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                outliers = df[(df[col] - mean).abs() > 5 * std]
                if len(outliers) > 0:
                    outlier_pct = len(outliers) / len(df) * 100
                    if outlier_pct > 1.0:  # More than 1% outliers
                        issues['outlier_columns'].append({
                            'column': col,
                            'count': len(outliers),
                            'percentage': outlier_pct
                        })

                        if fix_issues:
                            # Clip to ±5 std devs
                            lower_bound = mean - 5 * std
                            upper_bound = mean + 5 * std
                            df[col] = df[col].clip(lower_bound, upper_bound)

        # Check if any critical issues found
        has_critical_issues = bool(
            issues['nan_columns'] or issues['inf_columns']
        )

        # Log results
        if has_critical_issues:
            logger.error(
                f"Feature validation FAILED: {sum(len(v) if isinstance(v, (list, dict)) else 0 for v in issues.values())} issues found"
            )
            for warning in issues['warnings']:
                logger.error(f"  - {warning}")

            if raise_on_error:
                raise ValueError(
                    f"Feature validation failed. Issues: {issues}. "
                    f"Set fix_issues=True to attempt automatic fixes."
                )
        elif issues['warnings']:
            logger.warning("Feature validation passed with warnings:")
            for warning in issues['warnings']:
                logger.warning(f"  - {warning}")
        else:
            logger.info("✅ Feature validation passed - data quality OK")

        return (not has_critical_issues, issues)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features without scaling."""
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

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['returns'] = df['close'].pct_change(fill_method=None)
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
        df['volume_change'] = df['volume'].pct_change(fill_method=None)

        # Volume moving averages
        df['volume_sma'] = df['volume'].rolling(self.config.short_period).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

        # Volume-price features
        # ✅ FIXED: Use rolling window instead of cumsum to prevent data leakage
        # VWAP should only use past data, not future data
        vwap_window = self.config.short_period
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (
            (typical_price * df['volume']).rolling(vwap_window).sum() /
            df['volume'].rolling(vwap_window).sum()
        )
        df['vwap_diff'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)

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

        # Calculate true range for ADX
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        atr = true_range.rolling(self.config.short_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.config.short_period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(self.config.short_period).mean() / (atr + 1e-10))
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
        # Preserve OHLCV columns
        base_cols = ['open', 'high', 'low', 'close', 'volume']

        # Get numeric columns only, excluding base OHLCV columns
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in base_cols]

        if len(numeric_cols) < 2:
            return df

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()

        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Drop features with high correlation (never drop base OHLCV columns)
        to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > self.config.correlation_threshold)
            and column not in base_cols
        ]

        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            df = df.drop(columns=to_drop)

        return df

    def _fit_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler on features and transform. TRAINING DATA ONLY.

        This learns the scaling parameters (mean/std for StandardScaler,
        min/max for MinMaxScaler) from the training data.
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        # Don't scale OHLCV columns
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in base_cols]

        if not feature_cols:
            return df

        # Store feature columns for later use
        self._scaled_feature_cols = feature_cols

        # Select scaler
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:  # robust
            self.scaler = RobustScaler()

        # FIT on training data and transform
        logger.info(f"Fitting scaler ({self.config.scaling_method}) on {len(feature_cols)} features")
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        return df

    def _apply_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pre-fitted scaler to features. TEST/VALIDATION DATA ONLY.

        Uses scaling parameters learned from training data - NO refitting!
        This prevents data leakage from test set.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted! Call fit_transform() first.")

        # Use the same feature columns as training
        feature_cols = [col for col in self._scaled_feature_cols if col in df.columns]

        if not feature_cols:
            return df

        # TRANSFORM only (no fitting!)
        logger.info(f"Applying pre-fitted scaler to {len(feature_cols)} features")
        df[feature_cols] = self.scaler.transform(df[feature_cols])

        return df

    def save_scaler(self, path: str) -> None:
        """
        Save fitted scaler to file for production use.

        Args:
            path: Path to save scaler (e.g., "models/scaler.joblib")
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted! Call fit_transform() first.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scaler_data = {
            "scaler": self.scaler,
            "feature_cols": self._scaled_feature_cols,
            "feature_names": self.feature_names,
            "config": {
                "scaling_method": self.config.scaling_method,
                "correlation_threshold": self.config.correlation_threshold,
            }
        }

        joblib.dump(scaler_data, output_path)
        logger.info(f"Scaler saved to {output_path}")

    def load_scaler(self, path: str) -> None:
        """
        Load pre-fitted scaler from file.

        Args:
            path: Path to load scaler from
        """
        input_path = Path(path)
        scaler_data = joblib.load(input_path)

        self.scaler = scaler_data["scaler"]
        self._scaled_feature_cols = scaler_data["feature_cols"]
        self.feature_names = scaler_data["feature_names"]
        self._is_fitted = True

        logger.info(f"Scaler loaded from {input_path} ({len(self.feature_names)} features)")

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names

    def is_fitted(self) -> bool:
        """Check if scaler has been fitted."""
        return self._is_fitted
