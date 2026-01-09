"""
Feature Engineering Pipeline
=============================

Transform raw OHLCV data into ML-ready features.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import config

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
    include_meta_labeling_features: bool = True  # Meta-labeling features for De Prado methodology

    # Stationarity transformation
    enforce_stationarity: bool = False
    fractional_differentiation_d: float = 0.5  # Fractional differentiation parameter
    use_log_returns: bool = True  # Alternative to fractional differentiation

    # Lookback periods
    short_period: int = 14
    medium_period: int = 50
    long_period: int = 200

    # Feature scaling
    scale_features: bool = True
    scaling_method: str = "standard"  # standard, minmax, robust

    # Feature selection
    remove_correlated: bool = True
    correlation_threshold: float = 0.85  # Reduced from 0.95 to 0.85 per roadmap

    # Time features
    include_time_features: bool = True

    # Custom features
    custom_features: list[str] = field(default_factory=list)


class FractionalDifferentiator:
    """
    Fractional differentiation for creating stationary time series while preserving memory.

    Based on the binomial expansion method for fractional differentiation.
    This transforms non-stationary price series into stationary series suitable for ML models.

    Reference: Advances in Financial Machine Learning, Marcos Lopez de Prado
    """

    def __init__(self, d: float = 0.5, window_size: int = 100):
        """
        Initialize fractional differentiator.

        Args:
            d: Fractional differentiation parameter (0 < d < 1)
                d=0: original series
                d=1: first difference
                d=0.5: fractional difference (preserves some memory)
            window_size: Window size for practical implementation
        """
        self.d = d
        self.window_size = window_size
        self._weights = self._calculate_weights()

    def _calculate_weights(self) -> np.ndarray:
        """Calculate binomial weights for fractional differentiation."""
        weights = [1.0]
        for k in range(1, self.window_size):
            weight = -weights[-1] * (self.d - k + 1) / k
            weights.append(weight)
        return np.array(weights)

    def differentiate(self, series: pd.Series) -> pd.Series:
        """
        Apply fractional differentiation to a time series.

        Args:
            series: Input time series (must be sorted chronologically)

        Returns:
            Fractionally differentiated series
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            logger.warning("Series index is not DatetimeIndex, ensure chronological ordering")

        # Convert to numpy for efficiency
        values = series.values
        n = len(values)

        # Initialize result array
        result = np.full(n, np.nan, dtype=float)

        # Apply fractional differentiation
        for i in range(self.window_size, n):
            window = values[i - self.window_size + 1 : i + 1]
            result[i] = np.dot(window[::-1], self._weights)

        # For first window_size elements, use simple difference
        if self.window_size > 0:
            result[: self.window_size] = np.diff(values[: self.window_size + 1], prepend=values[0])

        return pd.Series(result, index=series.index)

    def inverse_transform(self, diff_series: pd.Series, initial_value: float) -> pd.Series:
        """
        Inverse transform of fractional differentiation (approximate).

        Note: Exact inversion is not possible, but this provides an approximation.

        Args:
            diff_series: Fractionally differentiated series
            initial_value: Initial value of original series

        Returns:
            Approximate original series
        """
        # This is a simplified approximation
        # In practice, fractional differentiation is not easily invertible
        logger.warning("Fractional differentiation inverse transform is approximate")

        result = np.zeros(len(diff_series))
        result[0] = initial_value

        for i in range(1, len(diff_series)):
            # Simple cumulative sum approximation
            result[i] = result[i - 1] + diff_series.iloc[i]

        return pd.Series(result, index=diff_series.index)


class FeatureEngineer:
    """
    Feature engineering pipeline for trading ML models.

    Transforms raw OHLCV data into ML-ready features with:
    - Technical indicators
    - Statistical features
    - Time-based features
    - Feature scaling and selection
    - Stationarity transformation (fractional differentiation)

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

    def __init__(self, config_obj: FeatureConfig | None = None):
        """
        Initialize feature engineer.

        Args:
            config_obj: Feature engineering configuration
        """
        self.config = config_obj or FeatureConfig()
        self.feature_names: list[str] = []
        self.scaler = None
        self._is_fitted = False
        self._scaled_feature_cols: list[str] = []
        self._fractional_differentiator = None
        self._stationarity_applied = False
        self._feature_cache = {}

    def prepare_data(self, df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        Prepare data: Stationarity -> Engineer -> Clean.
        Does NOT remove correlated features or scale.
        Safe to call on full dataset if you split afterwards.
        """
        if use_cache:
            cache_key = hash(tuple(df.index))
            if cache_key in self._feature_cache:
                logger.info("Using cached features")
                return self._feature_cache[cache_key]

        logger.info(f"Preparing features from {len(df)} rows")

        # Apply stationarity transformation if configured
        result = self._apply_stationarity_transformation(df.copy())

        # Engineer features
        result = self._engineer_features(result)

        # AGGRESSIVE CLEANING: Replace inf with NaN and drop all NaN rows
        result = self._apply_aggressive_cleaning(result)

        if use_cache:
            self._feature_cache[cache_key] = result

        return result

    def fit_scaler_and_selector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and feature selector on TRAINING data.

        1. Identifies and removes correlated features (from Train)
        2. Fits scaler (on Train)
        3. Returns transformed Train data
        """
        result = df.copy()

        # Remove highly correlated features (learn which to remove from train)
        if self.config.remove_correlated:
            result = self._remove_correlated_features(result)

        # Fit scaler on training data and transform
        if self.config.scale_features:
            result = self._fit_scale_features(result)

        # Store feature names (after correlation removal)
        self.feature_names = [
            col for col in result.columns if col not in ["open", "high", "low", "close", "volume"]
        ]

        self._is_fitted = True
        logger.info(f"Fitted scaler and selector. Kept {len(self.feature_names)} features.")

        return result

    def transform_scaler_and_selector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaler and selector to TEST data.

        1. Selects same features as Train
        2. Applies pre-fitted scaler
        """
        if not self._is_fitted:
            raise ValueError("Selector/Scaler not fitted! Call fit_scaler_and_selector() first.")

        result = df.copy()

        # Filter columns to match training set
        cols_to_keep = ["open", "high", "low", "close", "volume"] + [
            col for col in self.feature_names if col in result.columns
        ]
        result = result[[c for c in cols_to_keep if c in result.columns]]

        # Apply pre-fitted scaler (NO fitting on test data!)
        if self.config.scale_features:
            result = self._apply_scale_features(result)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler on data and transform. Use this for TRAINING data only.

        This method:
        1. Applies stationarity transformation (fractional differentiation or log returns)
        2. Engineers features (indicators, time features, etc.)
        3. Applies aggressive cleaning (replace inf with NaN, drop all NaN rows)
        4. Removes correlated features
        5. FITS the scaler on this data (learns mean/std or min/max)
        6. Transforms the data using the fitted scaler

        IMPORTANT: Only call this on training data to avoid data leakage!

        Args:
            df: Training DataFrame with OHLCV data

        Returns:
            DataFrame with engineered and scaled features
        """
        logger.info(f"Fit-transforming features from {len(df)} rows (TRAINING MODE)")

        # Apply stationarity transformation if configured
        result = self._apply_stationarity_transformation(df.copy())

        result = self._engineer_features(result)

        # AGGRESSIVE CLEANING: Replace inf with NaN and drop all NaN rows
        result = self._apply_aggressive_cleaning(result)

        # Validate features before further processing
        is_valid, issues = self.validate_features(
            result,
            fix_issues=True,  # Auto-fix issues in training
            raise_on_error=False,  # Don't fail, just warn
            drop_low_variance=False,  # Don't drop low variance, let scaler handle it
        )

        # Remove highly correlated features (learn which to remove from train)
        if self.config.remove_correlated:
            result = self._remove_correlated_features(result)

        # Fit scaler on training data and transform
        if self.config.scale_features:
            result = self._fit_scale_features(result)

        # Store feature names (after correlation removal)
        self.feature_names = [
            col for col in result.columns if col not in ["open", "high", "low", "close", "volume"]
        ]

        self._is_fitted = True
        logger.info(f"Generated {len(self.feature_names)} features (scaler fitted)")

        return result

    def transform(self, df: pd.DataFrame, drop_low_variance: bool = False) -> pd.DataFrame:
        """
        Transform data using pre-fitted scaler. Use this for TEST/VALIDATION data.

        This method:
        1. Applies the same stationarity transformation as training
        2. Engineers features (same indicators as training)
        3. Applies aggressive cleaning (replace inf with NaN, drop all NaN rows)
        4. Applies the same scaler fitted on training data (NO refitting!)

        IMPORTANT: Must call fit_transform() first on training data!

        Args:
            df: Test/validation DataFrame with OHLCV data
            drop_low_variance: Whether to drop low variance features (default False for test)

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

        # Apply the same stationarity transformation as training
        result = self._apply_stationarity_transformation(df.copy(), is_training=False)

        result = self._engineer_features(result)

        # AGGRESSIVE CLEANING: Replace inf with NaN and drop all NaN rows
        result = self._apply_aggressive_cleaning(result)

        # Validate features (allow auto-fixing in test mode too)
        is_valid, issues = self.validate_features(
            result,
            fix_issues=True,  # Auto-fix issues in test mode as well
            raise_on_error=False,  # Don't fail, just warn
            drop_low_variance=drop_low_variance,
        )

        # Apply same correlation filter (use stored feature names)
        if self.config.remove_correlated and self.feature_names:
            # Keep only features that were kept during training
            cols_to_keep = ["open", "high", "low", "close", "volume"] + [
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
        raise_on_error: bool = True,
        drop_low_variance: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
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
            "nan_columns": [],
            "inf_columns": [],
            "low_variance_columns": [],
            "outlier_columns": [],
            "warnings": [],
        }

        # Check for NaN values
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            nan_counts = df[nan_cols].isnull().sum().to_dict()
            issues["nan_columns"] = nan_counts
            issues["warnings"].append(f"Found NaN values in {len(nan_cols)} columns: {nan_counts}")

            if fix_issues:
                logger.warning(f"Filling NaN values in {nan_cols}")
                # Smart filling: forward fill first, then backward fill, then fill with 0
                for col in nan_cols:
                    # For price-related columns, use forward fill then backward fill
                    if col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].ffill().bfill().fillna(0)
                    # For indicator columns, fill with 0 or appropriate default
                    elif "rsi" in col.lower():
                        df[col] = df[col].fillna(50)  # RSI default to neutral 50
                    elif "stoch" in col.lower():
                        df[col] = df[col].fillna(50)  # Stochastic default to 50
                    elif "bb_" in col.lower():
                        # For Bollinger Bands, fill with price or moving average
                        if col == "bb_position":
                            df[col] = df[col].fillna(0.5)  # Middle of band
                        else:
                            df[col] = df[col].ffill().bfill().fillna(df["close"])
                    elif "atr" in col.lower():
                        df[col] = df[col].ffill().bfill().fillna(df["close"] * 0.01)
                    elif "macd" in col.lower():
                        df[col] = df[col].fillna(0)  # MACD default to 0
                    elif "returns" in col.lower() or "change" in col.lower():
                        df[col] = df[col].fillna(0)  # Returns default to 0
                    else:
                        # Generic fill: forward fill, backward fill, then 0
                        df[col] = df[col].ffill().bfill().fillna(0)

        # Check for Inf values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Handle potential duplicate columns during selection safely
        try:
            inf_mask = np.isinf(df[numeric_cols])
        except Exception:
            # Fallback for complex duplicate column cases
            inf_mask = df[numeric_cols].apply(np.isinf)

        # Use columns from mask to ensure alignment if duplicates expanded
        has_inf = inf_mask.any()
        inf_cols = inf_mask.columns[has_inf].tolist()

        if inf_cols:
            # Use safe indexing for counts
            inf_counts = inf_mask.loc[:, has_inf].sum().to_dict()
            issues["inf_columns"] = inf_counts
            issues["warnings"].append(f"Found Inf values in {len(inf_cols)} columns: {inf_counts}")

            if fix_issues:
                logger.warning(f"Replacing Inf values in {inf_cols}")
                # Use robust replacement that handles duplicates
                for col in set(inf_cols):
                    if col in df.columns:
                        col_data = df[col]
                        if isinstance(col_data, pd.DataFrame):
                            # Update all duplicates
                            df[col] = col_data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
                        else:
                            df[col] = col_data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

        # Check for low variance features (potentially useless)
        if len(df) > 1:
            variance = df[numeric_cols].var()
            # Reduced threshold to support 5m/1m return features which are naturally small
            low_var_threshold = 1e-12
            low_var_cols = variance[variance < low_var_threshold].index.tolist()

            if low_var_cols:
                issues["low_variance_columns"] = low_var_cols
                issues["warnings"].append(
                    f"Found {len(low_var_cols)} low-variance features "
                    f"(var < {low_var_threshold}): {low_var_cols[:5]}..."
                )
                # Log the variance of these columns to understand why they are low
                for col in low_var_cols[:5]:
                    logger.debug(f"Low variance col {col}: {variance[col]}")

                if fix_issues and drop_low_variance:
                    # Drop low variance columns
                    # Filter only columns that exist (in case of duplicates or changes)
                    cols_to_drop = [c for c in low_var_cols if c in df.columns]
                    if cols_to_drop:
                        df.drop(columns=cols_to_drop, inplace=True)
                        logger.info(f"Dropped {len(cols_to_drop)} low variance features")

        # Check for extreme outliers (beyond 5 std devs)
        for col in numeric_cols:
            if col not in df.columns:
                continue  # Skip dropped columns

            if col in ["open", "high", "low", "close", "volume"]:
                continue  # Skip OHLCV columns

            # Handle duplicate columns safely
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                # If duplicate columns exist, use the first one for stats
                # This prevents ambiguous truth value errors and matrix operations
                col_data = col_data.iloc[:, 0]

            mean = col_data.mean()
            std = col_data.std()

            if std > 0:
                # Use Series operations to find outliers (avoids df[mask] indexing issues)
                outlier_mask = (col_data - mean).abs() > 5 * std
                num_outliers = outlier_mask.sum()

                if num_outliers > 0:
                    outlier_pct = num_outliers / len(df) * 100
                    if outlier_pct > 1.0:  # More than 1% outliers
                        issues["outlier_columns"].append(
                            {
                                "column": col,
                                "count": int(num_outliers),
                                "percentage": float(outlier_pct),
                            }
                        )

                        if fix_issues:
                            # Clip to ±5 std devs
                            lower_bound = mean - 5 * std
                            upper_bound = mean + 5 * std
                            # Apply clip (works for both Series and DataFrame if duplicate cols)
                            df[col] = df[col].clip(lower_bound, upper_bound)

        # Check if any critical issues found
        has_critical_issues = bool(issues["nan_columns"] or issues["inf_columns"])

        # Log results
        if has_critical_issues:
            logger.error(
                f"Feature validation FAILED: {sum(len(v) if isinstance(v, (list, dict)) else 0 for v in issues.values())} issues found"
            )
            for warning in issues["warnings"]:
                logger.error(f"  - {warning}")

            if raise_on_error:
                raise ValueError(
                    f"Feature validation failed. Issues: {issues}. "
                    f"Set fix_issues=True to attempt automatic fixes."
                )
        elif issues["warnings"]:
            logger.warning("Feature validation passed with warnings:")
            for warning in issues["warnings"]:
                logger.warning(f"  - {warning}")
        else:
            logger.info("✅ Feature validation passed - data quality OK")

        return (not has_critical_issues, issues)

    def _apply_stationarity_transformation(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> pd.DataFrame:
        """
        Apply stationarity transformation to price data.

        Options:
        1. Fractional differentiation (preserves memory)
        2. Log returns (simpler, less memory)

        Args:
            df: DataFrame with OHLCV data
            is_training: Whether this is training data (affects fitting)

        Returns:
            DataFrame with stationarity transformations applied
        """
        if not self.config.enforce_stationarity:
            return df

        result = df.copy()

        # Apply fractional differentiation if configured
        if self.config.fractional_differentiation_d > 0 and not self.config.use_log_returns:
            if is_training or self._fractional_differentiator is not None:
                if is_training:
                    # Create and fit fractional differentiator on training data
                    self._fractional_differentiator = FractionalDifferentiator(
                        d=self.config.fractional_differentiation_d, window_size=100
                    )

                # Apply fractional differentiation to close prices
                result["close_fractional_diff"] = self._fractional_differentiator.differentiate(
                    result["close"]
                )

                # Replace original close with fractionally differentiated version for feature engineering
                # Keep original close for reference
                result["close_original"] = result["close"]
                result["close"] = result["close_fractional_diff"].fillna(result["close"])

                logger.info(
                    f"Applied fractional differentiation (d={self.config.fractional_differentiation_d})"
                )

        # Apply log returns if configured (simpler alternative)
        elif self.config.use_log_returns:
            try:
                # Safe log calculation
                # Ensure close prices are positive and non-zero to avoid log errors
                close_prices = result["close"].replace(0, np.nan).ffill()
                prev_close = close_prices.shift(1)

                # Clip ratio to avoid extreme values (though unlikely with price data)
                price_ratio = close_prices / prev_close
                price_ratio = price_ratio.clip(lower=1e-9)

                result["log_returns"] = np.log(price_ratio).fillna(0)
            except Exception as e:
                logger.error(f"Error calculating log_returns: {e}")
                result["log_returns"] = result["close"].pct_change(fill_method=None).fillna(0)

            # For feature engineering, we can use log returns directly
            # or create a cumulative sum for a stationary price-like series
            result["close_log_cumsum"] = result["log_returns"].cumsum()

            # Replace close with log cumulative sum for feature engineering
            # This ensures all technical indicators are computed on stationary series
            result["close_original"] = result["close"]
            result["close"] = result["close_log_cumsum"]

            logger.info(
                "Applied log returns for stationarity - using close_log_cumsum for feature engineering"
            )

        self._stationarity_applied = True
        return result

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

        # Add meta-labeling features (De Prado methodology)
        if self.config.include_meta_labeling_features:
            result = self._add_meta_labeling_features(result)

        # Add time features
        if self.config.include_time_features:
            result = self._add_time_features(result)

        # Prevent massive data loss due to rolling windows by backfilling initial NaNs
        # Rolling windows create NaNs at the start. If we don't fill them,
        # _apply_aggressive_cleaning will drop all these rows (potentially 40%+ of data).

        # FIX: Replace Inf with NaN first, so they can be filled
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # We forward fill first (to fill gaps) then backfill (to fill initial NaNs)
        result = result.ffill().bfill()

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features with lag features and rolling statistics."""
        # Returns
        df["returns"] = df["close"].pct_change(fill_method=None)

        try:
            # Safe log returns
            # Use log1p(pct_change) which is equivalent to log(close/prev_close) but safer for small changes
            # log(close/prev) = log((prev + diff)/prev) = log(1 + diff/prev) = log1p(pct_change)
            # Ensure returns are > -1 to avoid log errors
            safe_returns = df["returns"].clip(lower=-0.999999)
            df["returns_log"] = np.log1p(safe_returns)
        except Exception as e:
            logger.warning(f"Error calculating returns_log: {e}. Using simple returns.")
            df["returns_log"] = df["returns"]

        # Multiple timeframes returns
        for period in [2, 3, 5, 10, 20]:
            df[f"returns_{period}"] = df["close"].pct_change(period, fill_method=None)

        # Lag features (past prices)
        for lag in [1, 2, 3, 5, 10]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        # Price position in range
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        # Gap features
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Price change from open
        df["intraday_return"] = (df["close"] - df["open"]) / df["open"]

        # High/Low ratios
        df["high_low_ratio"] = df["high"] / (df["low"] + 1e-10)
        df["close_open_ratio"] = df["close"] / (df["open"] + 1e-10)

        # Price momentum
        for period in [5, 10, 20]:
            df[f"price_momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f"close_rolling_mean_{window}"] = df["close"].rolling(window).mean()
            df[f"close_rolling_std_{window}"] = df["close"].rolling(window).std()
            df[f"close_rolling_min_{window}"] = df["close"].rolling(window).min()
            df[f"close_rolling_max_{window}"] = df["close"].rolling(window).max()
            df[f"volume_rolling_mean_{window}"] = df["volume"].rolling(window).mean()

        # Price vs rolling statistics
        for window in [5, 10, 20]:
            df[f"close_vs_rolling_mean_{window}"] = (
                df["close"] - df[f"close_rolling_mean_{window}"]
            ) / df[f"close_rolling_mean_{window}"]
            df[f"close_vs_rolling_min_{window}"] = (
                df["close"] - df[f"close_rolling_min_{window}"]
            ) / df[f"close_rolling_min_{window}"]
            df[f"close_vs_rolling_max_{window}"] = (
                df["close"] - df[f"close_rolling_max_{window}"]
            ) / df[f"close_rolling_max_{window}"]

        # Volatility features
        df["returns_volatility_5"] = df["returns"].rolling(5).std()
        df["returns_volatility_10"] = df["returns"].rolling(10).std()
        df["returns_volatility_20"] = df["returns"].rolling(20).std()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume changes
        df["volume_change"] = df["volume"].pct_change(fill_method=None)

        # Volume moving averages
        df["volume_sma"] = df["volume"].rolling(self.config.short_period).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)

        # Volume-price features
        # ✅ FIXED: Use rolling window instead of cumsum to prevent data leakage
        # VWAP should only use past data, not future data
        vwap_window = self.config.short_period
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # Calculate VWAP
        vwap_raw = (typical_price * df["volume"]).rolling(vwap_window).sum() / df["volume"].rolling(
            vwap_window
        ).sum()

        # CRITICAL FIX: Shift VWAP by 1 to avoid lookahead bias.
        # If we are at Open[i], we cannot know Volume[i] or Close[i].
        # We must use VWAP calculated up to Close[i-1].
        df["vwap"] = vwap_raw.shift(1)

        # Recalculate diff using shifted VWAP
        df["vwap_diff"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-10)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.config.short_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config.short_period).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_short = df["close"].ewm(span=12, adjust=False).mean()
        ema_long = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_short - ema_long
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Stochastic
        low_min = df["low"].rolling(self.config.short_period).min()
        high_max = df["high"].rolling(self.config.short_period).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(self.config.short_period).mean()
        df["atr_percent"] = df["atr"] / df["close"]

        # Bollinger Bands
        sma = df["close"].rolling(self.config.medium_period).mean()
        std = df["close"].rolling(self.config.medium_period).std()
        df["bb_upper"] = sma + (2 * std)
        df["bb_lower"] = sma - (2 * std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-10
        )

        # Historical volatility
        df["volatility"] = df["returns"].rolling(self.config.medium_period).std()

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Moving averages
        df["sma_short"] = df["close"].rolling(self.config.short_period).mean()
        df["sma_medium"] = df["close"].rolling(self.config.medium_period).mean()
        df["sma_long"] = df["close"].rolling(self.config.long_period).mean()

        # EMA
        df["ema_short"] = df["close"].ewm(span=self.config.short_period, adjust=False).mean()
        df["ema_medium"] = df["close"].ewm(span=self.config.medium_period, adjust=False).mean()

        # Price vs MA
        df["price_vs_sma_short"] = (df["close"] - df["sma_short"]) / df["sma_short"]
        df["price_vs_sma_medium"] = (df["close"] - df["sma_medium"]) / df["sma_medium"]

        # MA crossovers
        df["ma_cross_short_medium"] = (df["sma_short"] > df["sma_medium"]).astype(int)
        df["ma_cross_medium_long"] = (df["sma_medium"] > df["sma_long"]).astype(int)

        # ADX (trend strength)
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate true range for ADX
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        atr = true_range.rolling(self.config.short_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.config.short_period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(self.config.short_period).mean() / (atr + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.rolling(self.config.short_period).mean()

        return df

    def _add_meta_labeling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features for Meta-Labeling (De Prado Methodology).

        Meta-Labeling uses a secondary model to predict whether
        primary signals will be profitable.

        Features added:
        1. Primary signal strength (combination of RSI, MACD, etc.)
        2. Volatility features (for risk assessment)
        3. Spread estimation (bid-ask spread proxy)
        4. Order book imbalance proxy
        5. Market regime features
        """
        # 1. Primary signal strength (simple combination of indicators)
        # RSI-based signal: RSI < 30 (oversold) or RSI > 70 (overbought)
        if "rsi" in df.columns:
            rsi_signal = ((df["rsi"] < 30) | (df["rsi"] > 70)).astype(int)
            rsi_strength = np.abs(df["rsi"] - 50) / 50  # Normalized 0-1
        else:
            rsi_signal = 0
            rsi_strength = 0

        # MACD signal: MACD histogram positive and increasing
        if "macd_hist" in df.columns:
            macd_signal = (
                (df["macd_hist"] > 0) & (df["macd_hist"] > df["macd_hist"].shift(1))
            ).astype(int)
            macd_strength = df["macd_hist"].abs() / (
                df["macd_hist"].abs().rolling(20).mean() + 1e-10
            )
            macd_strength = macd_strength.clip(upper=2.0) / 2.0  # Normalize to 0-1
        else:
            macd_signal = 0
            macd_strength = 0

        # EMA crossover signal
        if "ema_short" in df.columns and "ema_medium" in df.columns:
            ema_cross_signal = (df["ema_short"] > df["ema_medium"]).astype(int)
            ema_strength = (df["ema_short"] - df["ema_medium"]).abs() / df["ema_medium"] * 100
            ema_strength = ema_strength.clip(upper=5.0) / 5.0  # Normalize to 0-1
        else:
            ema_cross_signal = 0
            ema_strength = 0

        # Combined primary signal (binary)
        combined_signal = (rsi_signal + macd_signal + ema_cross_signal) >= 2
        # Handle case where combined_signal is a scalar boolean (if all inputs are 0)
        if isinstance(combined_signal, (bool, np.bool_)):
            df["primary_signal"] = int(combined_signal)
        else:
            df["primary_signal"] = combined_signal.astype(int)

        # Primary signal strength (0-1 scale)
        df["primary_signal_strength"] = (rsi_strength + macd_strength + ema_strength) / 3

        # 2. Volatility features (already have volatility, but add more)
        if "volatility" not in df.columns:
            # Ensure returns exist
            if "returns" not in df.columns:
                df["returns"] = df["close"].pct_change(fill_method=None)
            df["volatility"] = df["returns"].rolling(self.config.medium_period).std()

        # Volatility regime (high/medium/low)
        # Calculate quantiles separately to avoid list issue
        vol_q33 = df["volatility"].rolling(100).quantile(0.33)
        vol_q66 = df["volatility"].rolling(100).quantile(0.66)
        df["volatility_regime"] = 0  # Default medium
        df.loc[df["volatility"] > vol_q66, "volatility_regime"] = 1  # High
        df.loc[df["volatility"] < vol_q33, "volatility_regime"] = -1  # Low

        # 3. Spread estimation (bid-ask spread proxy)
        # Use high-low range as proxy for spread
        df["spread_pct"] = (df["high"] - df["low"]) / df["close"] * 100
        df["spread_ratio"] = df["spread_pct"] / df["spread_pct"].rolling(20).mean()

        # 4. Order book imbalance proxy
        # Use volume-price relationship as proxy
        if "volume" in df.columns and "close" in df.columns:
            # Volume-weighted price change
            price_change = df["close"].pct_change(fill_method=None)
            volume_change = df["volume"].pct_change(fill_method=None)
            df["order_imbalance"] = price_change * volume_change

            # Normalized order imbalance - optimized to avoid lambda
            # Calculate rolling mean and std first
            rolling_mean = df["order_imbalance"].rolling(20).mean()
            rolling_std = df["order_imbalance"].rolling(20).std()
            df["order_imbalance_norm"] = (df["order_imbalance"] - rolling_mean) / (
                rolling_std + 1e-10
            )

        # 5. Market regime features
        # Trend vs mean reversion regime
        if "adx" in df.columns:
            df["trend_regime"] = (df["adx"] > 25).astype(int)  # Strong trend if ADX > 25

        # Volatility clustering
        df["volatility_cluster"] = (df["volatility"] > df["volatility"].shift(1)).astype(int)

        # 6. Signal context features
        # How many consecutive signals
        df["signal_consecutive"] = df["primary_signal"].rolling(5).sum()

        # Time since last signal - optimized version
        # Create a Series with the last signal time for each row
        signal_mask = df["primary_signal"] == 1
        if signal_mask.any():
            # Forward fill the last signal time
            last_signal_times = df.index.where(signal_mask)
            # Use forward fill to propagate last signal time forward
            last_signal_times_ffilled = pd.Series(last_signal_times, index=df.index).ffill()
            # Calculate hours since last signal
            # Convert to Timedelta and get total seconds for each element
            time_diffs = df.index - last_signal_times_ffilled

            def get_hours(td):
                if pd.isna(td):
                    return 24.0
                if hasattr(td, "total_seconds"):
                    return td.total_seconds() / 3600.0
                try:
                    # Try to convert to float (assuming it might be seconds/nanoseconds)
                    # If index is timestamp (int/float), difference is number
                    val = float(td)
                    # Heuristic: if value is huge, it's likely ms or ns
                    if val > 1e12:  # nanoseconds
                        return val / 1e9 / 3600.0
                    elif val > 1e9:  # milliseconds
                        return val / 1000.0 / 3600.0
                    else:  # seconds
                        return val / 3600.0
                except (ValueError, TypeError):
                    return 24.0

            df["time_since_signal"] = time_diffs.apply(get_hours)
            # Fill any remaining NaN values with 24 hours
            df["time_since_signal"] = df["time_since_signal"].fillna(24)
        else:
            df["time_since_signal"] = 24

        # Normalize time since signal
        # We start collecting new features here to avoid fragmentation
        new_features = {}

        new_features["time_since_signal_norm"] = df["time_since_signal"].clip(upper=24) / 24

        # 7. Additional meta-features as requested
        # Volatility Z-Score: How many standard deviations current volatility is from its mean
        if "volatility" in df.columns:
            # Calculate rolling mean and std of volatility
            vol_mean = df["volatility"].rolling(window=100).mean()
            vol_std = df["volatility"].rolling(window=100).std()
            new_features["volatility_zscore"] = (df["volatility"] - vol_mean) / (vol_std + 1e-10)
            logger.debug("Added volatility_zscore feature")

        # Volume Shock: Abnormal volume relative to recent history
        if "volume" in df.columns:
            # Calculate rolling statistics for volume
            volume_mean = df["volume"].rolling(window=20).mean()
            volume_std = df["volume"].rolling(window=20).std()
            # Volume shock is how many standard deviations current volume is from mean
            new_features["volume_shock"] = (df["volume"] - volume_mean) / (volume_std + 1e-10)
            # Also create a binary indicator for extreme volume shocks (> 2 std devs)
            new_features["volume_shock_extreme"] = (new_features["volume_shock"].abs() > 2).astype(
                int
            )
            logger.debug("Added volume_shock and volume_shock_extreme features")

        # Batch add new features to avoid fragmentation
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_features_df], axis=1)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Hour, day of week, month
        hour = df.index.hour
        day_of_week = df.index.dayofweek
        month = df.index.month

        # Cyclical encoding (sin/cos for periodicity)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        # Create a new DataFrame with all time features to avoid fragmentation
        time_features = pd.DataFrame(
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "day_sin": day_sin,
                "day_cos": day_cos,
            },
            index=df.index,
        )

        # Concatenate with original df to avoid fragmentation
        df = pd.concat([df, time_features], axis=1)

        return df

    def _apply_aggressive_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning to remove NaN and Inf values.

        Steps:
        1. Replace all np.inf and -np.inf with NaN
        2. Impute NaNs (Forward Fill)
        3. Drop remaining NaNs (at the beginning) - NO bfill to avoid leakage

        NOTE: We do NOT drop rows in live trading to prevent "silent failures"
        where the strategy goes blind due to a single NaN.

        Args:
            df: DataFrame with engineered features

        Returns:
            Cleaned DataFrame with no NaN or Inf values
        """
        logger.info(f"Applying cleaning: {len(df)} rows")

        # Make a copy to avoid modifying the original
        result = df.copy()

        # Step 1: Replace all inf values with NaN
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        # Avoid recursion error in pandas replace by iterating columns
        for col in numeric_cols:
            # Check if column has any infinite values before attempting replacement
            # This is safer and often faster than attempting replacement on everything
            try:
                if np.isinf(result[col]).any():
                    result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            except Exception:
                # Fallback for safe handling
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)

        # Step 2: Impute NaNs
        # Count NaNs before
        nan_count_before = result.isnull().sum().sum()

        if nan_count_before > 0:
            logger.warning(
                f"Found {nan_count_before} NaNs. Applying strict cleaning (ffill + dropna)."
            )

            # Forward fill (propagate last valid value)
            result = result.ffill()

            # STAGE 3 FIX: Removed bfill() to prevent future data leakage.
            # Instead of backfilling, we drop the initial rows that contain NaNs.
            # This reduces dataset size slightly but guarantees 0 leakage.

            # Check if any NaNs remain (these would be at the start)
            if result.isnull().any().any():
                rows_before = len(result)
                result = result.dropna()
                rows_dropped = rows_before - len(result)
                logger.info(f"Dropped {rows_dropped} rows with initial NaNs to prevent leakage.")
            else:
                logger.info("No initial NaNs found after ffill.")

            nan_count_after = result.isnull().sum().sum()
            if nan_count_after == 0:
                logger.info("NaN cleaning successful.")
            else:
                logger.error(f"NaN cleaning failed! {nan_count_after} NaNs remain.")

        return result

    def _remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Preserve OHLCV columns
        base_cols = ["open", "high", "low", "close", "volume"]

        # Get numeric columns only, excluding base OHLCV columns
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns if col not in base_cols
        ]

        if len(numeric_cols) < 2:
            return df

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()

        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Drop features with high correlation (never drop base OHLCV columns)
        to_drop = [
            column
            for column in upper_tri.columns
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
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        # Don't scale OHLCV columns
        base_cols = ["open", "high", "low", "close", "volume"]
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
        logger.info(
            f"Fitting scaler ({self.config.scaling_method}) on {len(feature_cols)} features"
        )
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

    def save_scaler(self, path: str | None = None) -> None:
        """
        Save fitted scaler to file for production use.

        Args:
            path: Path to save scaler (e.g., "models/scaler.joblib")
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted! Call fit_transform() first.")

        path = path or str(config().paths.models_dir / "scaler.joblib")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scaler_data = {
            "scaler": self.scaler,
            "feature_cols": self._scaled_feature_cols,
            "feature_names": self.feature_names,
            "config": {
                "scaling_method": self.config.scaling_method,
                "correlation_threshold": self.config.correlation_threshold,
            },
        }

        joblib.dump(scaler_data, output_path)
        logger.info(f"Scaler saved to {output_path}")

    def load_scaler(self, path: str | None = None) -> None:
        """
        Load pre-fitted scaler from file.

        Args:
            path: Path to load scaler from
        """
        path = path or str(config().paths.models_dir / "scaler.joblib")
        input_path = Path(path)
        # Use mmap_mode='r' to share memory across processes (crucial for Hyperopt)
        try:
            scaler_data = joblib.load(input_path, mmap_mode="r")
        except Exception:
            # Fallback if mmap fails (e.g. compressed file)
            logger.warning(f"Failed to mmap scaler from {path}, loading into RAM")
            scaler_data = joblib.load(input_path)

        self.scaler = scaler_data["scaler"]
        self._scaled_feature_cols = scaler_data["feature_cols"]
        self.feature_names = scaler_data["feature_names"]
        self._is_fitted = True

        logger.info(f"Scaler loaded from {input_path} ({len(self.feature_names)} features)")

    def get_feature_names(self) -> list[str]:
        """Get list of generated feature names."""
        return self.feature_names

    def is_fitted(self) -> bool:
        """Check if scaler has been fitted."""
        return self._is_fitted

    @classmethod
    def generate_indicators_for_freqtrade(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Generate indicators using FeatureEngineer for Freqtrade strategy.
        Ensures consistent feature engineering between training and live trading.

        Args:
            dataframe: Input OHLCV dataframe

        Returns:
            Dataframe with added indicators (unscaled)
        """
        # Create default config suitable for indicators
        config_obj = FeatureConfig(
            include_price_features=True,
            include_volume_features=True,
            include_momentum_features=True,
            include_volatility_features=True,
            include_trend_features=True,
            include_time_features=False,  # Time features often cause issues in live if not careful
            scale_features=False,  # Indicators should be raw
            remove_correlated=False,  # Keep all
        )
        engineer = cls(config_obj)
        return engineer.prepare_data(dataframe)
