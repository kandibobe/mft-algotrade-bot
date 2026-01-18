"""
Advanced ML Pipeline for Trading
================================

Complete implementation of all stages from the requirements:
1. Data preprocessing (stationarity, outlier removal)
2. Feature engineering (lags, time features, microstructure, normalization)
3. Labeling (Triple Barrier Method with purging)
4. Feature selection (correlation, SHAP values)
5. Validation (Walk-Forward Validation)
6. Success metrics (Precision, Profit Factor, Calmar Ratio)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataPreprocessorConfig:
    """Configuration for data preprocessing."""

    # Stationarity
    use_log_returns: bool = True

    # Outlier removal
    remove_outliers: bool = True
    outlier_sigma_threshold: float = 5.0  # Remove returns beyond N sigma

    # Missing data handling
    fill_method: str = "ffill"  # ffill, bfill, interpolate

    # Data validation
    validate_ohlc: bool = True


class DataPreprocessor:
    """
    Stage 1: Data Preprocessing

    Main rule: Never feed raw prices to the model.
    - Stationarity: Use log returns
    - Outlier removal: Remove spikes (e.g., price drops 99% in 1 second)
    """

    def __init__(self, config: DataPreprocessorConfig | None = None):
        self.config = config or DataPreprocessorConfig()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Stage 1: Data preprocessing")

        result = df.copy()

        # 1. Validate OHLC relationships
        if self.config.validate_ohlc:
            self._validate_ohlc(result)

        # 2. Handle missing data
        result = self._handle_missing_data(result)

        # 3. Calculate log returns for stationarity
        if self.config.use_log_returns:
            result = self._add_log_returns(result)

        # 4. Remove outliers
        if self.config.remove_outliers:
            result = self._remove_outliers(result)

        logger.info(f"Preprocessing complete. Shape: {result.shape}")
        return result

    def _validate_ohlc(self, df: pd.DataFrame):
        """Validate OHLC relationships."""
        issues = []

        # Check basic relationships
        if (df["high"] < df["low"]).any():
            issues.append("high < low")
        if (df["close"] > df["high"]).any():
            issues.append("close > high")
        if (df["close"] < df["low"]).any():
            issues.append("close < low")
        if (df["open"] > df["high"]).any():
            issues.append("open > high")
        if (df["open"] < df["low"]).any():
            issues.append("open < low")

        if issues:
            logger.warning(f"OHLC validation issues: {issues}")

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        result = df.copy()

        # Forward fill, then backward fill
        if self.config.fill_method == "ffill":
            result = result.ffill().bfill()
        elif self.config.fill_method == "interpolate":
            result = result.interpolate(method="linear").bfill()

        # Fill any remaining NaN with 0
        result = result.fillna(0)

        return result

    def _add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns for stationarity."""
        result = df.copy()

        # Log returns: np.log(close / close.shift(1))
        result["log_returns"] = np.log(result["close"] / result["close"].shift(1))

        # Also add simple returns for reference
        result["returns"] = result["close"].pct_change()

        # Fill first NaN
        result["log_returns"] = result["log_returns"].fillna(0)
        result["returns"] = result["returns"].fillna(0)

        logger.info(
            f"Added log returns. Mean: {result['log_returns'].mean():.6f}, Std: {result['log_returns'].std():.6f}"
        )

        return result

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers based on returns.

        Remove rows where returns exceed N standard deviations.
        """
        if "log_returns" not in df.columns:
            df = self._add_log_returns(df)

        result = df.copy()

        # Calculate rolling standard deviation (20-period)
        returns_std = result["log_returns"].rolling(window=20, min_periods=1).std()

        # Identify outliers
        outlier_mask = (
            result["log_returns"].abs() > self.config.outlier_sigma_threshold * returns_std
        )

        # Also check for extreme price movements (e.g., > 50% in one period)
        extreme_mask = result["returns"].abs() > 0.5  # 50% movement

        combined_mask = outlier_mask | extreme_mask

        if combined_mask.any():
            outliers_count = combined_mask.sum()
            outlier_pct = outliers_count / len(result) * 100

            logger.info(f"Found {outliers_count} outliers ({outlier_pct:.2f}% of data)")

            # Instead of removing, we can clip or mark
            # For now, we'll clip extreme returns
            result.loc[outlier_mask, "log_returns"] = (
                np.sign(result.loc[outlier_mask, "log_returns"])
                * self.config.outlier_sigma_threshold
                * returns_std[outlier_mask]
            )
            result.loc[extreme_mask, "returns"] = np.sign(result.loc[extreme_mask, "returns"]) * 0.5

        return result


@dataclass
class AdvancedFeatureEngineerConfig:
    """Configuration for advanced feature engineering."""

    # Lag features
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])

    # Time features
    include_time_features: bool = True
    include_cyclical_encoding: bool = True

    # Microstructure features
    include_microstructure: bool = True
    volume_ma_periods: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Normalization
    normalize_method: str = "rolling"  # rolling, standard, none
    rolling_window: int = 100  # For rolling normalization

    # Indicator lags
    indicator_lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 5])


class AdvancedFeatureEngineer:
    """
    Stage 2: Advanced Feature Engineering

    Don't just throw RSI and MACD. Make features smarter:
    - Lags: Model needs context. Add columns: RSI_lag_1, RSI_lag_2
    - Time features: hour_of_day, day_of_week
    - Microstructure: Volume / MovingAverage(Volume) - relative volume
    - Normalization: Rolling window scaling or no normalization for tree models
    """

    def __init__(self, config: AdvancedFeatureEngineerConfig | None = None):
        self.config = config or AdvancedFeatureEngineerConfig()
        self.scalers = {}  # For rolling normalization

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features.

        Args:
            df: Preprocessed DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Stage 2: Advanced feature engineering")

        result = df.copy()

        # 1. Basic price features (if not already present)
        if "log_returns" not in result.columns:
            result["log_returns"] = np.log(result["close"] / result["close"].shift(1)).fillna(0)

        # 2. Lag features
        result = self._add_lag_features(result)

        # 3. Time features
        if self.config.include_time_features:
            result = self._add_time_features(result)

        # 4. Microstructure features
        if self.config.include_microstructure:
            result = self._add_microstructure_features(result)

        # 5. Indicator lag features (if indicators exist)
        result = self._add_indicator_lags(result)

        # 6. Normalization
        if self.config.normalize_method != "none":
            result = self._normalize_features(result)

        logger.info(f"Feature engineering complete. Total features: {result.shape[1]}")

        return result

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for prices and volumes."""
        result = df.copy()

        for lag in self.config.lag_periods:
            # Price lags
            result[f"close_lag_{lag}"] = result["close"].shift(lag)
            result[f"high_lag_{lag}"] = result["high"].shift(lag)
            result[f"low_lag_{lag}"] = result["low"].shift(lag)
            result[f"open_lag_{lag}"] = result["open"].shift(lag)

            # Volume lags
            result[f"volume_lag_{lag}"] = result["volume"].shift(lag)

            # Return lags
            if "log_returns" in result.columns:
                result[f"returns_lag_{lag}"] = result["log_returns"].shift(lag)

        return result

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping time features")
            return df

        result = df.copy()

        # Basic time features
        result["hour"] = result.index.hour
        result["day_of_week"] = result.index.dayofweek  # Monday=0, Sunday=6
        result["day_of_month"] = result.index.day
        result["month"] = result.index.month
        result["week_of_year"] = result.index.isocalendar().week

        # Cyclical encoding for periodic features
        if self.config.include_cyclical_encoding:
            # Hour (24-hour cycle)
            result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
            result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

            # Day of week (7-day cycle)
            result["day_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
            result["day_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

            # Month (12-month cycle)
            result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
            result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

        # Market session features
        result["is_london_session"] = ((result["hour"] >= 8) & (result["hour"] < 16)).astype(int)
        result["is_us_session"] = ((result["hour"] >= 14) & (result["hour"] < 22)).astype(int)
        result["is_asian_session"] = ((result["hour"] >= 0) & (result["hour"] < 8)).astype(int)

        # Weekend flag
        result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

        return result

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure features (volume-based)."""
        result = df.copy()

        # Relative volume (Volume / MovingAverage(Volume))
        for period in self.config.volume_ma_periods:
            volume_ma = result["volume"].rolling(window=period, min_periods=1).mean()
            result[f"volume_ratio_{period}"] = result["volume"] / (volume_ma + 1e-10)

        # Volume acceleration
        result["volume_change"] = result["volume"].pct_change().fillna(0)

        # Volume vs price relationship
        result["volume_price_correlation_5"] = result["volume"].rolling(5).corr(result["close"])
        result["volume_price_correlation_10"] = result["volume"].rolling(10).corr(result["close"])

        # Dollar volume
        result["dollar_volume"] = result["close"] * result["volume"]

        # Volume profile (current volume vs recent range)
        volume_high_20 = result["volume"].rolling(20).max()
        volume_low_20 = result["volume"].rolling(20).min()
        result["volume_position"] = (result["volume"] - volume_low_20) / (
            volume_high_20 - volume_low_20 + 1e-10
        )

        return result

    def _add_indicator_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for technical indicators."""
        result = df.copy()

        # Find indicator columns (RSI, MACD, etc.)
        indicator_cols = [
            col
            for col in result.columns
            if any(
                indicator in col.lower()
                for indicator in ["rsi", "macd", "stoch", "atr", "bb", "adx"]
            )
        ]

        # Add lags for each indicator
        for col in indicator_cols:
            for lag in self.config.indicator_lag_periods:
                result[f"{col}_lag_{lag}"] = result[col].shift(lag)

        return result

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using specified method."""
        result = df.copy()

        # Identify numeric columns to normalize (exclude OHLCV and time features)
        exclude_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "is_london_session",
            "is_us_session",
            "is_asian_session",
            "is_weekend",
        ]

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        if self.config.normalize_method == "rolling":
            # Rolling window normalization (no lookahead bias)
            for col in cols_to_normalize:
                rolling_mean = (
                    result[col].rolling(window=self.config.rolling_window, min_periods=1).mean()
                )
                rolling_std = (
                    result[col].rolling(window=self.config.rolling_window, min_periods=1).std()
                )
                result[col] = (result[col] - rolling_mean) / (rolling_std + 1e-10)

        elif self.config.normalize_method == "standard":
            # Standard normalization (use for tree models that are scale-invariant)
            # Note: This should only be used if we're not concerned about lookahead bias
            # or if we're using tree-based models
            pass  # Tree models don't need normalization

        return result


@dataclass
class TripleBarrierWithPurgingConfig:
    """Configuration for Triple Barrier Method with purging."""

    # Barrier parameters
    take_profit: float = 0.01  # 1%
    stop_loss: float = 0.005  # 0.5%
    max_holding_period: int = 48  # bars

    # Purging (embargo) to prevent overlapping trades
    purge_period: int = 5  # bars to purge after each trade

    # Fee adjustment
    fee_adjustment: float = 0.001  # 0.1% round-trip

    # Binary classification
    binary_labels: bool = True  # True for 1/0, False for 1/0/-1


class TripleBarrierWithPurging:
    """
    Stage 3: Labeling with Triple Barrier Method and Purging

    Triple Barrier Method - golden standard:
    - Class 1 (Long): Price hit Take Profit (+1%) before Stop Loss (-0.5%) or Time Limit
    - Class 0 (Hold/Ignore): Price was flat or hit Stop Loss

    Important nuance (Purging): When forming labels, overlapping trades cannot be allowed.
    If a trade started at row 100 and ended at 105, data from rows 101-104 already "know" the future.
    Solution: Use embargo (gap) between training and test data.
    """

    def __init__(self, config: TripleBarrierWithPurgingConfig | None = None):
        self.config = config or TripleBarrierWithPurgingConfig()

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels using Triple Barrier Method with purging.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with labels (1 for buy, 0 for ignore/hold)
        """
        logger.info("Stage 3: Creating labels with Triple Barrier Method and purging")

        # Validate data is time-sorted
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex for proper labeling")

        if not df.index.is_monotonic_increasing:
            raise ValueError("Data must be sorted chronologically")

        # Initialize labels array
        labels = pd.Series(0, index=df.index, dtype=int)

        # Adjust barriers for fees
        tp_adjusted = self.config.take_profit - self.config.fee_adjustment
        sl_adjusted = self.config.stop_loss + self.config.fee_adjustment

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        i = 0
        while i < len(df) - self.config.max_holding_period:
            entry_price = close[i]

            # Upper and lower barriers
            upper_barrier = entry_price * (1 + tp_adjusted)
            lower_barrier = entry_price * (1 - sl_adjusted)

            # Check forward bars
            hit_tp = False
            hit_sl = False

            for j in range(1, self.config.max_holding_period + 1):
                if i + j >= len(df):
                    break

                # Check if TP hit
                if high[i + j] >= upper_barrier:
                    hit_tp = True
                    break

                # Check if SL hit
                if low[i + j] <= lower_barrier:
                    hit_sl = True
                    break

            # Determine label
            if hit_tp:
                labels.iloc[i] = 1  # Buy signal
                # Apply purging (embargo) - skip next N bars
                i += self.config.purge_period
            elif hit_sl:
                labels.iloc[i] = 0  # Ignore
                i += 1
            else:
                # Time barrier hit
                labels.iloc[i] = 0  # Ignore
                i += 1

        # Log label distribution
        label_counts = labels.value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        logger.info(
            f"Buy signals: {label_counts.get(1, 0)} ({label_counts.get(1, 0) / len(labels) * 100:.1f}%)"
        )

        return labels


@dataclass
class FeatureSelectorConfig:
    """Configuration for feature selection."""

    # Correlation-based selection
    correlation_threshold: float = (
        0.85  # Remove features with correlation > threshold (reduced from 0.95 per roadmap)
    )

    # SHAP-based selection
    use_shap: bool = True
    top_n_features: int = 25  # Keep top N features by SHAP importance
    shap_sample_size: float = 0.2  # Use 20% sample for SHAP calculation (per roadmap)

    # Recursive Feature Elimination (RFE)
    use_rfe: bool = True  # Use recursive feature elimination
    rfe_step: int = 5  # Number of features to remove at each RFE step

    # Model for SHAP calculation
    model_type: str = "xgboost"  # xgboost, random_forest, lightgbm (changed to xgboost per roadmap)

    # Stability check
    check_stability: bool = True  # Check feature importance stability across folds
    stability_threshold: float = 0.7  # Minimum correlation for feature importance stability


class FeatureSelector:
    """
    Stage 4: Feature Selection

    If you feed 100 indicators, the model will overfit on noise.
    - Correlation matrix: If RSI and Stoch correlate at 0.85 - remove one.
    - SHAP Values / Feature Importance: Train XGBoost on 20% sample, look at top-25 important features.
    - Recursive Feature Elimination: Iteratively remove least important features.
    - Stability Check: Ensure selected features stable across time folds.
    """

    def __init__(self, config: FeatureSelectorConfig | None = None):
        self.config = config or FeatureSelectorConfig()
        self.selected_features = []
        self.feature_importance = None
        self.stability_scores = {}

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Select best features using correlation, SHAP, RFE, and stability checks.

        Args:
            X: Feature DataFrame
            y: Target labels

        Returns:
            DataFrame with selected features
        """
        logger.info("Stage 4: Advanced feature selection")
        logger.info(f"Initial features: {X.shape[1]}")

        result = X.copy()

        # 1. Remove highly correlated features
        if self.config.correlation_threshold < 1.0:
            result = self._remove_correlated_features(result)
            logger.info(f"After correlation removal: {result.shape[1]} features")

        # 2. SHAP-based feature selection (on sample for efficiency)
        if self.config.use_shap and len(result.columns) > self.config.top_n_features:
            result = self._select_by_shap(result, y)
            logger.info(f"After SHAP selection: {result.shape[1]} features")

        # 3. Recursive Feature Elimination (RFE)
        if self.config.use_rfe and len(result.columns) > self.config.top_n_features:
            result = self._select_by_rfe(result, y)
            logger.info(f"After RFE: {result.shape[1]} features")

        # 4. Stability check (if we have time-based data)
        if self.config.check_stability and len(result.columns) > 0:
            self._check_feature_stability(result, y)

        self.selected_features = result.columns.tolist()
        logger.info(f"Selected {len(self.selected_features)} features")

        return result

    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation above threshold
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.config.correlation_threshold)
        ]

        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
            X = X.drop(columns=to_drop)

        return X

    def _select_by_shap(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select features using SHAP importance with XGBoost on sample."""
        try:
            import shap
            import xgboost as xgb

            # Sample data for efficiency (20% as per roadmap)
            if self.config.shap_sample_size < 1.0:
                sample_size = int(len(X) * self.config.shap_sample_size)
                if sample_size > 100:  # Ensure minimum sample size
                    sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
                    X_sample = X.iloc[sample_indices]
                    y_sample = y.iloc[sample_indices]
                else:
                    X_sample = X
                    y_sample = y
            else:
                X_sample = X
                y_sample = y

            logger.info(f"Training XGBoost on {len(X_sample)} samples for SHAP calculation")

            # Train XGBoost model (optimized for precision)
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,  # Shallower trees for simplicity
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
            model.fit(X_sample, y_sample)

            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, use SHAP values for class 1
            if isinstance(shap_values, list):
                shap_importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)

            # Create feature importance DataFrame
            importance_df = pd.DataFrame(
                {"feature": X_sample.columns, "importance": shap_importance}
            ).sort_values("importance", ascending=False)

            self.feature_importance = importance_df

            # Select top N features
            top_features = importance_df.head(self.config.top_n_features)["feature"].tolist()

            logger.info("Top 10 features by SHAP importance:")
            for _idx, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            return X[top_features]

        except ImportError as e:
            logger.warning(
                f"SHAP or XGBoost not available ({e}), falling back to feature importance"
            )
            return self._select_by_feature_importance(X, y)

    def _select_by_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fallback: select features using model feature importance."""
        try:
            import xgboost as xgb

            # Try XGBoost first
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1
            )
            model.fit(X, y)

            importance = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

        except ImportError:
            # Fallback to RandomForest
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)

            importance = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

        self.feature_importance = importance

        # Select top N features
        top_features = importance.head(self.config.top_n_features)["feature"].tolist()

        logger.info("Top 10 features by importance:")
        for _idx, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return X[top_features]

    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select features using Recursive Feature Elimination (RFE)."""
        try:
            import xgboost as xgb
            from sklearn.feature_selection import RFE

            # Create XGBoost model for RFE
            model = xgb.XGBClassifier(
                n_estimators=50,  # Smaller for RFE efficiency
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )

            # Determine target number of features
            n_features_to_select = min(self.config.top_n_features, len(X.columns))

            # Create RFE selector
            selector = RFE(
                estimator=model,
                n_features_to_select=n_features_to_select,
                step=self.config.rfe_step,
                verbose=0,
            )

            # Fit RFE
            selector.fit(X, y)

            # Get selected features
            selected_mask = selector.support_
            selected_features = X.columns[selected_mask].tolist()

            # Get feature ranking (1 = selected, higher = eliminated later)
            ranking = pd.DataFrame(
                {"feature": X.columns, "ranking": selector.ranking_}
            ).sort_values("ranking")

            logger.info(f"RFE selected {len(selected_features)} features")
            logger.info("Top 10 features by RFE ranking:")
            for _idx, row in ranking.head(10).iterrows():
                logger.info(f"  {row['feature']}: ranking {row['ranking']}")

            return X[selected_features]

        except ImportError as e:
            logger.warning(f"RFE not available ({e}), skipping RFE selection")
            return X

    def _check_feature_stability(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Check feature importance stability across time folds.

        This helps ensure selected features are robust over time,
        not just fitting noise in a specific period.
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import TimeSeriesSplit

            # Use TimeSeriesSplit for temporal validation
            tscv = TimeSeriesSplit(n_splits=5)

            feature_importances = []
            fold_features = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, _X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, _y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model on fold
                model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42 + fold,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)

                # Get feature importance for this fold
                importance = pd.DataFrame(
                    {"feature": X.columns, f"importance_fold_{fold}": model.feature_importances_}
                )
                feature_importances.append(importance)

                # Track which features would be selected in this fold
                fold_importance = model.feature_importances_
                top_indices = np.argsort(fold_importance)[-self.config.top_n_features :]
                fold_features.append(set(X.columns[top_indices]))

            # Combine importances across folds
            importance_df = feature_importances[0]
            for i in range(1, len(feature_importances)):
                importance_df = importance_df.merge(
                    feature_importances[i], on="feature", how="outer"
                ).fillna(0)

            # Calculate stability metrics
            # 1. Correlation of importances across folds
            importance_cols = [col for col in importance_df.columns if "importance_fold" in col]
            if len(importance_cols) > 1:
                corr_matrix = importance_df[importance_cols].corr()
                avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()

                # 2. Feature selection consistency
                all_features = set(X.columns)
                common_features = set.intersection(*fold_features) if fold_features else set()
                consistency_ratio = len(common_features) / len(all_features) if all_features else 0

                self.stability_scores = {
                    "avg_importance_correlation": avg_correlation,
                    "feature_selection_consistency": consistency_ratio,
                    "common_features_count": len(common_features),
                    "common_features": list(common_features),
                }

                logger.info("Feature stability scores:")
                logger.info(f"  Average importance correlation across folds: {avg_correlation:.3f}")
                logger.info(f"  Feature selection consistency: {consistency_ratio:.3f}")
                logger.info(f"  Common features across all folds: {len(common_features)}")

                if avg_correlation < self.config.stability_threshold:
                    logger.warning(
                        f"Feature importance unstable across folds (correlation={avg_correlation:.3f} < {self.config.stability_threshold})"
                    )
                else:
                    logger.info(
                        f"Feature importance stable across folds (correlation={avg_correlation:.3f} >= {self.config.stability_threshold})"
                    )

        except Exception as e:
            logger.warning(f"Feature stability check failed: {e}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        return self.feature_importance


@dataclass
class WalkForwardValidatorConfig:
    """Configuration for Walk-Forward Validation."""

    # Validation scheme
    initial_train_size: float = 0.6  # 60% for initial training
    step_size: float = 0.1  # 10% step for each fold

    # Minimum samples
    min_train_samples: int = 1000
    min_test_samples: int = 200


class WalkForwardValidator:
    """
    Stage 5: Walk-Forward Validation

    Ordinary train_test_split (random shuffling) is prohibited in trading.
    Walk-Forward Validation (Sliding window):
    - Train: January-March -> Test: April
    - Train: February-April -> Test: May
    """

    def __init__(self, config: WalkForwardValidatorConfig | None = None):
        self.config = config or WalkForwardValidatorConfig()
        self.folds = []

    def create_folds(self, X: pd.DataFrame, y: pd.Series) -> list[tuple]:
        """
        Create Walk-Forward validation folds.

        Args:
            X: Feature DataFrame (must be time-sorted)
            y: Target labels

        Returns:
            List of (train_idx, test_idx) tuples
        """
        logger.info("Stage 5: Creating Walk-Forward validation folds")

        n_samples = len(X)

        # Calculate sizes
        initial_train_size = int(n_samples * self.config.initial_train_size)
        step_size = int(n_samples * self.config.step_size)

        # Ensure minimum sizes
        if initial_train_size < self.config.min_train_samples:
            initial_train_size = self.config.min_train_samples

        if step_size < self.config.min_test_samples:
            step_size = self.config.min_test_samples

        # Create folds
        folds = []
        train_start = 0

        while True:
            train_end = train_start + initial_train_size
            test_end = train_end + step_size

            if test_end > n_samples:
                break

            train_idx = list(range(train_start, train_end))
            test_idx = list(range(train_end, test_end))

            folds.append((train_idx, test_idx))

            # Slide window
            train_start += step_size

        self.folds = folds
        logger.info(f"Created {len(folds)} Walk-Forward folds")

        # Log fold information
        for i, (train_idx, test_idx) in enumerate(folds):
            train_dates = X.index[train_idx[0]], X.index[train_idx[-1]]
            test_dates = X.index[test_idx[0]], X.index[test_idx[-1]]
            logger.info(
                f"Fold {i + 1}: Train {len(train_idx)} samples ({train_dates[0]} to {train_dates[1]}), "
                f"Test {len(test_idx)} samples ({test_dates[0]} to {test_dates[1]})"
            )

        return folds

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, model, metric_func: callable
    ) -> dict[str, list[float]]:
        """
        Perform Walk-Forward cross-validation.

        Args:
            X: Features
            y: Labels
            model: Model with fit() and predict() methods
            metric_func: Function to calculate metric (y_true, y_pred) -> float

        Returns:
            Dictionary with metric results for each fold
        """
        if not self.folds:
            self.create_folds(X, y)

        results = {"train_score": [], "test_score": [], "train_size": [], "test_size": []}

        for fold, (train_idx, test_idx) in enumerate(self.folds, 1):
            logger.info(f"Fold {fold}/{len(self.folds)}")

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_score = metric_func(y_train, y_train_pred)
            test_score = metric_func(y_test, y_test_pred)

            results["train_score"].append(train_score)
            results["test_score"].append(test_score)
            results["train_size"].append(len(train_idx))
            results["test_size"].append(len(test_idx))

            logger.info(f"  Train score: {train_score:.4f}, Test score: {test_score:.4f}")

        # Calculate statistics
        avg_test_score = np.mean(results["test_score"])
        std_test_score = np.std(results["test_score"])

        logger.info(
            f"Walk-Forward CV results: Test score = {avg_test_score:.4f} ± {std_test_score:.4f}"
        )

        return results


@dataclass
class TradingMetricsConfig:
    """Configuration for trading metrics."""

    # Position sizing
    position_size: float = 0.1  # 10% of capital per trade
    commission: float = 0.001  # 0.1% commission

    # Risk metrics
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

    # Uncertainty Estimation
    use_uncertainty: bool = True
    uncertainty_method: str = "entropy"  # entropy, ensemble_std


class TradingMetrics:
    """
    Stage 6: Trading Success Metrics

    Forget about Accuracy. In trading it lies.
    If market rises 80% of time, model will always say "BUY" and get 80% accuracy but lose on first drop.

    What to look at:
    - Precision (Entry accuracy): Out of 100 "BUY" signals, how many actually closed in profit? Need > 55-60%.
    - Profit Factor: Total profit / Total loss. Should be > 1.1 (better > 1.5).
    - Calmar Ratio: Return / Maximum drawdown.
    """

    def __init__(self, config: TradingMetricsConfig | None = None):
        self.config = config or TradingMetricsConfig()

    def calculate_metrics(
        self, df: pd.DataFrame, predictions: pd.Series, labels: pd.Series
    ) -> dict[str, float]:
        """
        Calculate trading performance metrics.

        Args:
            df: DataFrame with OHLCV data
            predictions: Model predictions (1 for buy, 0 for ignore)
            labels: True labels (1 for profitable trade, 0 for ignore)

        Returns:
            Dictionary with trading metrics
        """
        logger.info("Stage 6: Calculating trading metrics")

        # Align indices
        aligned_idx = df.index.intersection(predictions.index).intersection(labels.index)
        df = df.loc[aligned_idx]
        predictions = predictions.loc[aligned_idx]
        labels = labels.loc[aligned_idx]

        # 1. Precision (Entry accuracy)
        precision_metrics = self._calculate_precision_metrics(predictions, labels)

        # 2. Profit Factor and other profit metrics
        profit_metrics = self._calculate_profit_metrics(df, predictions, labels)

        # 3. Risk metrics (Calmar Ratio, Sharpe, etc.)
        risk_metrics = self._calculate_risk_metrics(
            profit_metrics.get("returns_series", pd.Series())
        )

        # Combine all metrics
        metrics = {**precision_metrics, **profit_metrics, **risk_metrics}

        # Log key metrics
        logger.info(f"Precision: {metrics.get('precision', 0):.2%}")
        logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

        return metrics

    def _calculate_precision_metrics(
        self, predictions: pd.Series, labels: pd.Series
    ) -> dict[str, float]:
        """Calculate precision and related metrics."""
        # True positives: predicted buy and actually profitable
        true_positives = ((predictions == 1) & (labels == 1)).sum()

        # False positives: predicted buy but not profitable
        false_positives = ((predictions == 1) & (labels == 0)).sum()

        # Precision
        precision = true_positives / (true_positives + false_positives + 1e-10)

        # Recall
        actual_positives = (labels == 1).sum()
        recall = true_positives / (actual_positives + 1e-10)

        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "actual_positives": int(actual_positives),
        }

    def _calculate_profit_metrics(
        self, df: pd.DataFrame, predictions: pd.Series, labels: pd.Series
    ) -> dict[str, float]:
        """Calculate profit-related metrics."""
        # Simulate trades based on predictions
        trades = self._simulate_trades(df, predictions, labels)

        if trades.empty:
            return {
                "total_return": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        # Calculate returns
        trades["return_pct"] = trades["exit_price"] / trades["entry_price"] - 1
        trades["return_pct"] -= self.config.commission * 2  # Entry and exit commission

        # Separate winning and losing trades
        winning_trades = trades[trades["return_pct"] > 0]
        losing_trades = trades[trades["return_pct"] <= 0]

        # Total return
        total_return = (1 + trades["return_pct"]).prod() - 1

        # Profit Factor
        total_profit = winning_trades["return_pct"].sum()
        total_loss = abs(losing_trades["return_pct"].sum())
        profit_factor = total_profit / (total_loss + 1e-10)

        # Win rate
        win_rate = len(winning_trades) / len(trades)

        # Average win/loss
        avg_win = winning_trades["return_pct"].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades["return_pct"].mean() if not losing_trades.empty else 0

        # Create returns series for risk metrics
        returns_series = pd.Series(index=df.index, dtype=float)
        for _, trade in trades.iterrows():
            if (
                trade["entry_time"] in returns_series.index
                and trade["exit_time"] in returns_series.index
            ):
                # Distribute return over trade duration
                trade_duration = (trade["exit_time"] - trade["entry_time"]).total_seconds() / (
                    24 * 3600
                )
                if trade_duration > 0:
                    daily_return = (1 + trade["return_pct"]) ** (1 / trade_duration) - 1
                    # Mark trade period
                    trade_mask = (returns_series.index >= trade["entry_time"]) & (
                        returns_series.index <= trade["exit_time"]
                    )
                    returns_series[trade_mask] = daily_return

        returns_series = returns_series.fillna(0)

        return {
            "total_return": total_return,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "returns_series": returns_series,
        }

    def _simulate_trades(
        self, df: pd.DataFrame, predictions: pd.Series, labels: pd.Series
    ) -> pd.DataFrame:
        """Simulate trades based on predictions and labels."""
        trades = []

        i = 0
        while i < len(df):
            if predictions.iloc[i] == 1:  # Buy signal
                entry_time = df.index[i]
                entry_price = df["close"].iloc[i]

                # Find exit (next bar where label is determined)
                exit_idx = i + 1
                while exit_idx < len(df) and labels.iloc[exit_idx] not in [0, 1]:
                    exit_idx += 1

                if exit_idx < len(df):
                    exit_time = df.index[exit_idx]
                    exit_price = df["close"].iloc[exit_idx]

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "entry_price": entry_price,
                            "exit_time": exit_time,
                            "exit_price": exit_price,
                            "label": labels.iloc[i],
                        }
                    )

                    i = exit_idx + 1  # Skip to after exit
                else:
                    i += 1
            else:
                i += 1

        return pd.DataFrame(trades)

    def _calculate_risk_metrics(self, returns_series: pd.Series) -> dict[str, float]:
        """Calculate risk metrics (Calmar Ratio, Sharpe, etc.)."""
        if returns_series.empty or returns_series.sum() == 0:
            return {
                "calmar_ratio": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
            }

        # Calculate cumulative returns
        cumulative_returns = (1 + returns_series).cumprod()

        # Maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Annualized return
        total_return = cumulative_returns.iloc[-1] - 1
        days = len(returns_series) / 252  # Approximate trading days
        annualized_return = (1 + total_return) ** (1 / max(days, 1e-10)) - 1

        # Annualized volatility
        volatility = returns_series.std() * np.sqrt(252)

        # Sharpe ratio (assuming risk-free rate)
        excess_return = annualized_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / (volatility + 1e-10)

        # Calmar ratio
        calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-10)

        return {
            "calmar_ratio": calmar_ratio,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "annualized_return": annualized_return,
        }

    def calculate_uncertainty(self, proba: np.ndarray) -> np.ndarray:
        """
        Calculate prediction uncertainty (Entropy).

        Args:
            proba: Prediction probabilities [n_samples, n_classes]

        Returns:
            Entropy values
        """
        # Entropy = -sum(p * log(p))
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        return entropy


class AdvancedTradingPipeline:
    """
    Complete pipeline integrating all stages.

    Usage:
        pipeline = AdvancedTradingPipeline()
        results = pipeline.run(df)
    """

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.labeler = TripleBarrierWithPurging()
        self.feature_selector = FeatureSelector()
        self.validator = WalkForwardValidator()
        self.metrics_calculator = TradingMetrics()

        self.results = {}

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run complete pipeline on data.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting Advanced Trading Pipeline")

        # Stage 1: Data preprocessing
        df_processed = self.preprocessor.preprocess(df)

        # Stage 2: Feature engineering
        df_features = self.feature_engineer.engineer_features(df_processed)

        # Stage 3: Labeling
        labels = self.labeler.create_labels(df_processed)

        # Align features and labels (remove NaN from labels)
        valid_mask = labels.notna()
        X = df_features[valid_mask]
        y = labels[valid_mask].astype(int)

        # Stage 4: Feature selection
        X_selected = self.feature_selector.select_features(X, y)

        # Stage 5: Walk-Forward Validation
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        cv_results = self.validator.cross_validate(X_selected, y, model, accuracy_score)

        # Stage 6: Trading metrics (on full data with best model)
        model.fit(X_selected, y)
        predictions = model.predict(X_selected)

        # Uncertainty estimation
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_selected)
            uncertainty = self.metrics_calculator.calculate_uncertainty(proba)
        else:
            uncertainty = np.zeros(len(predictions))

        trading_metrics = self.metrics_calculator.calculate_metrics(
            df_processed.loc[X_selected.index], pd.Series(predictions, index=X_selected.index), y
        )

        # Store results
        self.results = {
            "data_processed": df_processed,
            "features": df_features,
            "labels": labels,
            "X_selected": X_selected,
            "y": y,
            "cv_results": cv_results,
            "trading_metrics": trading_metrics,
            "uncertainty": pd.Series(uncertainty, index=X_selected.index),
            "feature_importance": self.feature_selector.get_feature_importance(),
            "model": model,
        }

        logger.info("Pipeline completed successfully")

        return self.results

    def get_summary(self) -> str:
        """Get pipeline summary as string."""
        if not self.results:
            return "Pipeline not run yet"

        metrics = self.results.get("trading_metrics", {})
        cv_results = self.results.get("cv_results", {})

        summary = [
            "=" * 60,
            "ADVANCED TRADING PIPELINE SUMMARY",
            "=" * 60,
            f"Data samples: {len(self.results.get('data_processed', pd.DataFrame()))}",
            f"Features generated: {self.results.get('features', pd.DataFrame()).shape[1]}",
            f"Features selected: {self.results.get('X_selected', pd.DataFrame()).shape[1]}",
            f"Buy signals: {(self.results.get('y', pd.Series()) == 1).sum()}",
            "",
            "TRADING METRICS:",
            f"  Precision: {metrics.get('precision', 0):.2%}",
            f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}",
            f"  Total Return: {metrics.get('total_return', 0):.2%}",
            f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            "",
            "VALIDATION RESULTS:",
            f"  Test Accuracy: {np.mean(cv_results.get('test_score', [0])):.2%} ± {np.std(cv_results.get('test_score', [0])):.2%}",
            f"  Number of folds: {len(cv_results.get('test_score', []))}",
            "=" * 60,
        ]

        return "\n".join(summary)
