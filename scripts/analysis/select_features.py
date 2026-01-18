#!/usr/bin/env python3
"""
Feature Selection Script for Stoic Ensemble Strategy

PHASE 2: Feature Selection (The Diet)
The model is likely confused by too many noisy features.

Logic:
1. Train a temporary XGBoost model on the first 3 months of data.
2. Use SHAP values or feature_importances_ to rank features.
3. Drop the bottom 50% of features.
4. Drop features with Correlation > 0.95 (Collinearity).
5. Save the "Golden Feature List" to user_data/selected_features.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Try to import optional dependencies
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not installed. Install with: pip install xgboost")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. Install with: pip install shap")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import get_ohlcv
from src.ml.training.feature_engineering import FeatureConfig, FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    pair: str = "BTC/USDT", months: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for feature selection.

    Args:
        pair: Trading pair to load
        months: Number of months of data to use

    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    logger.info(f"Loading data for {pair} (first {months} months)")

    # Load data - get first N months of data
    # For simplicity, we'll just load a large number of candles
    # 3 months of 5m data = 3 * 30 * 24 * 12 = 25920 candles
    # Let's load a bit more to be safe
    limit = months * 30 * 24 * 12

    df = get_ohlcv(pair=pair, timeframe="5m", exchange="binance")

    # Take first N rows (chronological)
    df = df.head(limit)

    if df.empty:
        raise ValueError(f"No data loaded for {pair}")

    logger.info(f"Loaded {len(df)} rows for {pair}")

    # Engineer features
    config = FeatureConfig(
        enforce_stationarity=True,
        use_log_returns=True,
        include_price_features=True,
        include_volume_features=True,
        include_momentum_features=True,
        include_volatility_features=True,
        include_trend_features=True,
        include_meta_labeling_features=True,
        scale_features=False,  # Don't scale for feature selection
        remove_correlated=False,  # We'll handle correlation removal ourselves
    )

    engineer = FeatureEngineer(config)
    features_df = engineer.fit_transform(df)

    # For feature selection, use simple binary classification: predict if next candle will be up
    # This is simpler and more reliable for feature importance analysis
    # We'll predict if close price will increase in the next N candles
    lookahead = 5  # Predict 5 candles ahead (25 minutes)

    # Use original close price for return calculation (not the log cumulative sum)
    # The feature engineering saves original close as 'close_original'
    if "close_original" in features_df.columns:
        close_price = features_df["close_original"]
    else:
        close_price = features_df["close"]

    # Calculate future returns
    future_returns = close_price.shift(-lookahead) / close_price - 1

    # Create binary labels: 1 if future return > 0, else 0
    # First drop NaN values from future_returns
    valid_mask = ~future_returns.isna()
    future_returns = future_returns[valid_mask]
    features_df = features_df[valid_mask]

    # Now create labels
    labels = (future_returns > 0).astype(int)

    # Fill any remaining NaN in features_df with 0
    features_df = features_df.fillna(0)

    # SAFETY CAST: Ensure all numeric columns are float32 to prevent IntCastingNaNError
    # This is important because XGBoost may try to convert NaN/Inf values to integers
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].astype(np.float32)

    logger.info(f"Label distribution: {labels.value_counts().to_dict()}")

    # Check if we have enough data
    if len(features_df) < 100:
        raise ValueError(
            f"Not enough data after cleaning: {len(features_df)} samples. Need at least 100."
        )

    logger.info(f"Prepared data: {len(features_df)} samples, {len(features_df.columns)} features")

    return features_df, labels


def select_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    use_shap: bool = True,
    top_percentile: float = 0.5,  # Keep top 50%
) -> list[str]:
    """
    Select features using XGBoost feature importance or SHAP values.

    Args:
        X: Feature matrix
        y: Labels
        use_shap: Whether to use SHAP values (more accurate but slower)
        top_percentile: Percentage of top features to keep (0.0 to 1.0)

    Returns:
        List of selected feature names

    Raises:
        ValueError: If only one class exists in the labels
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required for feature selection")

    logger.info(f"Training XGBoost model for feature selection (SHAP: {use_shap})")

    # CLASS BALANCE CHECK: Ensure we have both classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        error_msg = (
            f"Data lacks variety. Only one class found in labels: {unique_classes}. "
            f"Try downloading a longer timerange (e.g., 6 months) to capture both Up and Down trends."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # SAFETY CAST: Ensure data types are float32 before passing to XGBoost
    # This prevents IntCastingNaNError when XGBoost tries to convert NaN/Inf to integers
    X_train_numeric = X_train.select_dtypes(include=np.number)
    X_val_numeric = X_val.select_dtypes(include=np.number)
    X_train = X_train_numeric.astype(np.float32)
    X_val = X_val_numeric.astype(np.float32)

    # Train XGBoost model for binary classification
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    logger.info(f"Model accuracy: {accuracy:.3f}")
    logger.info("Classification Report:")
    logger.info(f"\n{classification_report(y_val, y_pred)}")

    # Get feature importance
    if use_shap and SHAP_AVAILABLE:
        logger.info("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # For binary classification, shap_values is a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Calculate mean absolute SHAP value per feature
        importance_df = pd.DataFrame(
            {"feature": list(X_train.columns), "importance": np.abs(shap_values).mean(axis=0)}
        )
        importance_type = "SHAP"
    else:
        # Use built-in feature importance
        importance_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        )
        importance_type = "Gain"
        if use_shap and not SHAP_AVAILABLE:
            logger.warning("SHAP not available, falling back to feature_importances_")

    # Sort by importance
    importance_df = importance_df.sort_values("importance", ascending=False)

    # Select top features
    n_top = int(len(importance_df) * top_percentile)
    selected_features = importance_df.head(n_top)["feature"].tolist()

    logger.info(
        f"Selected {len(selected_features)} out of {len(X.columns)} features "
        f"(top {top_percentile * 100:.0f}%) using {importance_type}"
    )
    logger.info(f"Top 10 features: {selected_features[:10]}")

    return selected_features


def remove_correlated_features(
    X: pd.DataFrame, selected_features: list[str], correlation_threshold: float = 0.95
) -> list[str]:
    """
    Remove highly correlated features.

    Args:
        X: Feature matrix
        selected_features: List of feature names to consider
        correlation_threshold: Correlation threshold above which to remove features

    Returns:
        List of features after removing correlated ones
    """
    if not selected_features:
        return []

    # Calculate correlation matrix for selected features
    corr_matrix = X[selected_features].corr().abs()

    # Find features to remove
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []

    for column in upper_tri.columns:
        if column in to_drop:
            continue
        # Find features highly correlated with this one
        correlated = upper_tri[column][upper_tri[column] > correlation_threshold].index.tolist()
        to_drop.extend(correlated)

    # Remove duplicates
    to_drop = list(set(to_drop))

    # Keep features that are not in to_drop
    final_features = [f for f in selected_features if f not in to_drop]

    if to_drop:
        logger.info(
            f"Removed {len(to_drop)} highly correlated features "
            f"(correlation > {correlation_threshold}): {to_drop}"
        )

    logger.info(
        f"Final feature count: {len(final_features)} "
        f"(removed {len(selected_features) - len(final_features)} correlated features)"
    )

    return final_features


def save_selected_features(features: list[str], output_path: Path) -> None:
    """
    Save selected features to JSON file.

    Args:
        features: List of selected feature names
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_data = {
        "selected_features": features,
        "feature_count": len(features),
        "description": "Golden Feature List selected by XGBoost importance and correlation filtering",
        "selection_criteria": {
            "method": "XGBoost feature importance with correlation filtering",
            "top_percentile": 0.5,
            "correlation_threshold": 0.95,
            "data_period": "3 months",
            "pair": "BTC/USDT",
        },
    }

    with open(output_path, "w") as f:
        json.dump(feature_data, f, indent=2)

    logger.info(f"Saved {len(features)} selected features to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Select features for ML trading model")
    parser.add_argument(
        "--pair", type=str, default="BTC/USDT", help="Trading pair to use for feature selection"
    )
    parser.add_argument("--months", type=int, default=3, help="Number of months of data to use")
    parser.add_argument(
        "--use-shap",
        action="store_true",
        default=True,
        help="Use SHAP values for feature importance (more accurate)",
    )
    parser.add_argument(
        "--no-shap",
        action="store_false",
        dest="use_shap",
        help="Don't use SHAP values, use built-in feature importance",
    )
    parser.add_argument(
        "--top-percentile",
        type=float,
        default=0.5,
        help="Top percentile of features to keep (0.0 to 1.0)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for removing correlated features",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="user_data/selected_features.json",
        help="Output path for selected features JSON file",
    )

    args = parser.parse_args()

    # Check dependencies
    if not XGB_AVAILABLE:
        logger.error("xgboost is required but not installed. Install with: pip install xgboost")
        sys.exit(1)

    if args.use_shap and not SHAP_AVAILABLE:
        logger.warning("SHAP requested but not installed. Falling back to feature_importances_")
        logger.info("Install SHAP with: pip install shap")
        args.use_shap = False

    try:
        # Load and prepare data
        features_df, labels = load_and_prepare_data(args.pair, args.months)

        # Separate features from OHLCV columns
        ohlcv_cols = ["open", "high", "low", "close", "volume", "close_original"]
        feature_cols = [col for col in features_df.columns if col not in ohlcv_cols]

        if not feature_cols:
            logger.error("No features generated. Check feature engineering configuration.")
            sys.exit(1)

        X = features_df[feature_cols]
        y = labels

        # Select features by importance
        selected_features = select_features_by_importance(
            X, y, use_shap=args.use_shap, top_percentile=args.top_percentile
        )

        # Remove correlated features
        final_features = remove_correlated_features(
            X, selected_features, correlation_threshold=args.correlation_threshold
        )

        # Save selected features
        output_path = Path(args.output)
        save_selected_features(final_features, output_path)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE SELECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Initial features: {len(feature_cols)}")
        logger.info(f"After importance selection: {len(selected_features)}")
        logger.info(f"After correlation filtering: {len(final_features)}")
        logger.info(
            f"Reduction: {len(feature_cols) - len(final_features)} features removed "
            f"({(len(feature_cols) - len(final_features)) / len(feature_cols) * 100:.1f}%)"
        )
        logger.info(f"\nSelected features saved to: {output_path}")

    except Exception as e:
        logger.error(f"Feature selection failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()