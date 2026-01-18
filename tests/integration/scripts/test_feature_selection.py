#!/usr/bin/env python3
"""Test script to debug feature selection."""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import get_ohlcv
from src.ml.training.feature_engineering import FeatureConfig, FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_load_data():
    """Test loading and feature engineering."""
    logger.info("Testing data loading...")

    # Load small amount of data
    df = get_ohlcv(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="5m",
        use_cache=True
    )

    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"First few rows:\n{df.head()}")

    # Try feature engineering
    config = FeatureConfig(
        enforce_stationarity=True,
        use_log_returns=True,
        include_price_features=True,
        include_volume_features=True,
        include_momentum_features=True,
        include_volatility_features=True,
        include_trend_features=True,
        include_meta_labeling_features=True,
        scale_features=False,
        remove_correlated=False,
    )

    engineer = FeatureEngineer(config)

    try:
        logger.info("Attempting fit_transform...")
        features_df = engineer.fit_transform(df.head(1000))  # Only first 1000 rows
        logger.info("Feature engineering successful!")
        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Features columns: {features_df.columns.tolist()}")

        # Check for NaN
        nan_cols = features_df.columns[features_df.isnull().any()].tolist()
        logger.info(f"Columns with NaN: {nan_cols}")

        if nan_cols:
            logger.info("Filling NaN with 0...")
            features_df = features_df.fillna(0)

        return features_df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    test_load_data()
