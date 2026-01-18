#!/usr/bin/env python3
"""
Quick test of feature engineering with new indicators.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.training.feature_engineering import FeatureConfig, FeatureEngineer


def test_feature_engineering():
    print("Testing Feature Engineering with BTC/USDT data...")

    # Load data
    data_path = Path("user_data/data/binance/BTC_USDT-5m.feather")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    df = pd.read_feather(data_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        # Assume first column is timestamp
        df.set_index(df.columns[0], inplace=True)
        df.index = pd.to_datetime(df.index, unit='ms')

    print(f"Loaded {len(df)} candles")
    print(f"Columns: {df.columns.tolist()}")

    # Ensure we have OHLCV columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    # Use recent data for quick test
    df_test = df.tail(1000).copy()

    # Configure feature engineering with all indicators
    config = FeatureConfig(
        include_price_features=True,
        include_volume_features=True,
        include_momentum_features=True,
        include_volatility_features=True,
        include_trend_features=True,
        include_meta_labeling_features=True,
        enforce_stationarity=False,  # Disable for quick test
        scale_features=False,  # Disable scaling for inspection
        remove_correlated=False
    )

    engineer = FeatureEngineer(config)

    print("\nRunning fit_transform (training mode)...")
    try:
        features = engineer.fit_transform(df_test)
        print("✅ Feature engineering successful!")
        print(f"Generated {len(features.columns)} features")

        # Check for NaN values
        nan_cols = features.columns[features.isnull().any()].tolist()
        if nan_cols:
            nan_counts = features[nan_cols].isnull().sum()
            print(f"⚠️  Found NaN values in {len(nan_cols)} columns:")
            for col, count in nan_counts.items():
                print(f"   {col}: {count} NaN values ({count/len(features)*100:.1f}%)")
        else:
            print("✅ No NaN values found!")

        # Check for Inf values
        numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
        inf_mask = features[numeric_cols].applymap(lambda x: x in [float('inf'), float('-inf')])
        inf_cols = numeric_cols[inf_mask.any()].tolist()
        if inf_cols:
            print(f"⚠️  Found Inf values in {len(inf_cols)} columns")
        else:
            print("✅ No Inf values found!")

        # Show some sample features
        print("\nSample features (first 5 rows):")
        feature_cols = [col for col in features.columns if col not in required_cols]
        if feature_cols:
            print(features[feature_cols[:10]].head())

        # Validate features
        print("\nRunning feature validation...")
        is_valid, issues = engineer.validate_features(features, fix_issues=False, raise_on_error=False)
        if is_valid:
            print("✅ Feature validation passed!")
        else:
            print("❌ Feature validation failed!")
            for warning in issues.get('warnings', []):
                print(f"   - {warning}")

    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_feature_engineering()
    sys.exit(0 if success else 1)
