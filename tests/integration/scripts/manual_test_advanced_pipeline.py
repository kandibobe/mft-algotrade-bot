"""
Test script for Advanced Trading Pipeline.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.ml.training.advanced_pipeline import (
    AdvancedFeatureEngineer,
    AdvancedTradingPipeline,
    DataPreprocessor,
    FeatureSelector,
    TradingMetrics,
    TripleBarrierWithPurging,
    WalkForwardValidator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_test_data():
    """Load test data from feather file."""
    # Try different files
    candidates = [
        "user_data/data/binance/BNB_USDT-5m.feather",
        "user_data/data/binance/BTC_USDT-5m.feather",
        "user_data/data/binance/ETH_USDT-5m.feather",
        "user_data/data/BTC_USDT_1h.parquet"
    ]

    data_path = None
    for p in candidates:
        if Path(p).exists():
            data_path = Path(p)
            break

    if not data_path:
        raise FileNotFoundError(f"Test data not found. Tried: {candidates}")

    print(f"Loading test data from {data_path}")

    if data_path.suffix == '.feather':
        df = pd.read_feather(data_path)
    else:
        df = pd.read_parquet(data_path)

    # Ensure datetime index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df.columns = df.columns.str.lower()

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Sort by index
    df = df.sort_index()

    # Take only recent data for faster testing
    # Use at least 1000 candles for meaningful testing
    # min_candles = 1000
    # if len(df) > min_candles:
    #     df = df.tail(min_candles)

    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {df.columns.tolist()}")

    return df

def test_individual_stages(df):
    """Test each stage individually."""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL STAGES")
    print("="*60)

    # Stage 1: Data Preprocessing
    print("\n1. Testing DataPreprocessor...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    print(f"   Original shape: {df.shape}")
    print(f"   Processed shape: {df_processed.shape}")
    print(f"   Added columns: {set(df_processed.columns) - set(df.columns)}")

    # Stage 2: Feature Engineering
    print("\n2. Testing AdvancedFeatureEngineer...")
    feature_engineer = AdvancedFeatureEngineer()
    df_features = feature_engineer.engineer_features(df_processed)
    print(f"   Features shape: {df_features.shape}")
    print(f"   Number of features: {len(df_features.columns)}")

    # Show some example features
    example_features = [col for col in df_features.columns if 'lag' in col or 'rsi' in col][:5]
    print(f"   Example features: {example_features}")

    # Stage 3: Labeling
    print("\n3. Testing TripleBarrierWithPurging...")
    labeler = TripleBarrierWithPurging()
    labels = labeler.create_labels(df_processed)
    print(f"   Labels shape: {labels.shape}")
    print(f"   Label distribution: {labels.value_counts().to_dict()}")

    # Stage 4: Feature Selection
    print("\n4. Testing FeatureSelector...")
    # Align features and labels
    valid_mask = labels.notna()
    X = df_features[valid_mask]
    y = labels[valid_mask].astype(int)

    feature_selector = FeatureSelector()
    X_selected = feature_selector.select_features(X, y)
    print(f"   Original features: {X.shape[1]}")
    print(f"   Selected features: {X_selected.shape[1]}")

    # Stage 5: Walk-Forward Validation
    print("\n5. Testing WalkForwardValidator...")
    validator = WalkForwardValidator()
    folds = validator.create_folds(X_selected, y)
    print(f"   Created {len(folds)} folds")

    # Test with a simple model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    cv_results = validator.cross_validate(X_selected, y, model, accuracy_score)
    print(f"   Average test accuracy: {np.mean(cv_results['test_score']):.2%}")

    # Stage 6: Trading Metrics
    print("\n6. Testing TradingMetrics...")
    # Train model on full data for metrics
    model.fit(X_selected, y)
    predictions = model.predict(X_selected)

    metrics_calculator = TradingMetrics()
    metrics = metrics_calculator.calculate_metrics(
        df_processed.loc[X_selected.index],
        pd.Series(predictions, index=X_selected.index),
        y
    )

    print(f"   Precision: {metrics.get('precision', 0):.2%}")
    print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    print(f"   Total Return: {metrics.get('total_return', 0):.2%}")

    return {
        'df_processed': df_processed,
        'df_features': df_features,
        'labels': labels,
        'X_selected': X_selected,
        'y': y,
        'cv_results': cv_results,
        'metrics': metrics
    }

def test_complete_pipeline(df):
    """Test the complete pipeline."""
    print("\n" + "="*60)
    print("TESTING COMPLETE PIPELINE")
    print("="*60)

    pipeline = AdvancedTradingPipeline()
    results = pipeline.run(df)

    print("\nPipeline Results Summary:")
    print(pipeline.get_summary())

    return results

def main():
    """Main test function."""
    try:
        # Load test data
        df = load_test_data()

        # Test individual stages
        stage_results = test_individual_stages(df)

        # Test complete pipeline
        pipeline_results = test_complete_pipeline(df)

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)

        # Print final summary
        print("\nFINAL SUMMARY:")
        print("- Data preprocessing: ✓")
        print(f"- Feature engineering: {stage_results['df_features'].shape[1]} features generated")
        print(f"- Labeling: {stage_results['labels'].value_counts().get(1, 0)} buy signals")
        print(f"- Feature selection: {stage_results['X_selected'].shape[1]} features selected")
        print(f"- Walk-Forward Validation: {len(stage_results['cv_results']['test_score'])} folds")
        print(f"- Trading metrics: Precision = {stage_results['metrics'].get('precision', 0):.2%}")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    success = main()
    sys.exit(0 if success else 1)
