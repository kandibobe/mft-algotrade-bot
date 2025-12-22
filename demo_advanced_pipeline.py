"""
Demo script for Advanced Trading Pipeline with synthetic data.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ml.training.advanced_pipeline import (
    AdvancedTradingPipeline,
    DataPreprocessor,
    AdvancedFeatureEngineer,
    TripleBarrierWithPurging,
    FeatureSelector,
    WalkForwardValidator,
    TradingMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def generate_synthetic_data(n_samples=2000):
    """Generate synthetic OHLCV data for testing."""
    print(f"Generating {n_samples} synthetic candles...")
    
    # Generate time index (hourly)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate price with trend and noise
    np.random.seed(42)
    trend = np.linspace(100, 200, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    close = trend + noise
    
    # Generate OHLC with realistic relationships
    open_prices = close + np.random.normal(0, 1, n_samples)
    high = np.maximum(open_prices, close) + np.random.uniform(0.5, 2, n_samples)
    low = np.minimum(open_prices, close) - np.random.uniform(0.5, 2, n_samples)
    
    # Ensure OHLC relationships
    for i in range(n_samples):
        high[i] = max(open_prices[i], close[i], high[i])
        low[i] = min(open_prices[i], close[i], low[i])
    
    # Generate volume (correlated with price movement)
    price_change = np.abs(close - np.roll(close, 1))
    price_change[0] = 0
    volume = 1000 + price_change * 500 + np.random.normal(0, 200, n_samples)
    volume = np.maximum(volume, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    print(f"Generated data from {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df

def demo_individual_stages():
    """Demo each stage individually."""
    print("\n" + "="*60)
    print("DEMONSTRATING INDIVIDUAL STAGES")
    print("="*60)
    
    # Generate synthetic data
    df = generate_synthetic_data(2000)
    
    # Stage 1: Data Preprocessing
    print("\n1. DataPreprocessor Demo:")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    print(f"   Added columns: {[c for c in df_processed.columns if c not in df.columns]}")
    print(f"   Log returns stats: mean={df_processed['log_returns'].mean():.6f}, std={df_processed['log_returns'].std():.6f}")
    
    # Stage 2: Feature Engineering
    print("\n2. AdvancedFeatureEngineer Demo:")
    feature_engineer = AdvancedFeatureEngineer()
    df_features = feature_engineer.engineer_features(df_processed)
    print(f"   Total features generated: {df_features.shape[1]}")
    
    # Show feature categories
    lag_features = [c for c in df_features.columns if 'lag' in c]
    time_features = [c for c in df_features.columns if c in ['hour', 'day_of_week', 'month']]
    volume_features = [c for c in df_features.columns if 'volume' in c]
    
    print(f"   Lag features: {len(lag_features)}")
    print(f"   Time features: {len(time_features)}")
    print(f"   Volume features: {len(volume_features)}")
    
    # Stage 3: Labeling
    print("\n3. TripleBarrierWithPurging Demo:")
    labeler = TripleBarrierWithPurging()
    labels = labeler.create_labels(df_processed)
    print(f"   Labels created: {len(labels)}")
    print(f"   Buy signals (1): {(labels == 1).sum()} ({((labels == 1).sum()/len(labels)*100):.1f}%)")
    print(f"   Ignore signals (0): {(labels == 0).sum()} ({((labels == 0).sum()/len(labels)*100):.1f}%)")
    
    # Stage 4: Feature Selection
    print("\n4. FeatureSelector Demo:")
    # Align features and labels
    valid_mask = labels.notna()
    X = df_features[valid_mask]
    y = labels[valid_mask].astype(int)
    
    feature_selector = FeatureSelector()
    X_selected = feature_selector.select_features(X, y)
    print(f"   Original features: {X.shape[1]}")
    print(f"   Selected features: {X_selected.shape[1]}")
    
    # Show top features
    importance_df = feature_selector.get_feature_importance()
    if importance_df is not None:
        print(f"   Top 5 features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
    
    # Stage 5: Walk-Forward Validation
    print("\n5. WalkForwardValidator Demo:")
    validator = WalkForwardValidator()
    folds = validator.create_folds(X_selected, y)
    print(f"   Created {len(folds)} folds")
    
    if folds:
        # Test with a simple model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        cv_results = validator.cross_validate(X_selected, y, model, accuracy_score)
        print(f"   Average test accuracy: {np.mean(cv_results['test_score']):.2%}")
        print(f"   Accuracy range: {np.min(cv_results['test_score']):.2%} - {np.max(cv_results['test_score']):.2%}")
    
    # Stage 6: Trading Metrics
    print("\n6. TradingMetrics Demo:")
    if folds:
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
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    return {
        'df': df,
        'df_processed': df_processed,
        'df_features': df_features,
        'labels': labels,
        'X_selected': X_selected,
        'y': y
    }

def demo_complete_pipeline():
    """Demo the complete pipeline."""
    print("\n" + "="*60)
    print("DEMONSTRATING COMPLETE PIPELINE")
    print("="*60)
    
    # Generate synthetic data
    df = generate_synthetic_data(1500)
    
    # Create and run pipeline
    pipeline = AdvancedTradingPipeline()
    results = pipeline.run(df)
    
    print("\nPipeline Results Summary:")
    print(pipeline.get_summary())
    
    return results

def main():
    """Main demo function."""
    print("="*60)
    print("ADVANCED TRADING PIPELINE DEMONSTRATION")
    print("="*60)
    print("\nThis demo shows the complete ML pipeline for trading:")
    print("1. Data preprocessing (stationarity, outlier removal)")
    print("2. Feature engineering (lags, time features, microstructure)")
    print("3. Labeling (Triple Barrier Method with purging)")
    print("4. Feature selection (correlation, SHAP values)")
    print("5. Validation (Walk-Forward Validation)")
    print("6. Success metrics (Precision, Profit Factor, Calmar Ratio)")
    
    try:
        # Demo individual stages
        stage_results = demo_individual_stages()
        
        # Demo complete pipeline
        pipeline_results = demo_complete_pipeline()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("\nKey Takeaways:")
        print("✓ All 6 stages implemented according to requirements")
        print("✓ Pipeline handles data preprocessing and feature engineering")
        print("✓ Triple Barrier Method with purging prevents lookahead bias")
        print("✓ Walk-Forward Validation for time-series data")
        print("✓ Trading-specific metrics (not just accuracy)")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
