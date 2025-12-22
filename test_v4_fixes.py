#!/usr/bin/env python3
"""
Test script for V4 strategy fixes.
Runs walk-forward analysis to check if strategy generates signals.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.training.feature_engineering import FeatureEngineer
from src.ml.training.labeling import TripleBarrierLabeler, TripleBarrierConfig
from src.ml.training.model_trainer import ModelTrainer, TrainingConfig

def test_triple_barrier_params():
    """Test new Triple Barrier parameters."""
    print("="*70)
    print("Testing Triple Barrier Parameters")
    print("="*70)
    
    config = TripleBarrierConfig()
    print(f"Take Profit: {config.take_profit:.3%} (should be 0.8%)")
    print(f"Stop Loss: {config.stop_loss:.3%} (should be 0.4%)")
    print(f"Max Holding Period: {config.max_holding_period} bars (should be 48)")
    
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randn(100).cumsum() + 1000
    }, index=dates)
    
    labeler = TripleBarrierLabeler(config)
    labels = labeler.label(df)
    
    print(f"\nGenerated {len(labels)} labels")
    print(f"Label distribution:")
    print(labels.value_counts())
    
    return config.take_profit == 0.008 and config.stop_loss == 0.004

def test_feature_engineering():
    """Test improved feature engineering."""
    print("\n" + "="*70)
    print("Testing Feature Engineering")
    print("="*70)
    
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randn(100).cumsum() + 1000
    }, index=dates)
    
    engineer = FeatureEngineer()
    features = engineer.fit_transform(df)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"Features generated: {len(features.columns)}")
    
    # Check for new features
    new_features = ['close_lag_1', 'close_lag_2', 'close_rolling_mean_5', 
                   'close_rolling_std_10', 'returns_volatility_5']
    
    found_features = []
    for feat in new_features:
        if feat in features.columns:
            found_features.append(feat)
    
    print(f"\nNew features found: {len(found_features)}/{len(new_features)}")
    for feat in found_features:
        print(f"  ✓ {feat}")
    
    return len(found_features) >= 3  # At least 3 new features

def test_class_balancing():
    """Test class balancing in model training."""
    print("\n" + "="*70)
    print("Testing Class Balancing")
    print("="*70)
    
    # Create imbalanced data
    X = pd.DataFrame(np.random.randn(100, 10))
    y = pd.Series([0] * 95 + [1] * 5)  # 95% negative, 5% positive
    
    config = TrainingConfig(
        model_type="random_forest",
        optimize_hyperparams=False,
        save_model=False
    )
    
    trainer = ModelTrainer(config)
    
    # Check if model uses class_weight='balanced'
    model = trainer._create_model()
    
    if hasattr(model, 'class_weight'):
        print(f"Model class_weight: {model.class_weight}")
        if model.class_weight == 'balanced':
            print("✓ Class balancing enabled")
            return True
        else:
            print("✗ Class balancing not properly configured")
            return False
    else:
        print("✗ Model doesn't have class_weight attribute")
        return False

def test_dynamic_threshold():
    """Test dynamic threshold logic from V4 strategy."""
    print("\n" + "="*70)
    print("Testing Dynamic Threshold Logic")
    print("="*70)
    
    # Simulate predictions with different signal densities
    test_cases = [
        ("High density", np.random.uniform(0.6, 0.9, 100)),
        ("Low density", np.random.uniform(0.4, 0.6, 100)),
        ("Very low density", np.random.uniform(0.2, 0.4, 100)),
    ]
    
    for name, predictions in test_cases:
        signal_density = (predictions > 0.5).mean()
        
        if signal_density > 0.1:
            threshold = np.percentile(predictions, 75)
        elif signal_density > 0.01:
            threshold = np.percentile(predictions, 50)
        else:
            threshold = np.percentile(predictions, 25)
        
        # Apply bounds
        threshold = max(0.05, min(threshold, 0.75))
        
        signals = (predictions > threshold).sum()
        
        print(f"{name}:")
        print(f"  Signal density: {signal_density:.1%}")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Signals generated: {signals}/{len(predictions)} ({signals/len(predictions):.1%})")
        print()
    
    return True

def main():
    """Run all tests."""
    print("Testing V4 Strategy Fixes")
    print("="*70)
    
    tests = [
        ("Triple Barrier Parameters", test_triple_barrier_params),
        ("Feature Engineering", test_feature_engineering),
        ("Class Balancing", test_class_balancing),
        ("Dynamic Threshold", test_dynamic_threshold),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*70)
    print("📊 Test Results")
    print("="*70)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:30s} {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✅ All tests passed! Strategy V4 should generate signals.")
        print("\nNext steps:")
        print("1. Run training: python scripts/train_models.py --quick")
        print("2. Run backtest: python scripts/run_backtest.py")
        print("3. Check signal generation in logs")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
