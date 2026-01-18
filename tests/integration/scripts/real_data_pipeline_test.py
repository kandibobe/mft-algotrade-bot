"""
Full pipeline test on real data with improved parameters.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ml.training.advanced_pipeline import (
    AdvancedFeatureEngineer,
    AdvancedFeatureEngineerConfig,
    DataPreprocessor,
    DataPreprocessorConfig,
    FeatureSelector,
    FeatureSelectorConfig,
    TradingMetrics,
    TradingMetricsConfig,
    TripleBarrierWithPurging,
    TripleBarrierWithPurgingConfig,
    WalkForwardValidator,
    WalkForwardValidatorConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_real_data():
    """Load real BTC 1-minute data."""
    data_path = Path("user_data/data/binance/BTC_USDT-1m.feather")

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    print(f"Loading real data from {data_path}")
    df = pd.read_feather(data_path)

    # Ensure datetime index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df.columns = df.columns.str.lower()

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Sort by index
    df = df.sort_index()

    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    print("Timeframe: 1 minute")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df

def create_improved_configs():
    """Create improved configurations based on real data characteristics."""

    # Data preprocessing config
    preprocessor_config = DataPreprocessorConfig(
        use_log_returns=True,
        remove_outliers=True,
        outlier_sigma_threshold=5.0,
        fill_method="ffill",
        validate_ohlc=True
    )

    # Feature engineering config (optimized for 1-minute data)
    feature_config = AdvancedFeatureEngineerConfig(
        lag_periods=[1, 2, 3, 5, 10, 15, 20, 30, 50],  # More lags for 1-min data
        include_time_features=True,
        include_cyclical_encoding=True,
        include_microstructure=True,
        volume_ma_periods=[5, 10, 20, 50],
        normalize_method="rolling",
        rolling_window=100,  # 100 minutes for rolling stats
        indicator_lag_periods=[1, 2, 3, 5, 10]
    )

    # Labeling config (adjusted for 1-minute timeframe)
    labeler_config = TripleBarrierWithPurgingConfig(
        take_profit=0.002,  # 0.2% for 1-minute (more realistic)
        stop_loss=0.001,    # 0.1% stop loss
        max_holding_period=60,  # 60 minutes max hold
        purge_period=10,    # 10-minute embargo
        fee_adjustment=0.0005,  # 0.05% fee adjustment
        binary_labels=True
    )

    # Feature selection config
    selector_config = FeatureSelectorConfig(
        correlation_threshold=0.90,  # Slightly lower threshold
        use_shap=True,
        top_n_features=30,  # Keep more features for 1-min data
        model_type="random_forest"
    )

    # Validation config
    validator_config = WalkForwardValidatorConfig(
        initial_train_size=0.7,  # 70% initial training
        step_size=0.05,  # 5% step size
        min_train_samples=5000,  # At least 5000 samples for training
        min_test_samples=1000    # At least 1000 samples for testing
    )

    # Metrics config
    metrics_config = TradingMetricsConfig(
        position_size=0.05,  # 5% position size (more conservative)
        commission=0.0005,   # 0.05% commission
        risk_free_rate=0.02  # 2% annual risk-free rate
    )

    return {
        'preprocessor': preprocessor_config,
        'feature_engineer': feature_config,
        'labeler': labeler_config,
        'selector': selector_config,
        'validator': validator_config,
        'metrics': metrics_config
    }

def test_with_improved_configs(df):
    """Test pipeline with improved configurations."""
    print("\n" + "="*80)
    print("TESTING WITH IMPROVED CONFIGURATIONS")
    print("="*80)

    # Get improved configs
    configs = create_improved_configs()

    # Create components with improved configs
    preprocessor = DataPreprocessor(config=configs['preprocessor'])
    feature_engineer = AdvancedFeatureEngineer(config=configs['feature_engineer'])
    labeler = TripleBarrierWithPurging(config=configs['labeler'])
    feature_selector = FeatureSelector(config=configs['selector'])
    validator = WalkForwardValidator(config=configs['validator'])
    metrics_calculator = TradingMetrics(config=configs['metrics'])

    # Stage 1: Data preprocessing
    print("\n1. Data Preprocessing with improved config...")
    df_processed = preprocessor.preprocess(df)
    print(f"   Processed shape: {df_processed.shape}")
    print(f"   Log returns stats: mean={df_processed['log_returns'].mean():.6f}, std={df_processed['log_returns'].std():.6f}")

    # Stage 2: Feature engineering
    print("\n2. Feature Engineering with improved config...")
    df_features = feature_engineer.engineer_features(df_processed)
    print(f"   Total features generated: {df_features.shape[1]}")

    # Stage 3: Labeling
    print("\n3. Labeling with improved config...")
    labels = labeler.create_labels(df_processed)
    label_counts = labels.value_counts()
    print(f"   Buy signals (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(labels)*100:.1f}%)")
    print(f"   Ignore signals (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(labels)*100:.1f}%)")

    # Stage 4: Feature selection
    print("\n4. Feature Selection with improved config...")
    valid_mask = labels.notna()
    X = df_features[valid_mask]
    y = labels[valid_mask].astype(int)

    X_selected = feature_selector.select_features(X, y)
    print(f"   Original features: {X.shape[1]}")
    print(f"   Selected features: {X_selected.shape[1]}")

    # Show top features
    importance_df = feature_selector.get_feature_importance()
    if importance_df is not None:
        print("   Top 10 features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")

    # Stage 5: Walk-Forward Validation with multiple models
    print("\n5. Walk-Forward Validation with multiple models...")

    # Define models to test
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }

    # Test each model
    all_results = {}
    for model_name, model in models.items():
        print(f"\n   Testing {model_name}...")

        # Create folds
        folds = validator.create_folds(X_selected, y)
        print(f"   Created {len(folds)} folds")

        if folds:
            # Custom metric function
            def custom_metric(y_true, y_pred):
                return {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0)
                }

            # Perform cross-validation
            results = {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                'train_size': [], 'test_size': []
            }

            for fold, (train_idx, test_idx) in enumerate(folds, 1):
                # Split data
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
                model.fit(X_train, y_train)

                # Predict
                y_test_pred = model.predict(X_test)

                # Calculate metrics
                metrics = custom_metric(y_test, y_test_pred)
                for key in ['accuracy', 'precision', 'recall', 'f1']:
                    results[key].append(metrics[key])

                results['train_size'].append(len(train_idx))
                results['test_size'].append(len(test_idx))

            # Calculate statistics
            avg_accuracy = np.mean(results['accuracy'])
            avg_precision = np.mean(results['precision'])
            avg_recall = np.mean(results['recall'])
            avg_f1 = np.mean(results['f1'])

            print(f"   Average Accuracy: {avg_accuracy:.2%}")
            print(f"   Average Precision: {avg_precision:.2%}")
            print(f"   Average Recall: {avg_recall:.2%}")
            print(f"   Average F1: {avg_f1:.2%}")

            all_results[model_name] = {
                'results': results,
                'model': model,
                'avg_metrics': {
                    'accuracy': avg_accuracy,
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1': avg_f1
                }
            }

    # Stage 6: Trading metrics with best model
    print("\n6. Trading Metrics with best model...")

    if all_results:
        # Select best model by F1 score
        best_model_name = max(all_results.items(),
                            key=lambda x: x[1]['avg_metrics']['f1'])[0]
        best_model = all_results[best_model_name]['model']
        best_metrics = all_results[best_model_name]['avg_metrics']

        print(f"   Best model: {best_model_name}")
        print(f"   Best F1 score: {best_metrics['f1']:.2%}")

        # Train on full data for final metrics
        best_model.fit(X_selected, y)
        predictions = best_model.predict(X_selected)

        # Calculate trading metrics
        trading_metrics = metrics_calculator.calculate_metrics(
            df_processed.loc[X_selected.index],
            pd.Series(predictions, index=X_selected.index),
            y
        )

        print("\n   Trading Performance:")
        print(f"     Precision: {trading_metrics.get('precision', 0):.2%}")
        print(f"     Profit Factor: {trading_metrics.get('profit_factor', 0):.2f}")
        print(f"     Calmar Ratio: {trading_metrics.get('calmar_ratio', 0):.2f}")
        print(f"     Total Return: {trading_metrics.get('total_return', 0):.2%}")
        print(f"     Max Drawdown: {trading_metrics.get('max_drawdown', 0):.2%}")
        print(f"     Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"     Win Rate: {trading_metrics.get('win_rate', 0):.2%}")
        print(f"     Total Trades: {trading_metrics.get('total_trades', 0)}")

        return {
            'df_processed': df_processed,
            'df_features': df_features,
            'labels': labels,
            'X_selected': X_selected,
            'y': y,
            'all_results': all_results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'trading_metrics': trading_metrics,
            'feature_importance': importance_df
        }

    return None

def run_complete_pipeline_analysis():
    """Run complete analysis pipeline."""
    print("="*80)
    print("COMPLETE REAL DATA PIPELINE ANALYSIS")
    print("="*80)

    try:
        # Load real data
        df = load_real_data()

        # Test with improved configs
        results = test_with_improved_configs(df)

        if results:
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE - KEY FINDINGS")
            print("="*80)

            # Summary of findings
            trading_metrics = results['trading_metrics']
            feature_importance = results['feature_importance']

            print("\n1. DATA CHARACTERISTICS:")
            print(f"   Samples: {len(results['df_processed']):,}")
            print(f"   Features generated: {results['df_features'].shape[1]}")
            print(f"   Features selected: {results['X_selected'].shape[1]}")
            print(f"   Buy signals: {(results['y'] == 1).sum()} ({(results['y'] == 1).sum()/len(results['y'])*100:.1f}%)")

            print("\n2. MODEL PERFORMANCE:")
            for model_name, model_results in results['all_results'].items():
                metrics = model_results['avg_metrics']
                print(f"   {model_name}:")
                print(f"     Accuracy: {metrics['accuracy']:.2%}")
                print(f"     Precision: {metrics['precision']:.2%}")
                print(f"     F1 Score: {metrics['f1']:.2%}")

            print("\n3. TRADING PERFORMANCE:")
            print(f"   Precision (entry accuracy): {trading_metrics.get('precision', 0):.2%}")
            print(f"   Profit Factor: {trading_metrics.get('profit_factor', 0):.2f}")
            print(f"   Calmar Ratio: {trading_metrics.get('calmar_ratio', 0):.2f}")
            print(f"   Total Return: {trading_metrics.get('total_return', 0):.2%}")
            print(f"   Max Drawdown: {trading_metrics.get('max_drawdown', 0):.2%}")

            print("\n4. KEY FEATURES (Top 5):")
            if feature_importance is not None:
                for idx, row in feature_importance.head(5).iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")

            print("\n5. RECOMMENDATIONS:")

            # Generate recommendations based on results
            if trading_metrics.get('profit_factor', 0) > 1.5:
                print("   ✓ Profit Factor is GOOD (>1.5)")
            elif trading_metrics.get('profit_factor', 0) > 1.1:
                print("   ⚠ Profit Factor is MARGINAL (1.1-1.5)")
            else:
                print("   ✗ Profit Factor is POOR (<1.1)")

            if trading_metrics.get('precision', 0) > 0.55:
                print("   ✓ Precision is GOOD (>55%)")
            elif trading_metrics.get('precision', 0) > 0.45:
                print("   ⚠ Precision is MARGINAL (45-55%)")
            else:
                print("   ✗ Precision is POOR (<45%)")

            if trading_metrics.get('max_drawdown', 0) > -0.2:
                print("   ✓ Max Drawdown is ACCEPTABLE (<20%)")
            elif trading_metrics.get('max_drawdown', 0) > -0.3:
                print("   ⚠ Max Drawdown is HIGH (20-30%)")
            else:
                print("   ✗ Max Drawdown is VERY HIGH (>30%)")

            print("\n6. NEXT STEPS:")
            print("   - Consider hyperparameter tuning for better performance")
            print("   - Test on different timeframes (5m, 15m, 1h)")
            print("   - Add more sophisticated features (order book, sentiment)")
            print("   - Implement risk management rules")
            print("   - Backtest with transaction costs and slippage")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_pipeline_analysis()
    sys.exit(0 if success else 1)
