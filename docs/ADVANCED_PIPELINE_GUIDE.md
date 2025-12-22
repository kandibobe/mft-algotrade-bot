# Advanced Trading Pipeline Guide

## Overview

The Advanced Trading Pipeline implements a complete ML workflow for trading according to best practices described in the requirements. It consists of 6 stages that transform raw OHLCV data into actionable trading signals with proper validation and metrics.

## Pipeline Stages

### Stage 1: Data Preprocessing (`DataPreprocessor`)

**Main Rule**: Never feed raw prices to the model.

**Features**:
- **Stationarity**: Uses log returns: `np.log(df['close'] / df['close'].shift(1))`
- **Outlier Removal**: Removes spikes where returns exceed 5-10 sigma
- **Data Validation**: Checks OHLC relationships (high ≥ low, close between high/low)
- **Missing Data Handling**: Forward/backward fill or interpolation

**Usage**:
```python
from ml.training.advanced_pipeline import DataPreprocessor

preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess(df)
```

### Stage 2: Feature Engineering (`AdvancedFeatureEngineer`)

**Smart Features, Not Just RSI/MACD**:

1. **Lag Features**: Context for the model (RSI_lag_1, RSI_lag_2, etc.)
2. **Time Features**:
   - `hour_of_day`: Crypto behaves differently at 3 AM vs 4 PM (US market open)
   - `day_of_week`: Weekends are often flat
   - Cyclical encoding for periodic features
3. **Microstructure Features**:
   - `Volume / MovingAverage(Volume)`: Relative volume
   - Volume acceleration, dollar volume, volume profile
4. **Normalization**: Rolling window scaling (no lookahead bias)

**Usage**:
```python
from ml.training.advanced_pipeline import AdvancedFeatureEngineer

feature_engineer = AdvancedFeatureEngineer()
df_features = feature_engineer.engineer_features(df_processed)
```

### Stage 3: Labeling (`TripleBarrierWithPurging`)

**Golden Standard: Triple Barrier Method**:

- **Class 1 (Long)**: Price hits Take Profit (+1%) before Stop Loss (-0.5%) or Time Limit
- **Class 0 (Hold/Ignore)**: Price is flat or hits Stop Loss

**Important Nuance (Purging)**: Prevents overlapping trades. If a trade starts at row 100 and ends at 105, data from rows 101-104 already "know" the future.

**Usage**:
```python
from ml.training.advanced_pipeline import TripleBarrierWithPurging

labeler = TripleBarrierWithPurging()
labels = labeler.create_labels(df_processed)
```

### Stage 4: Feature Selection (`FeatureSelector`)

**Avoid Overfitting on Noise**:

1. **Correlation Matrix**: Removes features with correlation > 0.95 (e.g., RSI and Stoch)
2. **SHAP Values / Feature Importance**:
   - Train a trial model
   - Look at top-20 important features
   - Keep 15-25 strongest features

**Usage**:
```python
from ml.training.advanced_pipeline import FeatureSelector

feature_selector = FeatureSelector()
X_selected = feature_selector.select_features(X, y)
```

### Stage 5: Validation (`WalkForwardValidator`)

**Prohibited**: Ordinary `train_test_split` (random shuffling)

**Required**: Walk-Forward Validation (Sliding Window):
- Train: January-March → Test: April
- Train: February-April → Test: May

**Usage**:
```python
from ml.training.advanced_pipeline import WalkForwardValidator

validator = WalkForwardValidator()
folds = validator.create_folds(X, y)
cv_results = validator.cross_validate(X, y, model, accuracy_score)
```

### Stage 6: Success Metrics (`TradingMetrics`)

**Forget About Accuracy** - it lies in trading!

**What to Look At**:
- **Precision (Entry Accuracy)**: Out of 100 "BUY" signals, how many actually closed in profit? Need > 55-60%
- **Profit Factor**: Total profit / Total loss. Should be > 1.1 (better > 1.5)
- **Calmar Ratio**: Return / Maximum drawdown

**Usage**:
```python
from ml.training.advanced_pipeline import TradingMetrics

metrics_calculator = TradingMetrics()
metrics = metrics_calculator.calculate_metrics(df, predictions, labels)
```

## Complete Pipeline Usage

### Basic Usage

```python
from ml.training.advanced_pipeline import AdvancedTradingPipeline

# Initialize pipeline
pipeline = AdvancedTradingPipeline()

# Run on your data
results = pipeline.run(df)

# Get summary
print(pipeline.get_summary())

# Access individual results
df_processed = results['data_processed']
features = results['features']
labels = results['labels']
X_selected = results['X_selected']
trading_metrics = results['trading_metrics']
feature_importance = results['feature_importance']
model = results['model']
```

### Configuration

Each stage can be configured:

```python
from ml.training.advanced_pipeline import (
    DataPreprocessorConfig,
    AdvancedFeatureEngineerConfig,
    TripleBarrierWithPurgingConfig,
    FeatureSelectorConfig,
    WalkForwardValidatorConfig,
    TradingMetricsConfig
)

# Custom configurations
preprocessor_config = DataPreprocessorConfig(
    use_log_returns=True,
    outlier_sigma_threshold=5.0,
    fill_method="ffill"
)

feature_config = AdvancedFeatureEngineerConfig(
    lag_periods=[1, 2, 3, 5, 10, 20],
    include_time_features=True,
    normalize_method="rolling"
)

labeler_config = TripleBarrierWithPurgingConfig(
    take_profit=0.01,  # 1%
    stop_loss=0.005,   # 0.5%
    max_holding_period=48,
    purge_period=5
)

# Use with pipeline components
from ml.training.advanced_pipeline import DataPreprocessor
preprocessor = DataPreprocessor(config=preprocessor_config)
```

## Example with Real Data

```python
import pandas as pd
from ml.training.advanced_pipeline import AdvancedTradingPipeline

# Load your data (must have OHLCV columns)
df = pd.read_parquet("your_data.parquet")

# Ensure proper datetime index
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

# Run pipeline
pipeline = AdvancedTradingPipeline()
results = pipeline.run(df)

# Analyze results
print("="*60)
print("PIPELINE RESULTS")
print("="*60)
print(f"Data samples: {len(results['data_processed'])}")
print(f"Features generated: {results['features'].shape[1]}")
print(f"Features selected: {results['X_selected'].shape[1]}")
print(f"Buy signals: {(results['y'] == 1).sum()}")

metrics = results['trading_metrics']
print(f"\nTrading Performance:")
print(f"  Precision: {metrics.get('precision', 0):.2%}")
print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

# Feature importance analysis
importance_df = results['feature_importance']
if importance_df is not None:
    print(f"\nTop 10 Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
```

## Best Practices

### 1. Data Quality
- Always use log returns for stationarity
- Remove outliers (5-10 sigma threshold)
- Validate OHLC relationships
- Handle missing data appropriately

### 2. Feature Engineering
- Include lag features for context
- Add time features (hour, day, sessions)
- Use microstructure features (relative volume)
- Consider rolling normalization for no lookahead bias

### 3. Labeling
- Use Triple Barrier Method with purging
- Adjust barriers for fees
- Prevent overlapping trades with embargo

### 4. Feature Selection
- Remove highly correlated features (>0.95)
- Use SHAP values for importance
- Keep 15-25 strongest features

### 5. Validation
- Always use Walk-Forward Validation
- Never use random shuffling for time series
- Ensure sufficient samples per fold

### 6. Metrics
- Focus on Precision, not Accuracy
- Monitor Profit Factor (>1.1 minimum)
- Track Calmar Ratio for risk-adjusted returns
- Consider maximum drawdown

## Troubleshooting

### Common Issues

1. **No Buy Signals**:
   - Adjust Triple Barrier parameters (take_profit, stop_loss)
   - Check if price movements are sufficient
   - Ensure enough data samples

2. **Poor Feature Importance**:
   - Check for data leakage
   - Verify feature engineering logic
   - Consider different normalization methods

3. **Walk-Forward Validation Errors**:
   - Ensure sufficient data (min 1000 samples recommended)
   - Adjust fold sizes in configuration
   - Check time index sorting

4. **Unrealistic Metrics**:
   - Verify trade simulation logic
   - Check commission and fee adjustments
   - Validate risk metric calculations

### Performance Tips

1. **For Large Datasets**:
   - Use smaller lag periods
   - Reduce number of features
   - Consider batch processing

2. **For Real-time Usage**:
   - Precompute rolling statistics
   - Cache feature engineering results
   - Use incremental learning

## Integration with Existing Codebase

The pipeline is designed to integrate with the existing `mft-algotrade-bot`:

1. **Data Sources**: Works with existing data loaders in `src/data/`
2. **Models**: Compatible with existing ML models in `src/ml/`
3. **Monitoring**: Can be integrated with monitoring in `src/monitoring/`
4. **Strategies**: Results can feed into trading strategies in `user_data/strategies/`

## Next Steps

1. **Production Deployment**:
   - Add model persistence
   - Implement online learning
   - Add monitoring and alerts

2. **Advanced Features**:
   - Ensemble methods
   - Meta-learning
   - Reinforcement learning integration

3. **Optimization**:
   - Hyperparameter tuning
   - Feature store integration
   - Distributed training

## Support

For issues or questions:
1. Check the existing tests in `test_advanced_pipeline.py`
2. Review the demo in `demo_advanced_pipeline.py`
3. Examine the source code in `src/ml/training/advanced_pipeline.py`
4. Consult the project documentation in `docs/`
