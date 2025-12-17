# ML Training Pipeline

Complete MLOps pipeline for training, tracking, and deploying machine learning models.

## Features

### 1. Feature Engineering

Transform raw OHLCV data into ML-ready features:

- **50+ Technical Indicators** - Price, volume, momentum, volatility, trend
- **Time-based Features** - Cyclical encoding for hours, days, months
- **Feature Scaling** - StandardScaler, MinMaxScaler, RobustScaler
- **Correlation Removal** - Automatic detection and removal of redundant features
- **Configurable Pipeline** - Customize which feature groups to include

### 2. Model Training

Train and optimize ML models with hyperparameter tuning:

- **Supported Models** - Random Forest, XGBoost, LightGBM
- **Hyperparameter Optimization** - Optuna-based automated tuning
- **Cross-validation** - Time-series aware splitting
- **Feature Selection** - Automatic feature importance ranking
- **Model Persistence** - Save/load trained models

### 3. Experiment Tracking

Track experiments with W&B or MLflow:

- **Hyperparameter Logging** - Track all model configurations
- **Metrics Tracking** - Training/validation/backtest metrics
- **Artifact Management** - Save models, plots, data
- **Experiment Comparison** - Compare multiple runs
- **Backtest Integration** - Link ML models to trading performance

### 4. Model Registry

Version control and deployment management:

- **Version Management** - Track multiple model versions
- **Validation Workflow** - Validate before production
- **Production Promotion** - Deploy models with confidence
- **Rollback Mechanism** - Revert to previous versions
- **Model Archiving** - Clean up old models

## Installation

Install ML dependencies:

```bash
pip install optuna wandb xgboost lightgbm scikit-learn
```

For MLflow (alternative to W&B):

```bash
pip install mlflow
```

## Quick Start

### Basic Feature Engineering

```python
from src.ml.training import FeatureEngineer
import pandas as pd

# Load OHLCV data
df = pd.read_csv("ohlcv_data.csv", parse_dates=['date'], index_col='date')

# Create feature engineer
engineer = FeatureEngineer()

# Transform data
features_df = engineer.transform(df)

print(f"Generated {len(engineer.get_feature_names())} features")
```

### Train a Model

```python
from src.ml.training import ModelTrainer, ModelConfig

# Prepare data
X_train = features_df[engineer.get_feature_names()]
y_train = (features_df['close'].shift(-1) > features_df['close']).astype(int)

# Configure trainer
config = ModelConfig(
    model_type="random_forest",
    optimize_hyperparams=True,
    n_trials=50,
)

# Train model
trainer = ModelTrainer(config)
model, metrics = trainer.train(X_train, y_train)

print(f"F1 Score: {metrics['f1']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Track Experiments

```python
from src.ml.training import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(project="stoic-citadel-ml")

# Start run
tracker.start_run(
    name="rf_trend_classifier_v1",
    config=config.to_dict(),
    tags=["random_forest", "trend_prediction"]
)

# Train model
model, metrics = trainer.train(X_train, y_train)

# Log metrics
tracker.log_metrics(metrics)

# Log feature importance
feature_importance = dict(zip(
    X_train.columns,
    model.feature_importances_
))
tracker.log_feature_importance(feature_importance)

# Save model and finish
trainer.save_model(model, "models/rf_v1.pkl")
tracker.log_model("models/rf_v1.pkl")
tracker.finish()
```

### Register and Deploy Model

```python
from src.ml.training import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Register model
metadata = registry.register_model(
    model_name="trend_classifier",
    model_path="models/rf_v1.pkl",
    version="v1.0",
    metrics=metrics,
    feature_names=list(X_train.columns),
    tags=["production_candidate"]
)

# Validate model
is_valid = registry.validate_model(
    model_name="trend_classifier",
    version="v1.0",
    min_metrics={"f1": 0.60, "accuracy": 0.60},
    min_backtest_sharpe=1.0,
    min_backtest_trades=10
)

# Promote to production if valid
if is_valid:
    registry.promote_to_production(
        model_name="trend_classifier",
        version="v1.0",
        notes="Initial production deployment"
    )
    print("Model deployed to production!")
```

## Complete Workflow

### End-to-End ML Pipeline

```python
from src.ml.training import (
    FeatureEngineer,
    FeatureConfig,
    ModelTrainer,
    ModelConfig,
    ExperimentTracker,
    ModelRegistry
)
import pandas as pd

# 1. Load data
df = pd.read_csv("btc_1h.csv", parse_dates=['date'], index_col='date')

# 2. Feature engineering
feature_config = FeatureConfig(
    include_price_features=True,
    include_volume_features=True,
    include_momentum_features=True,
    include_volatility_features=True,
    include_trend_features=True,
    scale_features=True,
    remove_correlated=True,
)

engineer = FeatureEngineer(feature_config)
features_df = engineer.transform(df)

# 3. Prepare train/validation split (time-series aware)
split_idx = int(len(features_df) * 0.8)
train_df = features_df.iloc[:split_idx]
val_df = features_df.iloc[split_idx:]

feature_cols = engineer.get_feature_names()
X_train = train_df[feature_cols]
y_train = (train_df['close'].shift(-1) > train_df['close']).astype(int)
X_val = val_df[feature_cols]
y_val = (val_df['close'].shift(-1) > val_df['close']).astype(int)

# Remove NaN rows
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_val = X_val.dropna()
y_val = y_val.loc[X_val.index]

# 4. Start experiment tracking
tracker = ExperimentTracker(project="stoic-citadel-ml", backend="wandb")
tracker.start_run(
    name=f"xgboost_trend_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
    description="XGBoost classifier with hyperparameter optimization",
    config={"features": len(feature_cols), "train_size": len(X_train)},
    tags=["xgboost", "trend", "optimized"]
)

# 5. Train model with hyperparameter optimization
model_config = ModelConfig(
    model_type="xgboost",
    optimize_hyperparams=True,
    n_trials=100,
    cv_folds=5,
)

trainer = ModelTrainer(model_config)
model, metrics = trainer.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val
)

# 6. Log results
tracker.log_metrics(metrics)
tracker.log_feature_importance(trainer.get_feature_importance(model, feature_cols))

# 7. Save model
model_path = f"user_data/models/xgboost_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pkl"
trainer.save_model(model, model_path)
tracker.log_model(model_path)

# 8. Run backtest (integrate with your backtesting system)
# backtest_results = run_backtest(model, val_df)
# tracker.log_backtest_results(backtest_results)

tracker.finish(success=True)

# 9. Register model
registry = ModelRegistry()
metadata = registry.register_model(
    model_name="trend_classifier",
    model_path=model_path,
    metrics=metrics,
    # backtest_results=backtest_results,
    feature_names=feature_cols,
    tags=["xgboost", "trend", "v2"]
)

# 10. Validate
is_valid = registry.validate_model(
    model_name="trend_classifier",
    version=metadata.version,
    min_metrics={"f1": 0.60, "accuracy": 0.60},
    # min_backtest_sharpe=1.5,
    # min_backtest_trades=50,
)

# 11. Deploy to production
if is_valid:
    registry.promote_to_production(
        model_name="trend_classifier",
        version=metadata.version,
        notes="XGBoost model with optimized hyperparameters"
    )
    print(f"Model v{metadata.version} deployed to production!")
else:
    print("Model validation failed - review metrics before deployment")
```

## Integration with Freqtrade

### Using ML Model in Strategy

```python
from freqtrade.strategy import IStrategy
from src.ml.training import ModelRegistry
from src.ml.inference_service import ModelInferenceService
import pickle

class MLStrategy(IStrategy):
    def __init__(self, config: dict):
        super().__init__(config)

        # Load production model
        registry = ModelRegistry()
        prod_model = registry.get_production_model("trend_classifier")

        if prod_model:
            with open(prod_model.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.feature_names = prod_model.feature_names
        else:
            raise ValueError("No production model found!")

    def populate_indicators(self, dataframe, metadata):
        # Generate features using same FeatureEngineer
        from src.ml.training import FeatureEngineer

        engineer = FeatureEngineer()
        features_df = engineer.transform(dataframe)

        return features_df

    def populate_entry_trend(self, dataframe, metadata):
        # Predict using ML model
        X = dataframe[self.feature_names].values
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        # Entry signal: model predicts uptrend with >70% confidence
        dataframe['enter_long'] = (
            (predictions == 1) &
            (probabilities > 0.70)
        )

        return dataframe
```

### Automated Retraining Pipeline

```python
# retrain_model.py
from src.ml.training import (
    FeatureEngineer,
    ModelTrainer,
    ModelConfig,
    ModelRegistry,
    ExperimentTracker
)
import pandas as pd
from datetime import datetime

def retrain_trend_model():
    """Retrain model with latest data."""

    # Load latest data
    df = load_latest_ohlcv_data()  # Your data loading function

    # Feature engineering
    engineer = FeatureEngineer()
    features_df = engineer.transform(df)

    # Prepare data
    feature_cols = engineer.get_feature_names()
    X = features_df[feature_cols].dropna()
    y = (features_df['close'].shift(-1) > features_df['close']).astype(int).loc[X.index]

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Track experiment
    tracker = ExperimentTracker(project="stoic-citadel-ml")
    run_name = f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M')}"
    tracker.start_run(run_name, tags=["automated", "retraining"])

    # Train
    config = ModelConfig(model_type="xgboost", optimize_hyperparams=True)
    trainer = ModelTrainer(config)
    model, metrics = trainer.train(X_train, y_train, X_val, y_val)

    # Log
    tracker.log_metrics(metrics)

    # Save
    model_path = f"user_data/models/auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    trainer.save_model(model, model_path)
    tracker.log_model(model_path)
    tracker.finish()

    # Register
    registry = ModelRegistry()
    metadata = registry.register_model(
        model_name="trend_classifier",
        model_path=model_path,
        metrics=metrics,
        feature_names=feature_cols,
        tags=["automated_retrain"]
    )

    # Validate
    is_valid = registry.validate_model(
        model_name="trend_classifier",
        version=metadata.version,
        min_metrics={"f1": 0.60, "accuracy": 0.60}
    )

    if is_valid:
        print(f"New model v{metadata.version} validated - promoting to production")
        registry.promote_to_production(
            model_name="trend_classifier",
            version=metadata.version,
            notes=f"Automated retraining on {datetime.now().isoformat()}"
        )
    else:
        print("New model failed validation - keeping current production model")

    return metadata

# Run daily via cron or scheduler
if __name__ == "__main__":
    retrain_trend_model()
```

## Configuration

### Feature Engineering Configuration

```python
from src.ml.training import FeatureConfig

config = FeatureConfig(
    # Feature groups
    include_price_features=True,
    include_volume_features=True,
    include_momentum_features=True,
    include_volatility_features=True,
    include_trend_features=True,
    include_time_features=True,

    # Lookback periods
    short_period=14,
    medium_period=50,
    long_period=200,

    # Scaling
    scale_features=True,
    scaling_method="standard",  # standard, minmax, robust

    # Feature selection
    remove_correlated=True,
    correlation_threshold=0.95,
)
```

### Model Training Configuration

```python
from src.ml.training import ModelConfig

config = ModelConfig(
    # Model type
    model_type="xgboost",  # random_forest, xgboost, lightgbm

    # Hyperparameter optimization
    optimize_hyperparams=True,
    n_trials=100,
    optimization_metric="f1",  # f1, accuracy, precision, recall

    # Cross-validation
    cv_folds=5,
    cv_method="timeseries",  # timeseries, kfold

    # Feature selection
    feature_selection=True,
    min_feature_importance=0.001,
    max_features=50,

    # Early stopping
    early_stopping_rounds=50,

    # Random state
    random_state=42,
)
```

## Feature Engineering Details

### Generated Features

**Price Features:**
- Returns (simple & log)
- Price position in high-low range
- Gap from previous close
- Intraday return (close vs open)

**Volume Features:**
- Volume change rate
- Volume moving average
- Volume ratio
- VWAP and VWAP difference

**Momentum Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- MACD Signal & Histogram
- Stochastic Oscillator (K & D)

**Volatility Indicators:**
- ATR (Average True Range)
- ATR percentage
- Bollinger Bands (upper, lower, width, position)
- Historical volatility

**Trend Indicators:**
- SMA (Short, Medium, Long)
- EMA (Short, Medium)
- Price vs MA ratios
- MA crossovers
- ADX (Average Directional Index)

**Time Features:**
- Hour, day of week, month
- Cyclical encoding (sin/cos for hour and day)

## Model Registry Details

### Model Lifecycle

```
STAGED → [validation] → PRODUCTION → [rollback/archive] → ARCHIVED
   ↓
FAILED (if validation fails)
```

### Model Metadata

Each registered model includes:

- **Identification**: Name, version, type
- **Status**: STAGED, PRODUCTION, ARCHIVED, FAILED
- **Paths**: Model file, metadata file
- **Metrics**: Training/validation metrics
- **Backtest**: Trading performance results
- **Training Info**: Timestamp, config, trained_by
- **Features**: Feature count, feature names
- **Validation**: Pass/fail status, notes
- **Deployment**: Deployment timestamp, notes
- **Tags**: Custom labels

### Version Management

```python
# Get all versions
versions = registry.get_all_versions("trend_classifier")
for v in versions:
    print(f"v{v.version}: {v.status.value} - F1: {v.metrics.get('f1', 0):.4f}")

# Get production model
prod_model = registry.get_production_model("trend_classifier")
print(f"Production: v{prod_model.version}")

# Get model history
history = registry.get_model_history("trend_classifier")
print(f"Total versions: {history['total_versions']}")
print(f"Production: {history['production_version']}")

# Rollback to previous version
registry.rollback_to_version("trend_classifier", "v2.0")

# Archive old version
registry.archive_model("trend_classifier", "v1.0")

# Delete failed model
registry.delete_model("trend_classifier", "v1.5", delete_files=True)
```

## Experiment Tracking Details

### W&B Integration

```python
# Login to W&B
import wandb
wandb.login()

# Track experiment
tracker = ExperimentTracker(
    project="stoic-citadel-ml",
    entity="your_wandb_username",  # Optional
    backend="wandb"
)

# Advanced logging
tracker.log_confusion_matrix(y_true, y_pred, class_names=["down", "up"])
tracker.log_roc_curve(y_true, y_probas, class_names=["down", "up"])
```

### MLflow Integration

```python
# Use MLflow instead
tracker = ExperimentTracker(
    project="stoic-citadel-ml",
    backend="mlflow"
)

# View experiments
# mlflow ui --port 5000
# Visit http://localhost:5000
```

### Comparing Experiments

```python
# Compare multiple runs
comparison = tracker.compare_experiments([
    "xgboost_v1",
    "random_forest_v1",
    "lightgbm_v1"
])

print(comparison)
# {
#   'experiments': ['xgboost_v1', 'random_forest_v1', 'lightgbm_v1'],
#   'metrics': {
#       'f1': {'xgboost_v1': 0.72, 'random_forest_v1': 0.68, ...},
#       'accuracy': {'xgboost_v1': 0.71, ...}
#   }
# }
```

## Best Practices

### 1. Feature Engineering

```python
# DO: Use time-series aware splits
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]  # Earlier data
test = df.iloc[split_idx:]   # Later data

# DON'T: Use random splits (causes lookahead bias)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2)  # WRONG for time-series!
```

### 2. Model Validation

```python
# DO: Validate on holdout period + backtest
registry.validate_model(
    "trend_classifier", "v1.0",
    min_metrics={"f1": 0.60},
    min_backtest_sharpe=1.0,
    min_backtest_trades=10
)

# DON'T: Deploy without validation
registry.promote_to_production("trend_classifier", "v1.0")  # Risky!
```

### 3. Version Control

```python
# DO: Use semantic versioning
registry.register_model(
    model_name="trend_classifier",
    version="v2.1",  # Major.Minor
    ...
)

# DO: Add descriptive tags
tags=["xgboost", "1h_timeframe", "btc", "optimized"]
```

### 4. Experiment Tracking

```python
# DO: Track everything
tracker.start_run("experiment_name")
tracker.log_config(config.to_dict())
tracker.log_metrics(metrics)
tracker.log_feature_importance(feature_importance)
tracker.log_model(model_path)
tracker.finish()

# DON'T: Skip tracking (you'll lose context)
```

### 5. Production Deployment

```python
# DO: Test before promoting
prod_model = registry.get_production_model("trend_classifier")
print(f"Current production: v{prod_model.version}")

# Validate new model
if registry.validate_model("trend_classifier", "v3.0"):
    # Promote (automatically archives old production model)
    registry.promote_to_production("trend_classifier", "v3.0")

    # Test in paper trading first
    # If issues detected:
    registry.rollback_to_version("trend_classifier", prod_model.version)
```

## Monitoring

### Track Model Performance

```python
# Get production model info
prod_model = registry.get_production_model("trend_classifier")

print(f"Model: {prod_model.name} v{prod_model.version}")
print(f"Deployed: {prod_model.deployed_at}")
print(f"Training F1: {prod_model.metrics['f1']:.4f}")
print(f"Features: {prod_model.feature_count}")

# Monitor live performance vs training metrics
# If significant degradation, consider retraining
```

## Troubleshooting

### Features Not Generating

**Problem:** `FeatureEngineer.transform()` returns empty DataFrame

**Solutions:**
1. Check DataFrame has required columns: `['open', 'high', 'low', 'close', 'volume']`
2. Ensure DataFrame has DatetimeIndex for time features
3. Verify sufficient data (need >200 rows for long_period features)

### Model Training Fails

**Problem:** Training crashes or produces poor results

**Solutions:**
1. Check for NaN values: `X_train.isna().sum()`
2. Verify target variable is binary (0/1) for classification
3. Ensure sufficient training samples (>1000 recommended)
4. Try different scaling method in FeatureConfig

### W&B Login Issues

**Problem:** `wandb.errors.UsageError: api_key not configured`

**Solution:**
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

Or use MLflow instead:
```python
tracker = ExperimentTracker(backend="mlflow")
```

### Model Registry Not Finding Models

**Problem:** `registry.get_production_model()` returns None

**Solutions:**
1. Check model was registered: `registry.get_all_versions("trend_classifier")`
2. Verify model was promoted: `metadata.status == ModelStatus.PRODUCTION`
3. Check registry directory exists: `user_data/models/registry/`

## Performance Tips

### 1. Feature Engineering Speed

```python
# For large datasets, disable correlated feature removal
config = FeatureConfig(
    remove_correlated=False  # Faster, but more features
)

# Or increase correlation threshold
config = FeatureConfig(
    correlation_threshold=0.98  # Remove fewer features
)
```

### 2. Hyperparameter Optimization

```python
# Balance speed vs accuracy
config = ModelConfig(
    n_trials=20,  # Fewer trials = faster (but less optimal)
    cv_folds=3,   # Fewer folds = faster validation
)

# For quick experiments, disable optimization
config = ModelConfig(
    optimize_hyperparams=False  # Use default params
)
```

### 3. Model Selection

```python
# Random Forest: Fastest training
config = ModelConfig(model_type="random_forest")

# XGBoost: Good balance of speed/performance
config = ModelConfig(model_type="xgboost")

# LightGBM: Fast for large datasets
config = ModelConfig(model_type="lightgbm")
```

## References

- Feature Engineering: `src/ml/training/feature_engineering.py`
- Model Trainer: `src/ml/training/model_trainer.py`
- Experiment Tracker: `src/ml/training/experiment_tracker.py`
- Model Registry: `src/ml/training/model_registry.py`
- Documentation: `docs/ML_TRAINING_PIPELINE.md`
