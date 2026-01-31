#!/usr/bin/env python
"""
Retrain Production Model Script
===============================

Standalone script to retrain the production model with the latest features.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.training import (
    FeatureConfig,
    FeatureEngineer,
    TripleBarrierConfig,
    TripleBarrierLabeler,
    ModelTrainer,
    TrainingConfig
)
from src.utils.logger import log as logger

def retrain_production_model():
    logger.info("🔄 Starting Production Model Retraining...")

    # 1. Load data
    data_path = Path("user_data/data/binance/BTC_USDT-5m.feather")
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        return

    df = pd.read_feather(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    df = df.sort_index()
    
    logger.info(f"Loaded {len(df)} rows of data.")

    # 2. Feature Engineering (with new features and pruning)
    engineer = FeatureEngineer(FeatureConfig(
        scale_features=True,
        remove_correlated=True,
        correlation_threshold=0.75
    ))
    
    # We use fit_transform on a large enough sample
    training_data = df.tail(50000) 
    X = engineer.fit_transform(training_data)
    
    # 3. Labeling
    labeler = TripleBarrierLabeler(
        TripleBarrierConfig(take_profit=0.01, stop_loss=0.005, max_holding_period=24)
    )
    y_raw = labeler.label(training_data)
    
    # Align X and y FIRST to remove NaNs
    common_index = X.index.intersection(y_raw.dropna().index)
    X = X.loc[common_index]
    y_raw = y_raw.loc[common_index]

    # Convert labels to [0, 1, 2] for XGBoost
    y = (y_raw + 1).astype(int)
    
    logger.info(f"Training on {len(X)} samples with {X.shape[1]} features.")

    # 4. Training
    trainer = ModelTrainer(TrainingConfig(
        model_type="xgboost",
        optimize_hyperparams=False,
        use_calibration=False # Disable calibration to simplify training (Step 2.2 fix)
    ))
    
    model, metrics, features = trainer.train(X, y)
    
    logger.info(f"Retraining Complete. F1 Score: {metrics.get('f1', 0):.4f}")
    
    # Save the engineer/scaler
    engineer.save_scaler("user_data/models/production_scaler.joblib")

if __name__ == "__main__":
    retrain_production_model()
