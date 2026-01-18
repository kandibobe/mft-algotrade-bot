"""
Retrain Production Model
========================

This script retrains a production model using a pre-computed set of hyperparameters,
typically generated from a nightly hyperparameter optimization run.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import get_ohlcv
from src.ml.training.feature_engineering import FeatureEngineer
from src.ml.training.labeling import LabelGenerator
from src.ml.training.model_trainer import ModelTrainer, TrainingConfig
from src.ml.training.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

def retrain_model(pair: str, params_path: str, timeframe: str = "5m", days: int = 1095):
    """
    Retrains and registers a model for a given pair using specified hyperparameters.
    """
    logger.info(f"Starting retraining for {pair} using params from {params_path}")

    # 1. Load Hyperparameters
    with open(params_path) as f:
        best_params = json.load(f)

    # 2. Load and Prepare Data
    logger.info(f"Loading {days} days of data for {pair} {timeframe}")
    df = get_ohlcv(pair=pair, timeframe=timeframe)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    
    close_prices = df["close"]

    logger.info("Generating features...")
    feature_engineer = FeatureEngineer()
    X = feature_engineer.fit_transform(df.copy())
    X = X.select_dtypes(include=[np.number])

    logger.info("Generating labels...")
    labeler = LabelGenerator()
    y = labeler.label(df)
    
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    y_binary = (y == 1).astype(int)

    logger.info(f"Final dataset: {len(X)} samples")

    # 3. Train Model
    trainer_config = TrainingConfig(
        model_type="xgboost", 
        save_model=True, 
        models_dir="user_data/models"
    )
    trainer = ModelTrainer(trainer_config)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y_binary, test_size=0.2, shuffle=False)
    
    model, metrics, feature_names = trainer.train(X_train, y_train, X_val, y_val, hyperparams=best_params)

    # 4. Register Model as Production
    registry = ModelRegistry()
    model_path = Path(trainer_config.models_dir) / f"{pair.replace('/', '_')}_{timeframe}.joblib" # Simplified path
    
    # Manually save the model to a predictable path
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    registry.register_model(
        model_name=pair.replace("/", "_"),
        model_path=str(model_path),
        version="nightly",
        metrics=metrics,
        feature_names=feature_names,
        training_config={"params": best_params},
        status="production" # Promote to production
    )

    logger.info(f"Successfully retrained and promoted model for {pair} to production.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain and deploy a production model.")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair to retrain (e.g., BTC/USDT)")
    parser.add_argument("--params-file", type=str, default="user_data/nightly_hyperopt/best_params_nightly.json", help="Path to the JSON file with hyperparameters.")
    
    args = parser.parse_args()
    retrain_model(args.pair, args.params_file)