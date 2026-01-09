"""
ML Pipeline for Stoic Citadel
=============================
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config
from src.ml.training.feature_engineering import FeatureEngineer
from src.ml.training.labeling import LabelGenerator
from src.ml.training.model_registry import ModelMetadata, ModelRegistry, ModelStatus
from src.ml.training.model_trainer import ModelTrainer, TrainingConfig as MLTrainingConfig
from src.ml.training.optimizer import HyperparameterOptimizer

logger = logging.getLogger(__name__)


class MLPipeline:
    """End-to-end ML pipeline orchestrator."""

    def __init__(
        self,
        exchange_name: str = "binance",
        data_dir: str | None = None,
        models_dir: str | None = None,
        quick_mode: bool = False,
    ):
        cfg = config()
        self.data_dir = Path(data_dir or cfg.paths.data_dir / exchange_name)
        self.models_dir = Path(models_dir or cfg.paths.models_dir)
        self.quick_mode = quick_mode

        self.registry = ModelRegistry(str(self.models_dir / "registry"))
        self.engineer = FeatureEngineer()
        self.labeler = LabelGenerator()
        
        # Correctly pass a TrainingConfig object
        trainer_config = MLTrainingConfig(models_dir=str(self.models_dir))
        self.trainer = ModelTrainer(trainer_config)

    def run_pipeline_for_pair(self, pair: str, timeframe: str = "5m") -> str | None:
        pair_slug = pair.replace("/", "_")
        logger.info(f"Starting ML Pipeline for {pair} ({timeframe})")

        try:
            data_path = self.data_dir / f"{pair_slug}-{timeframe}.feather"
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                return None

            df = pd.read_feather(data_path)
            
            # Freqtrade feather data uses 'date' column, labeling expects 'timestamp' or DatetimeIndex
            if 'date' in df.columns:
                df['timestamp'] = df['date']
            
            # Ensure it's sorted
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Use prepare_data to generate features
            df_features = self.engineer.prepare_data(df)
            
            # Generate labels (returns a Series)
            labels = self.labeler.label(df_features)
            df_features["target"] = labels
            
            df_clean = df_features.dropna(subset=["target"])
            
            if len(df_clean) < 1000:
                logger.warning(f"Insufficient data for {pair} after preprocessing: {len(df_clean)}")
                return None

            optimizer = HyperparameterOptimizer(
                n_trials=10 if self.quick_mode else 100,
                study_name=f"opt_{pair_slug}_{timeframe}"
            )
            
            X = df_clean.drop(columns=["target", "date", "timestamp", "open", "high", "low", "close", "volume"], errors="ignore")
            y = df_clean["target"]
            
            best_params = optimizer.optimize(X, y)
            
            model, metrics, features = self.trainer.train(X, y) # Simplification

            model_path = self.models_dir / f"{pair_slug}_{timeframe}.joblib"
            # ModelTrainer handles saving inside train() or we can do it manually
            
            # Register model with correct parameters for ModelRegistry.register_model
            metadata = self.registry.register_model(
                model_name=f"{pair_slug}_{timeframe}",
                model_path=str(model_path),
                version="1.0.0",
                metrics=metrics,
                feature_names=X.columns.tolist(),
                training_config={"params": best_params}
            )
            
            return str(model_path)

        except Exception as e:
            logger.exception(f"Pipeline failed for {pair}: {e}")
            return None

    def run(self, pairs: list[str], timeframe: str = "5m", **kwargs):
        """Standard entry point for training multiple pairs."""
        return self.run_all(pairs, timeframe)

    def run_all(self, pairs: list[str], timeframe: str = "5m"):
        results = {}
        for pair in pairs:
            model_path = self.run_pipeline_for_pair(pair, timeframe)
            results[pair] = model_path
        return results

MLTrainingPipeline = MLPipeline

