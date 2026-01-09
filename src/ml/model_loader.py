"""
ML Model Loader
===============

Helper to load production models for strategy usage.
Bridges the gap between ML Pipeline and Freqtrade Strategy.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

from src.config import config
from src.ml.training.feature_engineering import FeatureEngineer
from src.ml.training.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads production ML models and associated artifacts."""

    def __init__(self, registry_dir: str | None = None):
        self.registry = ModelRegistry(registry_dir or str(config().paths.models_dir / "registry"))
        self.models_cache = {}

    def load_model_for_pair(
        self, pair: str
    ) -> tuple[Any | None, FeatureEngineer | None, list[str]]:
        """
        Load production model, FeatureEngineer, and feature names for a pair.

        Args:
            pair: Trading pair (e.g. "BTC/USDT")

        Returns:
            (model, feature_engineer, feature_names)
        """
        model_name = pair.replace("/", "_")

        # Check cache
        if model_name in self.models_cache:
            return self.models_cache[model_name]

        # Get production model metadata
        metadata = self.registry.get_production_model(model_name)
        if not metadata:
            logger.warning(f"No production model found for {pair}")
            return None, None, []

        try:
            # Load Model
            with open(metadata.model_path, "rb") as f:
                model = pickle.load(f)

            # Load Feature Engineer
            # Metadata path: user_data/models/BTC_USDT_20230101_120000.pkl
            # Scaler path: user_data/models/BTC_USDT_20230101_120000_scaler.joblib
            scaler_path = Path(metadata.model_path).with_suffix("")  # Strip .pkl
            scaler_path = Path(f"{scaler_path}_scaler.joblib")

            engineer = None
            if scaler_path.exists():
                engineer = FeatureEngineer()
                engineer.load_scaler(str(scaler_path))
            else:
                logger.warning(f"Scaler not found at {scaler_path}")

            # Cache
            result = (model, engineer, metadata.feature_names)
            self.models_cache[model_name] = result

            logger.info(f"Loaded production model for {pair} (v{metadata.version})")
            return result

        except Exception as e:
            logger.error(f"Failed to load model for {pair}: {e}")
            return None, None, []


_loader_instance = None


def get_production_model(pair: str) -> tuple[Any | None, FeatureEngineer | None, list[str]]:
    """Global helper to get production model."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ModelLoader()
    return _loader_instance.load_model_for_pair(pair)
