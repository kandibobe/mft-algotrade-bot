"""
ML Training Pipeline
====================

Complete MLOps pipeline for model training and deployment:
- Data versioning
- Feature engineering
- Model training & optimization
- Experiment tracking
- Model registry & versioning

Author: Stoic Citadel Team
License: MIT
"""

from src.ml.training.feature_engineering import (
    FeatureEngineer,
    FeatureConfig,
)

from src.ml.training.model_trainer import (
    ModelTrainer,
    TrainingConfig,
)

from src.ml.training.experiment_tracker import (
    ExperimentTracker,
    Experiment,
)

from src.ml.training.model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelStatus,
)

__all__ = [
    "FeatureEngineer",
    "FeatureConfig",
    "ModelTrainer",
    "TrainingConfig",
    "ExperimentTracker",
    "Experiment",
    "ModelRegistry",
    "ModelMetadata",
    "ModelStatus",
]
