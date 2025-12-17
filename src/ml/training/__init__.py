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


def __getattr__(name):
    """Lazy import to avoid loading sklearn when not needed."""
    if name in ("FeatureEngineer", "FeatureConfig"):
        from src.ml.training.feature_engineering import (
            FeatureEngineer,
            FeatureConfig,
        )
        return locals()[name]

    if name in ("ModelTrainer", "TrainingConfig"):
        from src.ml.training.model_trainer import (
            ModelTrainer,
            TrainingConfig,
        )
        return locals()[name]

    if name in ("ExperimentTracker", "Experiment"):
        from src.ml.training.experiment_tracker import (
            ExperimentTracker,
            Experiment,
        )
        return locals()[name]

    if name in ("ModelRegistry", "ModelMetadata", "ModelStatus"):
        from src.ml.training.model_registry import (
            ModelRegistry,
            ModelMetadata,
            ModelStatus,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
