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
            FeatureConfig,
            FeatureEngineer,
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
            Experiment,
            ExperimentTracker,
        )

        return locals()[name]

    if name in ("ModelRegistry", "ModelMetadata", "ModelStatus"):
        from src.ml.training.model_registry import (
            ModelMetadata,
            ModelRegistry,
            ModelStatus,
        )

        return locals()[name]

    if name in (
        "TripleBarrierLabeler",
        "TripleBarrierConfig",
        "DynamicBarrierLabeler",
        "create_labels_for_training",
    ):
        from src.ml.training.labeling import (
            DynamicBarrierLabeler,
            TripleBarrierConfig,
            TripleBarrierLabeler,
            create_labels_for_training,
        )

        return locals()[name]

    if name in (
        "FeatureSelector",
        "FeatureSelectionConfig",
        "RecursiveFeatureEliminator",
    ):
        from src.ml.training.feature_selection import (
            FeatureSelectionConfig,
            FeatureSelector,
            RecursiveFeatureEliminator,
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
    "TripleBarrierLabeler",
    "TripleBarrierConfig",
    "DynamicBarrierLabeler",
    "create_labels_for_training",
    "FeatureSelector",
    "FeatureSelectionConfig",
    "RecursiveFeatureEliminator",
]
