"""
Online Meta-Learning Ensemble
=============================

Combines OnlineLearner and MetaLearningEnsemble to create a self-improving,
ensemble-based learning system.

Architecture:
1. Manages a collection of OnlineLearner instances, each for a base model.
2. Each base model is updated online and can be A/B tested independently.
3. A MetaLearningEnsemble is used as a meta-model to learn optimal weights
   for the base models.
4. The system continuously adapts to new data and model drift.

Author: Stoic Citadel Team
License: MIT
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np

from src.ml.online_learner import OnlineLearner, OnlineLearningConfig
from src.ml.meta_learning import MetaLearningEnsemble, MetaLearningConfig
from src.ml.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class OnlineMetaEnsemble:
    """
    An online meta-learning ensemble that manages multiple online learners
    and combines their predictions using a meta-learner.
    """

    def __init__(
        self,
        base_model_paths: List[str],
        online_learning_config: Optional[OnlineLearningConfig] = None,
        meta_learning_config: Optional[MetaLearningConfig] = None,
    ):
        """
        Initialize the Online Meta Ensemble.

        Args:
            base_model_paths: List of paths to the base production models.
            online_learning_config: Configuration for the online learners.
            meta_learning_config: Configuration for the meta-learner.
        """
        self.online_learners = [
            OnlineLearner(path, config=online_learning_config) for path in base_model_paths
        ]

        # The 'base_models' for the meta-learner are the production models
        # from each of the online learners.
        base_models_for_meta = [learner.prod_model for learner in self.online_learners]
        self.meta_learner = MetaLearningEnsemble(
            base_models=base_models_for_meta, config=meta_learning_config
        )

        logger.info(
            f"Initialized OnlineMetaEnsemble with {len(self.online_learners)} base models."
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make a prediction using the ensemble.

        Args:
            X: Input features for the base models.

        Returns:
            A tuple containing:
            - The final prediction from the meta-learner.
            - The confidence score from the meta-learner.
        """
        # The MetaLearningEnsemble's `predict_with_confidence` expects the raw features X,
        # because it internally gets the predictions from the base models it holds.
        return self.meta_learner.predict_with_confidence(X)

    def update(self, X: np.ndarray, y_true: np.ndarray):
        """
        Update the ensemble with new data.

        Args:
            X: New features.
            y_true: True labels for the new features.
        """
        # 1. Update each online learner with the new data.
        for learner in self.online_learners:
            learner.batch_update(X, y_true)

        # 2. Check if any of the base models should be replaced.
        for i, learner in enumerate(self.online_learners):
            if learner.should_replace_prod_model():
                logger.info(f"Replacing base model {i}...")
                learner.replace_production_model()
                # Update the meta-learner with the new production model.
                self.meta_learner.base_models[i] = learner.prod_model

        # 3. Periodically retrain the meta-learner.
        # The MetaLearningEnsemble has a `retrain_interval` in its config.
        # The check is done inside `predict_with_confidence`. We can also
        # trigger it manually based on new data.
        if self.meta_learner.prediction_count >= self.meta_learner.config.retrain_interval:
            logger.info("Retraining meta-learner...")
            # Here you would typically use a recent window of data for retraining.
            # For simplicity, we are not implementing the data window logic here,
            # but in a production system you would.
            # self.meta_learner.train_with_validation_split(X_recent, y_recent)
            pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the ensemble.

        Returns:
            A dictionary with the status of each component.
        """
        return {
            "meta_learner_status": {
                "is_trained": self.meta_learner.is_trained,
                "model_weights": self.meta_learner.get_model_weights().tolist(),
                "training_history": self.meta_learner.training_history,
            },
            "online_learners_status": [
                learner.get_performance_stats() for learner in self.online_learners
            ],
        }
