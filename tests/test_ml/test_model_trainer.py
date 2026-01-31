"""
Tests for Model Trainer
"""


import numpy as np
import pandas as pd
import pytest

from src.ml.training.model_trainer import (
    ModelTrainer,
    TrainingConfig,
)


@pytest.fixture
def sample_training_data():
    """Generate sample training data."""
    np.random.seed(42)

    # Generate 100 samples with 10 features
    X = pd.DataFrame(
        np.random.randn(100, 10),
        columns=[f"feature_{i}" for i in range(10)]
    )

    # Generate binary target (classification)
    y = pd.Series((X['feature_0'] + X['feature_1'] > 0).astype(int))

    return X, y


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()

        assert config.model_type == "random_forest"
        assert config.optimize_hyperparams is False
        assert config.n_trials == 100
        assert config.n_splits == 5
        assert config.random_state == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            model_type="xgboost",
            optimize_hyperparams=True,
            n_trials=50,
        )

        assert config.model_type == "xgboost"
        assert config.optimize_hyperparams is True
        assert config.n_trials == 50


class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_initialization(self):
        """Test model trainer initialization."""
        trainer = ModelTrainer()

        assert trainer.config is not None
        assert isinstance(trainer.config, TrainingConfig)

    def test_train_random_forest(self, sample_training_data):
        """Test training Random Forest model."""
        X, y = sample_training_data

        config = TrainingConfig(
            model_type="random_forest",
            optimize_hyperparams=False,
            feature_selection=False,
            save_model=False,
        )
        trainer = ModelTrainer(config)

        model, metrics, _ = trainer.train(X, y)

        # Check model was trained
        assert model is not None
        assert hasattr(model, 'predict')

        # Check metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

        # Check reasonable performance
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1

    def test_train_with_validation(self, sample_training_data):
        """Test training with validation set."""
        X, y = sample_training_data

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        config = TrainingConfig(
            feature_selection=False,
            save_model=False,
        )
        trainer = ModelTrainer(config)
        model, metrics, _ = trainer.train(X_train, y_train, X_val, y_val)

        # Should have metrics
        assert 'accuracy' in metrics
        assert 'f1' in metrics

    def test_get_feature_importance(self, sample_training_data):
        """Test feature importance extraction."""
        X, y = sample_training_data

        config = TrainingConfig(
            feature_selection=False,
            save_model=False,
        )
        trainer = ModelTrainer(config)
        model, metrics, _ = trainer.train(X, y)

        # Feature importance should be stored in trainer
        assert trainer.feature_importance is not None
        assert isinstance(trainer.feature_importance, pd.DataFrame)
        assert 'feature' in trainer.feature_importance.columns
        assert 'importance' in trainer.feature_importance.columns

    def test_invalid_model_type(self, sample_training_data):
        """Test invalid model type handling."""
        X, y = sample_training_data

        config = TrainingConfig(
            model_type="invalid_model",
            feature_selection=False,
            save_model=False,
        )
        trainer = ModelTrainer(config)

        # Should raise error
        with pytest.raises(ValueError):
            trainer.train(X, y)

    def test_cross_validation(self, sample_training_data):
        """Test cross-validation."""
        X, y = sample_training_data

        config = TrainingConfig(
            n_splits=3,
            feature_selection=False,
        )
        trainer = ModelTrainer(config)

        cv_metrics = trainer.cross_validate(X, y)

        # Should have metrics for each fold
        assert 'accuracy' in cv_metrics
        assert 'f1' in cv_metrics
        assert len(cv_metrics['accuracy']) == 3
        assert len(cv_metrics['f1']) == 3

    def test_feature_selection(self, sample_training_data):
        """Test feature selection."""
        X, y = sample_training_data

        config = TrainingConfig(
            feature_selection=True,
            max_features=5,
            save_model=False,
        )

        trainer = ModelTrainer(config)
        model, metrics, _ = trainer.train(X, y)

        # Model should be trained
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])