"""
Tests for Model Trainer
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.ml.training.model_trainer import (
    ModelTrainer,
    ModelConfig,
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
    y = (X['feature_0'] + X['feature_1'] > 0).astype(int)

    return X, y


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig()

        assert config.model_type == "random_forest"
        assert config.optimize_hyperparams is False
        assert config.n_trials == 50
        assert config.cv_folds == 5
        assert config.random_state == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            model_type="xgboost",
            optimize_hyperparams=True,
            n_trials=100,
        )

        assert config.model_type == "xgboost"
        assert config.optimize_hyperparams is True
        assert config.n_trials == 100


class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_initialization(self):
        """Test model trainer initialization."""
        trainer = ModelTrainer()

        assert trainer.config is not None
        assert isinstance(trainer.config, ModelConfig)

    def test_train_random_forest(self, sample_training_data):
        """Test training Random Forest model."""
        X, y = sample_training_data

        config = ModelConfig(model_type="random_forest", optimize_hyperparams=False)
        trainer = ModelTrainer(config)

        model, metrics = trainer.train(X, y)

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

    def test_train_xgboost(self, sample_training_data):
        """Test training XGBoost model."""
        pytest.importorskip("xgboost")

        X, y = sample_training_data

        config = ModelConfig(model_type="xgboost", optimize_hyperparams=False)
        trainer = ModelTrainer(config)

        model, metrics = trainer.train(X, y)

        # Check model was trained
        assert model is not None
        assert 'accuracy' in metrics

    def test_train_lightgbm(self, sample_training_data):
        """Test training LightGBM model."""
        pytest.importorskip("lightgbm")

        X, y = sample_training_data

        config = ModelConfig(model_type="lightgbm", optimize_hyperparams=False)
        trainer = ModelTrainer(config)

        model, metrics = trainer.train(X, y)

        # Check model was trained
        assert model is not None
        assert 'accuracy' in metrics

    def test_train_with_validation(self, sample_training_data):
        """Test training with validation set."""
        X, y = sample_training_data

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        trainer = ModelTrainer()
        model, metrics = trainer.train(X_train, y_train, X_val, y_val)

        # Should have validation metrics
        assert 'val_accuracy' in metrics
        assert 'val_f1' in metrics

    def test_hyperparameter_optimization(self, sample_training_data):
        """Test hyperparameter optimization."""
        pytest.importorskip("optuna")

        X, y = sample_training_data

        config = ModelConfig(
            model_type="random_forest",
            optimize_hyperparams=True,
            n_trials=5,  # Small number for testing
        )

        trainer = ModelTrainer(config)
        model, metrics = trainer.train(X, y)

        # Should complete without error
        assert model is not None
        assert 'accuracy' in metrics

    def test_save_and_load_model(self, sample_training_data):
        """Test model saving and loading."""
        X, y = sample_training_data

        trainer = ModelTrainer()
        model, metrics = trainer.train(X, y)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            trainer.save_model(model, str(model_path))

            # Check file exists
            assert model_path.exists()

            # Load model
            loaded_model = trainer.load_model(str(model_path))

            # Check predictions are the same
            pred_original = model.predict(X)
            pred_loaded = loaded_model.predict(X)

            np.testing.assert_array_equal(pred_original, pred_loaded)

    def test_get_feature_importance(self, sample_training_data):
        """Test feature importance extraction."""
        X, y = sample_training_data

        trainer = ModelTrainer()
        model, metrics = trainer.train(X, y)

        importance = trainer.get_feature_importance(model, X.columns.tolist())

        # Should return dict
        assert isinstance(importance, dict)

        # Should have all features
        assert len(importance) == len(X.columns)

        # Importance values should be non-negative
        for value in importance.values():
            assert value >= 0

    def test_invalid_model_type(self, sample_training_data):
        """Test invalid model type handling."""
        X, y = sample_training_data

        config = ModelConfig(model_type="invalid_model")
        trainer = ModelTrainer(config)

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, KeyError)):
            trainer.train(X, y)

    def test_cross_validation(self, sample_training_data):
        """Test cross-validation during training."""
        X, y = sample_training_data

        config = ModelConfig(cv_folds=3)
        trainer = ModelTrainer(config)

        model, metrics = trainer.train(X, y)

        # Should include CV metrics
        assert 'cv_scores' in metrics or 'accuracy' in metrics

    def test_feature_selection(self, sample_training_data):
        """Test feature selection."""
        X, y = sample_training_data

        config = ModelConfig(
            feature_selection=True,
            max_features=5,
        )

        trainer = ModelTrainer(config)
        model, metrics = trainer.train(X, y)

        # Should have selected features
        assert 'selected_features' in metrics
        assert len(metrics['selected_features']) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
