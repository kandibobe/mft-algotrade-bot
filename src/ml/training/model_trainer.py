"""
Model Trainer
=============

Orchestrates model training with hyperparameter optimization.
"""

import json
import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model parameters
    model_type: str = "random_forest"  # random_forest, xgboost, lightgbm
    random_state: int = 42
    use_calibration: bool = True
    calibration_method: str = "sigmoid"  # sigmoid or isotonic

    # Data split
    test_size: float = 0.2
    validation_size: float = 0.1
    use_time_series_split: bool = True
    n_splits: int = 5

    # Training
    max_iter: int = 1000
    early_stopping_rounds: int = 50

    # Hyperparameter optimization
    optimize_hyperparams: bool = False
    n_trials: int = 100
    optimization_metric: str = "f1"  # accuracy, precision, recall, f1

    # Feature selection
    feature_selection: bool = True
    max_features: int | None = None

    # Output
    save_model: bool = True
    models_dir: str = "user_data/models"

    # Callbacks
    callbacks: list[Callable] = field(default_factory=list)


class ModelTrainer:
    """
    Trains ML models with hyperparameter optimization.

    Supports:
    - Random Forest
    - XGBoost
    - LightGBM
    - Hyperparameter optimization with Optuna
    - Cross-validation
    - Feature importance analysis

    Usage:
        trainer = ModelTrainer(config)
        model, metrics, features = trainer.train(X_train, y_train, X_val, y_val)
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """
        Initialize model trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model: Any = None
        self.feature_importance: pd.DataFrame | None = None
        self.training_history: list[Any] = []

        # Create models directory
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        hyperparams: dict | None = None,
    ) -> tuple[Any, dict[str, float], list[str]]:
        """
        Train model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            hyperparams: Pre-computed hyperparameters to use

        Returns:
            Tuple containing (model, metrics, feature_names)
        """
        logger.info(f"Training {self.config.model_type} model...")
        logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")

        # Split data if validation not provided
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                shuffle=False,
                stratify=None,
            )
            logger.info(f"Split into train: {len(X)}, val: {len(X_val)} (Time-series split)")

        # Feature selection
        if self.config.feature_selection:
            X, X_val = self._select_features(X, y, X_val)

        final_feature_names = X.columns.tolist()

        # Hyperparameter optimization or default training
        if hyperparams:
            logger.info(f"Using pre-computed hyperparameters: {hyperparams}")
            self.model = self._create_model(**hyperparams)
        elif self.config.optimize_hyperparams:
            self.model, best_params = self._optimize_hyperparams(X, y, X_val, y_val)
            logger.info(f"Best hyperparameters: {best_params}")
        else:
            self.model = self._create_model()

        # Wrap with calibration if requested
        if self.config.use_calibration:
            logger.info(
                f"Wrapping model with CalibratedClassifierCV ({self.config.calibration_method})"
            )

            if X_val is not None and y_val is not None:
                # If we have validation data, we fit the base model first,
                # then use the validation data for calibration.
                # In sklearn 1.8.0+, we use FrozenEstimator instead of cv="prefit".
                self.model.fit(X, y)
                self.model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(self.model),
                    method=self.config.calibration_method,
                    cv=None,  # Uses all provided data for calibration because estimator is frozen
                )
                self.model.fit(X_val, y_val)
            else:
                # If no validation data, let CalibratedClassifierCV handle cross-validation
                self.model = CalibratedClassifierCV(
                    estimator=self.model, method=self.config.calibration_method, cv=5
                )
                self.model.fit(X, y)
        else:
            if self.config.model_type == "xgboost" and X_val is not None and y_val is not None:
                self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            else:
                self.model.fit(X, y)

        # Calculate feature importance
        self._calculate_feature_importance(X)

        # Evaluate
        if X_val is not None and y_val is not None:
            metrics = self._evaluate(X_val, y_val)
        else:
            metrics = {}

        # Save model
        if self.config.save_model:
            self._save_model(metrics)

        logger.info(f"Training complete. Validation F1: {metrics.get('f1', 0.0):.4f}")

        return self.model, metrics, final_feature_names

    def _create_model(self, **hyperparams: Any) -> Any:
        """Create model with given hyperparameters."""
        if self.config.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            if "class_weight" not in hyperparams:
                hyperparams["class_weight"] = "balanced"

            return RandomForestClassifier(
                random_state=self.config.random_state, n_jobs=-1, **hyperparams
            )
        elif self.config.model_type == "xgboost":
            import xgboost as xgb

            default_params = {
                "max_depth": 3,
                "n_estimators": 500,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
            }
            merged_params = {**default_params, **hyperparams}
            return xgb.XGBClassifier(
                random_state=self.config.random_state,
                n_jobs=-1,
                early_stopping_rounds=self.config.early_stopping_rounds,
                **merged_params,
            )
        elif self.config.model_type == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                random_state=self.config.random_state, n_jobs=-1, **hyperparams
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _optimize_hyperparams(
        self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> tuple[Any, dict[str, Any]]:
        """Optimize hyperparameters using Optuna."""
        import optuna

        def objective(trial: optuna.Trial) -> float:
            if self.config.model_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
            elif self.config.model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                }
            elif self.config.model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                }
            else:
                params = {}

            model = self._create_model(**params)
            model.fit(X, y)
            y_pred = model.predict(X_val)

            if self.config.optimization_metric == "accuracy":
                return accuracy_score(y_val, y_pred)
            elif self.config.optimization_metric == "precision":
                return precision_score(y_val, y_pred, average="weighted", zero_division=0)
            elif self.config.optimization_metric == "recall":
                return recall_score(y_val, y_pred, average="weighted", zero_division=0)
            else:
                return f1_score(y_val, y_pred, average="weighted", zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)
        logger.info(f"Best {self.config.optimization_metric}: {study.best_value:.4f}")

        best_model = self._create_model(**study.best_params)
        best_model.fit(X, y)

        return best_model, study.best_params

    def _select_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select most important features."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel

        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.config.random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        selector = SelectFromModel(rf, prefit=True, max_features=self.config.max_features)

        X_train_selected = pd.DataFrame(
            selector.transform(X_train),
            columns=X_train.columns[selector.get_support()],
            index=X_train.index,
        )
        X_val_selected = pd.DataFrame(
            selector.transform(X_val),
            columns=X_val.columns[selector.get_support()],
            index=X_val.index,
        )
        logger.info(f"Selected {X_train_selected.shape[1]} features from {X_train.shape[1]} total")
        return X_train_selected, X_val_selected

    def _calculate_feature_importance(self, X: pd.DataFrame) -> None:
        """Calculate and store feature importance."""
        model_to_check = self.model
        if isinstance(self.model, CalibratedClassifierCV):
            model_to_check = self.model.estimator

        if hasattr(model_to_check, "feature_importances_"):
            importance = pd.DataFrame(
                {"feature": X.columns, "importance": model_to_check.feature_importances_}
            ).sort_values("importance", ascending=False)
            self.feature_importance = importance
            logger.info("Top 10 features:")
            for _idx, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def _evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
        """Evaluate model on validation set."""
        y_pred = self.model.predict(X_val)
        y_pred_proba = (
            self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, "predict_proba") else None
        )
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
        }
        if len(y_val.unique()) == 2 and y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score

            metrics["roc_auc"] = roc_auc_score(y_val, y_pred_proba)
        logger.info("Validation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        return metrics

    def _save_model(self, metrics: dict[str, float]) -> None:
        """Save trained model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.config.model_type}_{timestamp}.pkl"
        model_path = Path(self.config.models_dir) / model_name

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        metadata = {
            "model_type": self.config.model_type,
            "trained_at": timestamp,
            "metrics": metrics,
            "feature_count": (
                len(self.feature_importance) if self.feature_importance is not None else 0
            ),
            "config": {
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
                "use_calibration": self.config.use_calibration,
                "calibration_method": self.config.calibration_method,
            },
        }
        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.feature_importance is not None:
            importance_path = model_path.with_suffix(".csv")
            self.feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, list[float]]:
        """
        Perform time-series cross-validation.
        """
        logger.info(f"Running {self.config.n_splits}-fold cross-validation...")
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        cv_metrics: dict[str, list[float]] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"Fold {fold}/{self.config.n_splits}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self._create_model()
            if self.config.use_calibration:
                model = CalibratedClassifierCV(
                    estimator=model, method=self.config.calibration_method, cv=5
                )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            cv_metrics["accuracy"].append(float(accuracy_score(y_val, y_pred)))
            cv_metrics["precision"].append(
                float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
            )
            cv_metrics["recall"].append(
                float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
            )
            cv_metrics["f1"].append(
                float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
            )
        logger.info("Cross-validation results:")
        for metric, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"  {metric}: {mean_val:.4f} (+/- {std_val:.4f})")
        return cv_metrics

    def load_model(self, model_path: str) -> None:
        """Load trained model from disk."""
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from: {model_path}")

        metadata_path = Path(model_path).with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Model metadata: {metadata}")
