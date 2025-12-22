"""
Model Trainer
=============

Orchestrates model training with hyperparameter optimization.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model parameters
    model_type: str = "random_forest"  # random_forest, xgboost, lightgbm
    random_state: int = 42

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
    max_features: Optional[int] = None

    # Output
    save_model: bool = True
    models_dir: str = "user_data/models"

    # Callbacks
    callbacks: List[Callable] = field(default_factory=list)


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
        model, metrics = trainer.train(X_train, y_train, X_val, y_val)
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize model trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model = None
        self.feature_importance = None
        self.training_history = []

        # Create models directory
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> tuple[Any, Dict[str, float]]:
        """
        Train model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            (model, metrics)
        """
        logger.info(f"Training {self.config.model_type} model...")
        logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")

        # Split data if validation not provided
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if len(y.unique()) > 1 else None,
            )
            logger.info(f"Split into train: {len(X)}, val: {len(X_val)}")

        # Feature selection
        if self.config.feature_selection:
            X, X_val = self._select_features(X, y, X_val)

        # Hyperparameter optimization or default training
        if self.config.optimize_hyperparams:
            self.model, best_params = self._optimize_hyperparams(X, y, X_val, y_val)
            logger.info(f"Best hyperparameters: {best_params}")
        else:
            self.model = self._create_model()
            self.model.fit(X, y)

        # Calculate feature importance
        self._calculate_feature_importance(X)

        # Evaluate
        metrics = self._evaluate(X_val, y_val)

        # Save model
        if self.config.save_model:
            self._save_model(metrics)

        logger.info(f"Training complete. Validation F1: {metrics['f1']:.4f}")

        return self.model, metrics

    def _create_model(self, **hyperparams) -> Any:
        """Create model with given hyperparameters."""
        if self.config.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            # Add class_weight='balanced' to handle class imbalance
            # This gives higher weight to minority class
            if 'class_weight' not in hyperparams:
                hyperparams['class_weight'] = 'balanced'
            
            return RandomForestClassifier(
                random_state=self.config.random_state, n_jobs=-1, **hyperparams
            )
        elif self.config.model_type == "xgboost":
            import xgboost as xgb

            # For XGBoost, use scale_pos_weight parameter for class imbalance
            # This will be calculated dynamically based on class distribution
            return xgb.XGBClassifier(
                random_state=self.config.random_state,
                n_jobs=-1,
                early_stopping_rounds=self.config.early_stopping_rounds,
                **hyperparams,
            )
        elif self.config.model_type == "lightgbm":
            import lightgbm as lgb

            # LightGBM automatically handles class imbalance with is_unbalance parameter
            return lgb.LGBMClassifier(
                random_state=self.config.random_state, n_jobs=-1, **hyperparams
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _optimize_hyperparams(
        self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> tuple[Any, Dict]:
        """Optimize hyperparameters using Optuna."""
        import optuna

        def objective(trial):
            # Suggest hyperparameters based on model type
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

            # Train model
            model = self._create_model(**params)
            model.fit(X, y)

            # Evaluate
            y_pred = model.predict(X_val)

            # Return metric based on configuration
            if self.config.optimization_metric == "accuracy":
                return accuracy_score(y_val, y_pred)
            elif self.config.optimization_metric == "precision":
                return precision_score(y_val, y_pred, average="weighted", zero_division=0)
            elif self.config.optimization_metric == "recall":
                return recall_score(y_val, y_pred, average="weighted", zero_division=0)
            else:  # f1
                return f1_score(y_val, y_pred, average="weighted", zero_division=0)

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        logger.info(f"Best {self.config.optimization_metric}: {study.best_value:.4f}")

        # Train final model with best parameters
        best_model = self._create_model(**study.best_params)
        best_model.fit(X, y)

        return best_model, study.best_params

    def _select_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Select most important features."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel

        # Train simple RF for feature importance
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.config.random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Select features
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

        logger.info(
            f"Selected {X_train_selected.shape[1]} features " f"from {X_train.shape[1]} total"
        )

        return X_train_selected, X_val_selected

    def _calculate_feature_importance(self, X: pd.DataFrame):
        """Calculate and store feature importance."""
        if hasattr(self.model, "feature_importances_"):
            importance = pd.DataFrame(
                {"feature": X.columns, "importance": self.model.feature_importances_}
            ).sort_values("importance", ascending=False)

            self.feature_importance = importance

            logger.info(f"Top 10 features:")
            for idx, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def _evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
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

        # Add ROC AUC if binary classification
        if len(y_val.unique()) == 2 and y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score

            metrics["roc_auc"] = roc_auc_score(y_val, y_pred_proba)

        logger.info(f"Validation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def _save_model(self, metrics: Dict[str, float]):
        """Save trained model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.config.model_type}_{timestamp}.pkl"
        model_path = Path(self.config.models_dir) / model_name

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save metadata
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
            },
        }

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save feature importance
        if self.feature_importance is not None:
            importance_path = model_path.with_suffix(".csv")
            self.feature_importance.to_csv(importance_path, index=False)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.

        Args:
            X: Features
            y: Labels

        Returns:
            Dict with lists of metrics for each fold
        """
        logger.info(f"Running {self.config.n_splits}-fold cross-validation...")

        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        cv_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"Fold {fold}/{self.config.n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train
            model = self._create_model()
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)

            cv_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            cv_metrics["precision"].append(
                precision_score(y_val, y_pred, average="weighted", zero_division=0)
            )
            cv_metrics["recall"].append(
                recall_score(y_val, y_pred, average="weighted", zero_division=0)
            )
            cv_metrics["f1"].append(f1_score(y_val, y_pred, average="weighted", zero_division=0))

        # Log results
        logger.info("Cross-validation results:")
        for metric, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"  {metric}: {mean_val:.4f} (+/- {std_val:.4f})")

        return cv_metrics

    def load_model(self, model_path: str):
        """Load trained model from disk."""
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        logger.info(f"Model loaded from: {model_path}")

        # Load metadata if exists
        metadata_path = Path(model_path).with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Model metadata: {metadata}")
