"""
Hyperparameter Optimizer for ML Models
======================================

Uses Optuna to find best parameters for LightGBM, XGBoost, and other models.
"""

import logging
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.config import config

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Optuna-based optimizer for ML models."""

    def __init__(
        self,
        n_trials: int = 100,
        study_name: str = "stoic_optimization",
        models_dir: str | None = None,
        model_type: str = "lightgbm", # Default model type
        metric: str = "f1"
    ):
        cfg = config()
        self.n_trials = n_trials
        self.study_name = study_name
        self.models_dir = Path(models_dir or cfg.paths.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.metric = metric

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Run optimization study.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Starting optimization study: {self.study_name} for {self.model_type}")
        
        # Split data for validation during optimization
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        def objective(trial: optuna.Trial) -> float:
            # Define search space based on model type
            if self.model_type == "lightgbm":
                import lightgbm as lgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "n_jobs": -1,
                    "random_state": 42,
                    "verbose": -1
                }
                model = lgb.LGBMClassifier(**params)
                
            elif self.model_type == "xgboost":
                import xgboost as xgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "n_jobs": -1,
                    "random_state": 42,
                    "verbosity": 0
                }
                model = xgb.XGBClassifier(**params)
                
            elif self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "class_weight": "balanced",
                    "n_jobs": -1,
                    "random_state": 42
                }
                model = RandomForestClassifier(**params)
                
            else:
                # Fallback to simple RF
                from sklearn.ensemble import RandomForestClassifier
                params = {"n_estimators": 100}
                model = RandomForestClassifier(**params)

            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metric
            if self.metric == "accuracy":
                return accuracy_score(y_val, y_pred)
            elif self.metric == "precision":
                return precision_score(y_val, y_pred, average="weighted", zero_division=0)
            elif self.metric == "recall":
                return recall_score(y_val, y_pred, average="weighted", zero_division=0)
            else: # F1
                return f1_score(y_val, y_pred, average="weighted", zero_division=0)

        # Create study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", study_name=self.study_name, load_if_exists=False) # Don't load if exists to avoid stale studies
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        logger.info(f"Optimization finished. Best {self.metric}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
