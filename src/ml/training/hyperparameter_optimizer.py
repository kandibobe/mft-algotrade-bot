"""
Hyperparameter Optimization with Optuna for Trading Models
==========================================================

Advanced hyperparameter optimization focusing on simplicity and generalization
to combat overfitting in financial time series.

Based on roadmap requirements:
- Force simplicity: shallow trees, high regularization
- Optimize for Precision (not accuracy) with minimum 55% threshold
- Use Optuna for efficient Bayesian optimization
- Include ensemble of simple models (XGBoost, Random Forest, Logistic Regression)

Author: Stoic Citadel Team
Date: December 23, 2025
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.trial import Trial
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterOptimizerConfig:
    """Configuration for hyperparameter optimization."""

    # Optimization settings
    n_trials: int = 100  # Number of Optuna trials
    timeout: int | None = 3600  # Timeout in seconds (1 hour)
    n_jobs: int = -1  # Use all available cores

    # Validation settings
    n_splits: int = 5  # Number of time series splits
    test_size: float = 0.2  # Test size for each split

    # Model selection
    optimize_for: str = "precision"  # precision, f1, accuracy
    min_precision_threshold: float = 0.55  # Minimum precision required

    # Search space bounds
    xgb_max_depth_range: tuple[int, int] = (3, 7)  # Shallow trees for simplicity
    xgb_n_estimators_range: tuple[int, int] = (50, 200)
    xgb_learning_rate_range: tuple[float, float] = (0.01, 0.1)
    xgb_gamma_range: tuple[float, float] = (0.1, 1.0)  # High gamma for regularization
    xgb_min_child_weight_range: tuple[int, int] = (5, 20)  # High for simplicity
    xgb_subsample_range: tuple[float, float] = (0.5, 0.8)
    xgb_colsample_bytree_range: tuple[float, float] = (0.5, 0.8)
    xgb_reg_alpha_range: tuple[float, float] = (0.01, 1.0)  # L1 regularization
    xgb_reg_lambda_range: tuple[float, float] = (1.0, 5.0)  # L2 regularization

    # Random Forest bounds
    rf_n_estimators_range: tuple[int, int] = (50, 200)
    rf_max_depth_range: tuple[int, int] = (3, 10)
    rf_min_samples_split_range: tuple[int, int] = (5, 20)
    rf_min_samples_leaf_range: tuple[int, int] = (2, 10)

    # Logistic Regression bounds
    lr_c_range: tuple[float, float] = (0.01, 10.0)  # Inverse regularization strength
    lr_penalty: str = "l2"  # l1, l2, elasticnet

    # Early stopping
    early_stopping_rounds: int = 10
    early_stopping_patience: int = 3

    # Output
    save_best_model: bool = True
    output_dir: str = "user_data/models"


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for trading ML models.

    Key features:
    1. Forces simplicity to combat overfitting (shallow trees, high regularization)
    2. Optimizes for trading-specific metrics (Precision > 55%)
    3. Uses time-series aware cross-validation (no shuffling)
    4. Includes ensemble of simple models
    5. Bayesian optimization with Optuna for efficient search
    """

    def __init__(self, config: HyperparameterOptimizerConfig | None = None):
        self.config = config or HyperparameterOptimizerConfig()
        self.study = None
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.trial_results = []

        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    def optimize(
        self, X: pd.DataFrame, y: pd.Series, model_type: str = "xgboost"
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters for specified model type.

        Args:
            X: Feature DataFrame
            y: Target labels
            model_type: Type of model to optimize ("xgboost", "random_forest", "logistic_regression")

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        logger.info(f"Data shape: {X.shape}, target distribution: {y.value_counts().to_dict()}")

        # Create study
        study_name = f"{model_type}_optimization"
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        # Define objective function
        def objective(trial: Trial) -> float:
            return self._objective_function(trial, X, y, model_type)

        # Run optimization
        logger.info(f"Running {self.config.n_trials} trials...")
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        # Train best model on full data
        self.best_model = self._create_model(self.best_params, model_type)
        self.best_model.fit(X, y)

        # Save best model
        if self.config.save_best_model:
            self._save_best_model(model_type)

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_model": self.best_model,
            "study": self.study,
            "trial_results": self.trial_results,
        }

    def _objective_function(
        self, trial: Trial, X: pd.DataFrame, y: pd.Series, model_type: str
    ) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial
            X: Features
            y: Labels
            model_type: Type of model

        Returns:
            Validation score (precision, f1, or accuracy)
        """
        # Create model with trial parameters
        params = self._suggest_parameters(trial, model_type)
        model = self._create_model(params, model_type)

        # Time series cross-validation
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits, test_size=int(len(X) * self.config.test_size)
        )

        scores = []

        for _fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Calculate score based on optimization target
            if self.config.optimize_for == "precision":
                score = precision_score(y_val, y_pred, zero_division=0)
            elif self.config.optimize_for == "f1":
                score = f1_score(y_val, y_pred, zero_division=0)
            else:  # accuracy
                score = accuracy_score(y_val, y_pred)

            scores.append(score)

            # Early stopping check
            if len(scores) >= 2 and scores[-1] < scores[-2] * 0.95:
                # Performance dropped significantly
                trial.set_user_attr("early_stopped", True)
                break

        # Calculate average score
        avg_score = np.mean(scores) if scores else 0

        # Apply penalty if precision below threshold
        if (
            self.config.optimize_for == "precision"
            and avg_score < self.config.min_precision_threshold
        ):
            # Heavy penalty for not meeting minimum precision
            penalty = (self.config.min_precision_threshold - avg_score) * 10
            avg_score = max(0, avg_score - penalty)

        # Store trial results
        trial_result = {
            "trial_number": trial.number,
            "params": params,
            "score": avg_score,
            "fold_scores": scores,
            "model_type": model_type,
        }
        self.trial_results.append(trial_result)

        return avg_score

    def _suggest_parameters(self, trial: Trial, model_type: str) -> dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial
            model_type: Type of model

        Returns:
            Dictionary of hyperparameters
        """
        params = {}

        if model_type == "xgboost":
            # XGBoost parameters (force simplicity per roadmap)
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *self.config.xgb_n_estimators_range
                ),
                "max_depth": trial.suggest_int("max_depth", *self.config.xgb_max_depth_range),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *self.config.xgb_learning_rate_range, log=True
                ),
                "gamma": trial.suggest_float("gamma", *self.config.xgb_gamma_range),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", *self.config.xgb_min_child_weight_range
                ),
                "subsample": trial.suggest_float("subsample", *self.config.xgb_subsample_range),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *self.config.xgb_colsample_bytree_range
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", *self.config.xgb_reg_alpha_range),
                "reg_lambda": trial.suggest_float("reg_lambda", *self.config.xgb_reg_lambda_range),
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }

        elif model_type == "random_forest":
            # Random Forest parameters
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *self.config.rf_n_estimators_range
                ),
                "max_depth": trial.suggest_int("max_depth", *self.config.rf_max_depth_range),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", *self.config.rf_min_samples_split_range
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", *self.config.rf_min_samples_leaf_range
                ),
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced",
            }

        elif model_type == "logistic_regression":
            # Logistic Regression parameters
            params = {
                "C": trial.suggest_float("C", *self.config.lr_c_range, log=True),
                "penalty": self.config.lr_penalty,
                "random_state": 42,
                "max_iter": 1000,
                "solver": "saga" if self.config.lr_penalty in ["l1", "elasticnet"] else "lbfgs",
                "class_weight": "balanced",
            }

            if self.config.lr_penalty == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)

        return params

    def _create_model(self, params: dict[str, Any], model_type: str):
        """
        Create model instance from parameters.

        Args:
            params: Model parameters
            model_type: Type of model

        Returns:
            Model instance
        """
        if model_type == "xgboost":
            return xgb.XGBClassifier(**params)
        elif model_type == "random_forest":
            return RandomForestClassifier(**params)
        elif model_type == "logistic_regression":
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _save_best_model(self, model_type: str):
        """Save the best model to disk."""
        if self.best_model is None:
            logger.warning("No best model to save")
            return

        import pickle

        import joblib

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_optimized_{timestamp}.pkl"
        filepath = Path(self.config.output_dir) / filename

        # Save model
        with open(filepath, "wb") as f:
            pickle.dump(self.best_model, f)

        # Also save parameters and study
        results = {
            "model": self.best_model,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "model_type": model_type,
            "timestamp": timestamp,
            "config": self.config,
        }

        results_file = filepath.with_suffix(".joblib")
        joblib.dump(results, results_file)

        logger.info(f"Best model saved to {filepath}")
        logger.info(f"Results saved to {results_file}")

    def optimize_ensemble(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Optimize ensemble of simple models (XGBoost, Random Forest, Logistic Regression).

        Args:
            X: Features
            y: Labels

        Returns:
            Dictionary with ensemble results
        """
        logger.info("Optimizing ensemble of simple models")

        ensemble_results = {}

        # Optimize each model type
        for model_type in ["xgboost", "random_forest", "logistic_regression"]:
            logger.info(f"Optimizing {model_type}...")

            # Reset trial results for this model
            self.trial_results = []

            # Optimize
            results = self.optimize(X, y, model_type)
            ensemble_results[model_type] = results

            # Log results
            logger.info(f"{model_type} best score: {results['best_score']:.4f}")

        # Create weighted ensemble based on out-of-fold performance
        ensemble_model = self._create_weighted_ensemble(ensemble_results, X, y)

        # Save ensemble
        if self.config.save_best_model:
            self._save_ensemble(ensemble_results, ensemble_model)

        return {
            "ensemble_results": ensemble_results,
            "ensemble_model": ensemble_model,
            "weights": self._calculate_ensemble_weights(ensemble_results),
        }

    def _create_weighted_ensemble(
        self, ensemble_results: dict[str, Any], X: pd.DataFrame, y: pd.Series
    ):
        """
        Create weighted ensemble based on out-of-fold performance.

        Args:
            ensemble_results: Results from individual model optimizations
            X: Features
            y: Labels

        Returns:
            Ensemble model
        """
        from sklearn.ensemble import VotingClassifier

        # Get best models
        estimators = []
        weights = []

        for model_type, results in ensemble_results.items():
            model = results["best_model"]
            score = results["best_score"]

            estimators.append((model_type, model))
            weights.append(score)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting="soft",  # Use probability voting
            weights=weights.tolist(),
            n_jobs=-1,
        )

        # Fit ensemble
        ensemble.fit(X, y)

        logger.info(
            f"Ensemble created with weights: {dict(zip([e[0] for e in estimators], weights, strict=False))}"
        )

        return ensemble

    def _calculate_ensemble_weights(self, ensemble_results: dict[str, Any]) -> dict[str, float]:
        """Calculate ensemble weights based on model performance."""
        weights = {}
        total_score = 0

        for model_type, results in ensemble_results.items():
            score = results["best_score"]
            weights[model_type] = score
            total_score += score

        # Normalize
        if total_score > 0:
            for model_type in weights:
                weights[model_type] /= total_score

        return weights

    def _save_ensemble(self, ensemble_results: dict[str, Any], ensemble_model):
        """Save ensemble model and results."""
        import pickle

        import joblib

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ensemble_optimized_{timestamp}.pkl"
        filepath = Path(self.config.output_dir) / filename

        # Save ensemble model
        with open(filepath, "wb") as f:
            pickle.dump(ensemble_model, f)

        # Save ensemble results
        results = {
            "ensemble_model": ensemble_model,
            "ensemble_results": ensemble_results,
            "weights": self._calculate_ensemble_weights(ensemble_results),
            "timestamp": timestamp,
        }

        results_file = filepath.with_suffix(".joblib")
        joblib.dump(results, results_file)

        logger.info(f"Ensemble model saved to {filepath}")
        logger.info(f"Ensemble results saved to {results_file}")
