#!/usr/bin/env python3
"""
Nightly Hyperparameter Optimization for Stoic Citadel - Overnight Version
=======================================================================

Optimized for overnight execution with limited resource usage:
- Uses half of CPU cores (8 out of 16)
- Increased CPU limit to 98% (from 95%)
- Reduced memory monitoring threshold
- Longer timeout for full overnight run

Usage:
    python scripts/nightly_hyperopt_overnight.py --trials 1000 --timeout 43200

Author: Stoic Citadel Team
Date: December 25, 2025
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import psutil
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
from optuna.trial import Trial
from sklearn.model_selection import TimeSeriesSplit

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not available. Using scikit-learn GradientBoosting.")

from src.data.loader import get_ohlcv
from src.ml.training.feature_engineering import FeatureEngineer
from src.ml.training.labeling import TripleBarrierConfig, TripleBarrierLabeler
from src.utils.logger import get_logger
from src.utils.risk import calculate_sharpe_ratio

# Configure logging
logger = get_logger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ResourceMonitorOvernight:
    """Monitor system resources during optimization - Overnight version with relaxed limits."""

    def __init__(self, max_memory_gb: float = 24.0, min_disk_gb: float = 20.0):
        self.max_memory_gb = max_memory_gb
        self.min_disk_gb = min_disk_gb
        self.start_time = time.time()

    def check_resources(self) -> tuple[bool, str]:
        """Check if system has enough resources to continue."""
        issues = []

        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        if memory_gb > self.max_memory_gb:
            issues.append(f"Memory usage {memory_gb:.1f}GB > {self.max_memory_gb}GB limit")

        # Check disk space
        disk = psutil.disk_usage(".")
        disk_free_gb = disk.free / (1024**3)
        if disk_free_gb < self.min_disk_gb:
            issues.append(f"Disk free {disk_free_gb:.1f}GB < {self.min_disk_gb}GB minimum")

        # Check CPU load (relaxed for overnight)
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 98:  # Increased from 95% to 98%
            issues.append(f"CPU load {cpu_percent}% > 98%")

        if issues:
            return False, "; ".join(issues)
        return True, "OK"

    def get_stats(self) -> dict[str, Any]:
        """Get current resource statistics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(".")

        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_hours": (time.time() - self.start_time) / 3600,
            "memory_used_gb": memory.used / (1024**3),
            "memory_percent": memory.percent,
            "disk_free_gb": disk.free / (1024**3),
            "disk_percent": disk.percent,
            "cpu_percent": psutil.cpu_percent(interval=1),
        }


class NightlySharpeRatioObjectiveOvernight:
    """
    Objective function for Optuna that optimizes Sharpe Ratio - Overnight version.

    Enhanced version with:
    - Early stopping based on resource limits
    - Intermediate result saving
    - Better handling of edge cases
    - Reduced n_jobs for XGBoost
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        close_prices: pd.Series,
        n_splits: int = 5,
        resource_monitor: ResourceMonitorOvernight | None = None,
        n_jobs_xgb: int = 4,
    ):
        """
        Initialize objective function.

        Args:
            X: Feature matrix
            y: Target labels (1 for LONG, 0 for NEUTRAL/SHORT)
            close_prices: Close prices for PnL calculation
            n_splits: Number of folds for TimeSeriesSplit
            resource_monitor: Resource monitor instance
            n_jobs_xgb: Number of jobs for XGBoost (reduced for overnight)
        """
        self.X = X
        self.y = y
        self.close_prices = close_prices
        self.n_splits = n_splits
        self.resource_monitor = resource_monitor
        self.n_jobs_xgb = n_jobs_xgb

        # Initialize TimeSeriesSplit (no shuffling)
        self.cv = TimeSeriesSplit(n_splits=n_splits)

    def calculate_strategy_returns(
        self, y_pred_proba: pd.Series, close_prices: pd.Series
    ) -> pd.Series:
        """
        Calculate strategy returns based on trading signals.

        Simple strategy: Enter LONG position if probability > 0.55
        Hold for 1 period, exit at next close.

        Args:
            y_pred_proba: Predicted probabilities for class 1 (LONG)
            close_prices: Close prices

        Returns:
            Series of strategy returns
        """
        # Align indices
        common_idx = y_pred_proba.index.intersection(close_prices.index)
        y_pred_proba = y_pred_proba.loc[common_idx]
        close_prices = close_prices.loc[common_idx]

        # Generate trading signals: 1 if prob > 0.55, else 0
        signals = (y_pred_proba > 0.55).astype(int)

        # Calculate returns: future return if signal is 1, else 0
        future_returns = close_prices.pct_change().shift(-1)  # Next period return

        # Strategy returns: signal * future_return
        strategy_returns = signals * future_returns

        return strategy_returns.dropna()

    def __call__(self, trial: Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial

        Returns:
            Average Sharpe Ratio across all folds (to be maximized)
        """
        # Check resources before starting trial
        if self.resource_monitor:
            ok, msg = self.resource_monitor.check_resources()
            if not ok:
                logger.warning(f"Resource check failed: {msg}")
                raise optuna.TrialPruned(f"Resource limits exceeded: {msg}")

        # Suggest hyperparameters with strict constraints
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "gamma": trial.suggest_float("gamma", 0.1, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "n_jobs": self.n_jobs_xgb,  # Reduced from -1
            "verbosity": 0,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "objective": "binary:logistic",
        }

        sharpe_ratios = []

        # TimeSeriesSplit cross-validation
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X)):
            # Check resources between folds
            if self.resource_monitor:
                ok, msg = self.resource_monitor.check_resources()
                if not ok:
                    logger.warning(f"Resource check failed during fold {fold}: {msg}")
                    raise optuna.TrialPruned(f"Resource limits exceeded: {msg}")

            # Split data
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            val_close_prices = self.close_prices.iloc[val_idx]

            # Create and train model
            if XGB_AVAILABLE:
                model = xgb.XGBClassifier(**params)
            else:
                from sklearn.ensemble import GradientBoostingClassifier

                model = GradientBoostingClassifier(
                    n_estimators=params["n_estimators"],
                    learning_rate=params["learning_rate"],
                    max_depth=params["max_depth"],
                    subsample=params["subsample"],
                    random_state=42,
                )

            model.fit(X_train, y_train)

            # Predict probabilities on validation set
            y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1 (LONG)
            y_pred_proba_series = pd.Series(y_pred_proba, index=X_val.index)

            # Calculate strategy returns
            strategy_returns = self.calculate_strategy_returns(
                y_pred_proba_series, val_close_prices
            )

            # Calculate Sharpe Ratio
            if len(strategy_returns) > 1:
                sharpe = calculate_sharpe_ratio(strategy_returns)
                sharpe_ratios.append(sharpe)
            else:
                sharpe_ratios.append(0.0)

            # Pruning after first fold
            if fold == 0:
                trial.report(sharpe_ratios[-1], fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # Return average Sharpe Ratio across folds
        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0.0

        # Add custom attributes to trial
        trial.set_user_attr("completed_folds", len(sharpe_ratios))
        trial.set_user_attr("sharpe_std", np.std(sharpe_ratios) if len(sharpe_ratios) > 1 else 0.0)

        return avg_sharpe


class NightlyHyperparameterOptimizerOvernight:
    """
    Nightly hyperparameter optimizer for XGBoost trading models - Overnight version.

    Features:
    - Automatic resume from previous runs
    - Intermediate saving
    - Resource monitoring with relaxed limits
    - Telegram notifications
    - Detailed reporting
    - Reduced parallelization for stable overnight run
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.study = None
        self.best_params = None
        self.best_value = None
        self.resource_monitor = ResourceMonitorOvernight()

        # Create output directories
        self.output_dir = Path("user_data")
        self.nightly_dir = self.output_dir / "nightly_hyperopt_overnight"
        self.nightly_dir.mkdir(parents=True, exist_ok=True)

        # Database for resume capability
        self.db_path = self.nightly_dir / "nightly_hyperopt_overnight.db"

        logger.info(f"Nightly hyperopt overnight initialized. Output directory: {self.nightly_dir}")

    def check_data_freshness(
        self, symbol: str = "BTC/USDT", timeframe: str = "5m", max_age_hours: int = 24
    ) -> tuple[bool, str | None]:
        """
        Check if data is fresh enough for optimization.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            max_age_hours: Maximum age of data in hours

        Returns:
            Tuple of (is_fresh, message)
        """
        try:
            # Check if data file exists
            pair_filename = symbol.replace("/", "_")
            data_file = (
                self.output_dir / "data" / "binance" / f"{pair_filename}-{timeframe}.feather"
            )

            if not data_file.exists():
                return False, f"Data file not found: {data_file}"

            # Check file modification time
            mtime = data_file.stat().st_mtime
            file_age_hours = (time.time() - mtime) / 3600

            if file_age_hours > max_age_hours:
                return False, f"Data is {file_age_hours:.1f} hours old (max: {max_age_hours}h)"

            return True, f"Data is fresh ({file_age_hours:.1f} hours old)"

        except Exception as e:
            return False, f"Error checking data freshness: {e}"

    def load_and_prepare_data(
        self, symbol: str = "BTC/USDT", timeframe: str = "5m", days: int = 1095
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare data for optimization.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days to load (1095 for 3 years)

        Returns:
            Tuple of (X, y, close_prices) - features, labels, and close prices
        """
        logger.info(f"Loading {days} days of data for {symbol} {timeframe}")

        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Load OHLCV data
        df = get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            exchange="binance",
            use_cache=True,
        )

        logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        # Feature engineering
        logger.info("Generating features...")
        feature_engineer = FeatureEngineer()
        X = feature_engineer.fit_transform(df)

        # Labeling
        logger.info("Generating labels...")
        labeler = TripleBarrierLabeler(config=TripleBarrierConfig())
        y = labeler.label(df)

        # Extract close prices
        close_prices = df["close"]

        # Align features, labels, and close prices
        common_index = X.index.intersection(y.index).intersection(close_prices.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        close_prices = close_prices.loc[common_index]

        # Convert labels to binary: 1 for LONG (original 1), 0 for NEUTRAL/SHORT (-1, 0)
        y_binary = (y == 1).astype(int)

        # Remove rows with NaN
        nan_mask = y_binary.isna() | close_prices.isna()
        if nan_mask.any():
            logger.info(f"Removing {nan_mask.sum()} rows with NaN values")
            X = X[~nan_mask]
            y_binary = y_binary[~nan_mask]
            close_prices = close_prices[~nan_mask]

        logger.info(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Binary label distribution: {y_binary.value_counts().to_dict()}")

        return X, y_binary, close_prices

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        close_prices: pd.Series,
        n_trials: int = 1000,
        timeout: int | None = 43200,  # 12 hours default for overnight
        n_jobs: int = 8,  # Half of 16 cores
        resume: bool = True,
        n_jobs_xgb: int = 4,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization with resume capability.

        Args:
            X: Features
            y: Binary labels
            close_prices: Close prices for PnL calculation
            n_trials: Number of Optuna trials
            timeout: Optimization timeout in seconds
            n_jobs: Number of parallel jobs (8 for half of 16 cores)
            resume: Whether to resume from previous study
            n_jobs_xgb: Number of jobs for XGBoost (reduced)

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting overnight hyperparameter optimization")
        logger.info(
            f"Target: {n_trials} trials, timeout: {timeout}s, jobs: {n_jobs}, XGBoost jobs: {n_jobs_xgb}"
        )

        # Create objective function with resource monitoring
        objective = NightlySharpeRatioObjectiveOvernight(
            X,
            y,
            close_prices,
            n_splits=5,
            resource_monitor=self.resource_monitor,
            n_jobs_xgb=n_jobs_xgb,
        )

        # Create or load study
        study_name = "nightly_xgboost_sharpe_optimization_overnight"

        if resume and self.db_path.exists():
            logger.info(f"Resuming from existing study: {self.db_path}")
            storage = RDBStorage(f"sqlite:///{self.db_path}")
            self.study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
            )
            completed_trials = len(self.study.trials)
            logger.info(f"Resumed study with {completed_trials} completed trials")
            remaining_trials = max(0, n_trials - completed_trials)
            logger.info(f"Remaining trials to run: {remaining_trials}")
        else:
            # Create new study
            storage = RDBStorage(f"sqlite:///{self.db_path}")
            self.study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage,
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
                load_if_exists=resume,
            )
            logger.info("Created new study")
            remaining_trials = n_trials

        # Callback for intermediate saving
        def save_intermediate_results(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            """Save intermediate results every 50 trials."""
            if trial.number % 50 == 0:
                self._save_intermediate_results(study)
                # Log resource stats
                stats = self.resource_monitor.get_stats()
                logger.info(f"Trial {trial.number} completed. Resource stats: {stats}")

        # Run optimization
        try:
            self.study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=True,
                callbacks=[save_intermediate_results],
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Save partial results
            self._save_intermediate_results(self.study)
            raise

        # Get best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(f"Optimization completed. Best Sharpe Ratio: {self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        # Train final model with best parameters
        final_model = self._train_final_model(X, y, self.best_params, n_jobs_xgb)

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "study": self.study,
            "final_model": final_model,
            "timestamp": datetime.now().isoformat(),
            "total_trials": len(self.study.trials),
            "resource_stats": self.resource_monitor.get_stats(),
        }

    def _save_intermediate_results(self, study: optuna.Study):
        """Save intermediate results to disk."""
        intermediate_path = (
            self.nightly_dir
            / f"intermediate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Convert study to serializable format
        serializable_results = {
            "timestamp": datetime.now().isoformat(),
            "n_trials_completed": len(study.trials),
            "trials_summary": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                    "user_attrs": t.user_attrs,
                }
                for t in study.trials
            ],
        }

        # Safely get best value and params
        try:
            serializable_results["best_value"] = study.best_value
            serializable_results["best_params"] = study.best_params
        except ValueError:
            # No completed trials yet
            serializable_results["best_value"] = None
            serializable_results["best_params"] = {}

        with open(intermediate_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Intermediate results saved to {intermediate_path}")

    def _train_final_model(
        self, X: pd.DataFrame, y: pd.Series, params: dict, n_jobs_xgb: int = 4
    ) -> Any:
        """
        Train final model with best parameters on full dataset.

        Args:
            X: Features
            y: Labels
            params: Best hyperparameters
            n_jobs_xgb: Number of jobs for XGBoost

        Returns:
            Trained XGBoost model
        """
        logger.info("Training final model with best parameters...")

        # Prepare parameters for final training
        final_params = params.copy()
        final_params["n_estimators"] = 800  # Use max from search space
        final_params["n_jobs"] = n_jobs_xgb

        # Train on full dataset
        if XGB_AVAILABLE:
            model = xgb.XGBClassifier(**final_params)
        else:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=final_params["n_estimators"],
                learning_rate=final_params["learning_rate"],
                max_depth=final_params["max_depth"],
                subsample=final_params["subsample"],
                random_state=42,
            )

        model.fit(X, y)

        logger.info("Final model trained successfully")
        return model

    def save_results(self, results: dict[str, Any]):
        """
        Save optimization results to disk.

        Args:
            results: Optimization results
        """
        # Save best parameters
        params_path = self.nightly_dir / "best_params_overnight.json"
        with open(params_path, "w") as f:
            json.dump(results["best_params"], f, indent=2)
        logger.info(f"Best parameters saved to {params_path}")

        # Also save to main user_data directory for compatibility
        main_params_path = self.output_dir / "model_best_params_overnight.json"
        with open(main_params_path, "w") as f:
            json.dump(results["best_params"], f, indent=2)

        # Save full results
        results_path = self.nightly_dir / "full_results_overnight.json"

        # Convert study to serializable format
        serializable_results = {
            "best_params": results["best_params"],
            "best_value": results["best_value"],
            "timestamp": results["timestamp"],
            "total_trials": results["total_trials"],
            "resource_stats": results["resource_stats"],
            "trials_summary": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                    "user_attrs": t.user_attrs,
                }
                for t in results["study"].trials
            ],
        }

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Full results saved to {results_path}")

        # Save importance plot if possible
        try:
            self._plot_importance(results["final_model"])
        except Exception as e:
            logger.warning(f"Could not create importance plot: {e}")

        # Generate comparison report
        self.generate_report(results)

    def _plot_importance(self, model: Any):
        """
        Plot and save feature importance.

        Args:
            model: Trained model
        """
        try:
            import matplotlib.pyplot as plt

            # Get feature importance
            if XGB_AVAILABLE and hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                feature_names = model.get_booster().feature_names
            elif hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                feature_names = [f"feature_{i}" for i in range(len(importance))]
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance))]

            # Create DataFrame
            importance_df = (
                pd.DataFrame({"feature": feature_names, "importance": importance})
                .sort_values("importance", ascending=False)
                .head(20)
            )

            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(importance_df)), importance_df["importance"])
            plt.yticks(range(len(importance_df)), importance_df["feature"])
            plt.xlabel("Importance")
            plt.title("Top 20 Feature Importance (Overnight Model)")
            plt.tight_layout()

            # Save
            plot_path = self.nightly_dir / "feature_importance_overnight.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()

            logger.info(f"Feature importance plot saved to {plot_path}")

        except Exception as e:
            logger.warning(f"Could not create importance plot: {e}")

    def _send_notification(self, message: str, level: str = "info"):
        """
        Send notification (placeholder for Telegram/email integration).

        Args:
            message: Notification message
            level: info, warning, error
        """
        # This is a placeholder. In production, integrate with Telegram/email
        logger.info(f"Notification ({level}): {message}")

        # Example Telegram integration (uncomment and configure)
        # if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        #     send_telegram_message(message)

    def generate_report(self, results: dict[str, Any]):
        """
        Generate detailed comparison report.

        Args:
            results: Optimization results
        """
        report_path = self.nightly_dir / "optimization_report_overnight.md"

        # Load previous best parameters for comparison
        previous_params_path = self.output_dir / "model_best_params_3y.json"
        previous_best = None
        if previous_params_path.exists():
            with open(previous_params_path) as f:
                previous_best = json.load(f)

        with open(report_path, "w") as f:
            f.write("# Overnight Hyperparameter Optimization Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Best Sharpe Ratio:** {results['best_value']:.4f}\n")
            f.write(f"- **Total Trials:** {results['total_trials']}\n")
            f.write(
                f"- **Optimization Duration:** {results['resource_stats']['elapsed_hours']:.2f} hours\n\n"
            )

            f.write("## Best Parameters\n\n")
            f.write("```json\n")
            f.write(json.dumps(results["best_params"], indent=2))
            f.write("\n```\n\n")

            if previous_best:
                f.write("## Comparison with Previous Best (3-Year Model)\n\n")
                f.write("| Parameter | Previous Best | Overnight Best | Change |\n")
                f.write("|-----------|---------------|----------------|--------|\n")

                all_params = set(list(results["best_params"].keys()) + list(previous_best.keys()))
                for param in sorted(all_params):
                    prev = previous_best.get(param, "N/A")
                    curr = results["best_params"].get(param, "N/A")

                    if param in [
                        "learning_rate",
                        "gamma",
                        "reg_alpha",
                        "reg_lambda",
                        "subsample",
                        "colsample_bytree",
                    ]:
                        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
                            change_pct = ((curr - prev) / prev * 100) if prev != 0 else 0
                            f.write(f"| {param} | {prev:.4f} | {curr:.4f} | {change_pct:+.1f}% |\n")
                        else:
                            f.write(f"| {param} | {prev} | {curr} | - |\n")
                    else:
                        f.write(f"| {param} | {prev} | {curr} | - |\n")

                f.write("\n")

            f.write("## Resource Usage\n\n")
            stats = results["resource_stats"]
            f.write(
                f"- **Peak Memory Usage:** {stats['memory_used_gb']:.1f} GB ({stats['memory_percent']:.1f}%)\n"
            )
            f.write(f"- **Disk Free:** {stats['disk_free_gb']:.1f} GB\n")
            f.write(f"- **CPU Usage:** {stats['cpu_percent']:.1f}%\n")
            f.write(f"- **Elapsed Time:** {stats['elapsed_hours']:.2f} hours\n\n")

            f.write("## Recommendations\n\n")
            f.write(
                "1. **Model Deployment:** Consider deploying the new parameters if Sharpe Ratio improvement > 5%\n"
            )
            f.write("2. **Next Optimization:** Schedule next run in 3-7 days\n")
            f.write("3. **Data Freshness:** Ensure data is updated before next optimization\n")
            f.write("4. **Monitoring:** Monitor model performance after deployment\n")

        logger.info(f"Report generated: {report_path}")

        # Send notification
        improvement = ""
        if previous_best and "best_value" in results:
            # Try to load previous best value
            previous_results_path = self.output_dir / "hyperopt_results_3y.json"
            if previous_results_path.exists():
                try:
                    with open(previous_results_path) as f:
                        prev_results = json.load(f)
                        prev_value = prev_results.get("best_value", 0)
                        improvement_pct = (
                            ((results["best_value"] - prev_value) / abs(prev_value) * 100)
                            if prev_value != 0
                            else 0
                        )
                        improvement = f" ({improvement_pct:+.1f}% change)"
                except:
                    pass

        self._send_notification(
            f"Overnight hyperopt completed. Best Sharpe: {results['best_value']:.4f}{improvement}. "
            f"Trials: {results['total_trials']}. Report: {report_path}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Overnight hyperparameter optimization for XGBoost trading models"
    )

    parser.add_argument(
        "--symbol", type=str, default="BTC/USDT", help="Trading pair (default: BTC/USDT)"
    )
    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe (default: 5m)")
    parser.add_argument(
        "--days", type=int, default=1095, help="Number of days to load (default: 1095 for 3 years)"
    )
    parser.add_argument(
        "--trials", type=int, default=1000, help="Number of Optuna trials (default: 1000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=43200,
        help="Optimization timeout in seconds (default: 43200 = 12 hours)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs (default: 8 for half of 16 cores)",
    )
    parser.add_argument(
        "--n-jobs-xgb", type=int, default=4, help="Number of jobs for XGBoost (default: 4)"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Do not resume from previous study"
    )
    parser.add_argument(
        "--check-data", action="store_true", help="Check data freshness before starting"
    )
    parser.add_argument(
        "--max-data-age", type=int, default=24, help="Maximum data age in hours (default: 24)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (use only recent data for testing)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🌙 OVERNIGHT HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Symbol:          {args.symbol}")
    print(f"  Timeframe:       {args.timeframe}")
    print(f"  Days:            {args.days} ({args.days // 365} years)")
    print(f"  Trials:          {args.trials}")
    print(f"  Timeout:         {args.timeout}s ({args.timeout / 3600:.1f}h)")
    print(f"  N Jobs:          {args.n_jobs} (Optuna)")
    print(f"  N Jobs XGBoost:  {args.n_jobs_xgb}")
    print(f"  Resume:          {not args.no_resume}")
    print(f"  Check Data:      {args.check_data}")
    print(f"  Max Data Age:    {args.max_data_age}h")
    print(f"  Quick Mode:      {args.quick}")
    print("\n" + "=" * 70)

    try:
        # Initialize optimizer
        optimizer = NightlyHyperparameterOptimizerOvernight()

        # Check data freshness
        if args.check_data:
            is_fresh, message = optimizer.check_data_freshness(
                symbol=args.symbol, timeframe=args.timeframe, max_age_hours=args.max_data_age
            )
            if not is_fresh:
                print(f"\n❌ Data freshness check failed: {message}")
                print("Consider running data download first:")
                print("  python scripts/download_data.py --pair BTC/USDT --days 30")
                return 1
            print(f"\n✅ {message}")

        # Load and prepare data
        X, y, close_prices = optimizer.load_and_prepare_data(
            symbol=args.symbol, timeframe=args.timeframe, days=args.days
        )

        # For quick mode, use only recent data
        if args.quick:
            print("\n⚡ Quick mode: Using only 1000 most recent samples")
            X = X.tail(1000)
            y = y.tail(1000)
            close_prices = close_prices.tail(1000)
            args.trials = min(args.trials, 50)  # Reduce trials for quick mode

        # Run optimization
        results = optimizer.optimize(
            X=X,
            y=y,
            close_prices=close_prices,
            n_trials=args.trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            resume=not args.no_resume,
            n_jobs_xgb=args.n_jobs_xgb,
        )

        # Save results
        optimizer.save_results(results)

        print("\n" + "=" * 70)
        print("✅ OVERNIGHT OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nBest Sharpe Ratio: {results['best_value']:.4f}")
        print("\nResults saved to: user_data/nightly_hyperopt_overnight/")
        print("Report: user_data/nightly_hyperopt_overnight/optimization_report_overnight.md")
        print("\n" + "=" * 70)

        return 0

    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
