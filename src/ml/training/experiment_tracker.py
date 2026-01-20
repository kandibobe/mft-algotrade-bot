"""
Experiment Tracker
==================

Track ML experiments with Weights & Biases or MLflow.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Experiment metadata."""

    name: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    run_id: str | None = None


class ExperimentTracker:
    """
    Track ML experiments with Weights & Biases.

    Features:
    - Log hyperparameters
    - Log metrics during training
    - Log model artifacts
    - Compare experiments
    - Link experiments to backtests

    Usage:
        tracker = ExperimentTracker(project="stoic-citadel")
        tracker.start_run("my_experiment")
        tracker.log_config({"learning_rate": 0.01})
        tracker.log_metrics({"accuracy": 0.95})
        tracker.finish()
    """

    def __init__(
        self,
        project: str = "stoic-citadel-ml",
        entity: str | None = None,
        backend: str = "wandb",  # "wandb" or "mlflow"
    ):
        """
        Initialize experiment tracker.

        Args:
            project: Project name
            entity: W&B entity (username/team)
            backend: Tracking backend ("wandb" or "mlflow")
        """
        self.project = project
        self.entity = entity
        self.backend = backend
        self.current_run = None
        self.experiments: list[Experiment] = []

        # Initialize backend
        if backend == "wandb":
            try:
                import wandb

                self.wandb = wandb
                logger.info(f"Initialized W&B tracker for project: {project}")
            except ImportError:
                logger.warning("wandb not installed. Install with: pip install wandb")
                self.wandb = None
        elif backend == "mlflow":
            try:
                import mlflow

                self.mlflow = mlflow
                mlflow.set_experiment(project)
                logger.info(f"Initialized MLflow tracker for project: {project}")
            except ImportError:
                logger.warning("mlflow not installed. Install with: pip install mlflow")
                self.mlflow = None
        else:
            logger.warning(f"Unknown backend: {backend}. Using offline mode.")
            self.wandb = None
            self.mlflow = None

    def start_run(
        self,
        name: str,
        description: str = "",
        config: dict | None = None,
        tags: list[str] | None = None,
    ):
        """
        Start a new experiment run.

        Args:
            name: Experiment name
            description: Description
            config: Configuration dict
            tags: List of tags
        """
        experiment = Experiment(
            name=name, description=description, config=config or {}, tags=tags or []
        )

        if self.backend == "wandb" and self.wandb:
            self.current_run = self.wandb.init(
                project=self.project,
                entity=self.entity,
                name=name,
                config=config,
                tags=tags,
                notes=description,
            )
            experiment.run_id = self.current_run.id
            logger.info(f"Started W&B run: {name} (ID: {experiment.run_id})")

        elif self.backend == "mlflow" and self.mlflow:
            self.current_run = self.mlflow.start_run(run_name=name)
            if config:
                self.mlflow.log_params(config)
            if tags:
                for tag in tags:
                    self.mlflow.set_tag("tags", tag)
            experiment.run_id = self.current_run.info.run_id
            logger.info(f"Started MLflow run: {name} (ID: {experiment.run_id})")

        else:
            # Offline mode
            experiment.run_id = f"offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Started offline run: {name}")

        self.experiments.append(experiment)

        return experiment

    def log_config(self, config: dict[str, Any]):
        """
        Log configuration/hyperparameters.

        Args:
            config: Configuration dictionary
        """
        if not self.current_run:
            logger.warning("No active run. Call start_run() first.")
            return

        if self.backend == "wandb" and self.wandb:
            self.wandb.config.update(config)

        elif self.backend == "mlflow" and self.mlflow:
            self.mlflow.log_params(config)

        logger.debug(f"Logged config: {config}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """
        Log metrics.

        Args:
            metrics: Metrics dictionary
            step: Training step/epoch
        """
        if not self.current_run:
            logger.warning("No active run. Call start_run() first.")
            return

        if self.backend == "wandb" and self.wandb:
            self.wandb.log(metrics, step=step)

        elif self.backend == "mlflow" and self.mlflow:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step=step)

        # Store in experiment
        if self.experiments:
            self.experiments[-1].metrics.update(metrics)

        logger.debug(f"Logged metrics: {metrics}")

    def log_backtest_results(self, backtest_results: dict[str, Any]):
        """
        Log backtest results.

        Args:
            backtest_results: Backtest metrics
        """
        # Extract key metrics
        metrics = {
            "backtest/sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
            "backtest/max_drawdown": backtest_results.get("max_drawdown", 0),
            "backtest/total_trades": backtest_results.get("total_trades", 0),
            "backtest/win_rate": backtest_results.get("win_rate", 0),
            "backtest/profit_total": backtest_results.get("profit_total", 0),
        }

        self.log_metrics(metrics)

        # Log full results as artifact
        if self.backend == "wandb" and self.wandb:
            self.wandb.log(
                {
                    "backtest_results": self.wandb.Table(
                        dataframe=backtest_results if isinstance(backtest_results, dict) else None
                    )
                }
            )

        logger.info(f"Logged backtest results: Sharpe={metrics['backtest/sharpe_ratio']:.2f}")

    def log_model(self, model_path: str, model_name: str = "model"):
        """
        Log model artifact.

        Args:
            model_path: Path to model file
            model_name: Name for the model
        """
        if not self.current_run:
            logger.warning("No active run. Call start_run() first.")
            return

        if self.backend == "wandb" and self.wandb:
            self.wandb.save(model_path, base_path=str(Path(model_path).parent))

        elif self.backend == "mlflow" and self.mlflow:
            self.mlflow.log_artifact(model_path)

        logger.info(f"Logged model: {model_path}")

    def log_feature_importance(self, feature_importance: dict[str, float]):
        """
        Log feature importance.

        Args:
            feature_importance: Dict of feature -> importance
        """
        if self.backend == "wandb" and self.wandb:
            import pandas as pd

            # Create table
            df = pd.DataFrame(
                [{"feature": feat, "importance": imp} for feat, imp in feature_importance.items()]
            ).sort_values("importance", ascending=False)

            self.wandb.log({"feature_importance": self.wandb.Table(dataframe=df)})

        logger.info(f"Logged feature importance for {len(feature_importance)} features")

    def log_confusion_matrix(
        self, y_true: list, y_pred: list, class_names: list[str] | None = None
    ):
        """
        Log confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
        """
        if self.backend == "wandb" and self.wandb:
            self.wandb.log(
                {
                    "confusion_matrix": self.wandb.plot.confusion_matrix(
                        y_true=y_true, preds=y_pred, class_names=class_names
                    )
                }
            )

        logger.info("Logged confusion matrix")

    def log_roc_curve(self, y_true: list, y_probas: list, class_names: list[str] | None = None):
        """
        Log ROC curve.

        Args:
            y_true: True labels
            y_probas: Predicted probabilities
            class_names: Class names
        """
        if self.backend == "wandb" and self.wandb:
            self.wandb.log(
                {
                    "roc_curve": self.wandb.plot.roc_curve(
                        y_true=y_true, y_probas=y_probas, labels=class_names
                    )
                }
            )

        logger.info("Logged ROC curve")

    def finish(self, success: bool = True):
        """
        Finish current run.

        Args:
            success: Whether run was successful
        """
        if not self.current_run:
            return

        if self.backend == "wandb" and self.wandb:
            self.wandb.finish(exit_code=0 if success else 1)

        elif self.backend == "mlflow" and self.mlflow:
            self.mlflow.end_run(status="FINISHED" if success else "FAILED")

        logger.info(f"Finished run: {'success' if success else 'failed'}")
        self.current_run = None

    def get_experiment_summary(self, experiment_name: str) -> Experiment | None:
        """
        Get experiment summary by name.

        Args:
            experiment_name: Name of experiment

        Returns:
            Experiment object or None
        """
        for exp in self.experiments:
            if exp.name == experiment_name:
                return exp
        return None

    def compare_experiments(self, experiment_names: list[str]) -> dict:
        """
        Compare multiple experiments.

        Args:
            experiment_names: List of experiment names

        Returns:
            Comparison dictionary
        """
        experiments = [exp for exp in self.experiments if exp.name in experiment_names]

        if not experiments:
            logger.warning("No experiments found for comparison")
            return {}

        comparison = {"experiments": [exp.name for exp in experiments], "metrics": {}}

        # Get all metric keys
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.metrics.keys())

        # Compare each metric
        for metric in all_metrics:
            comparison["metrics"][metric] = {
                exp.name: exp.metrics.get(metric, None) for exp in experiments
            }

        return comparison


# Helper function for easy tracking
def track_training(
    project: str = "stoic-citadel-ml", name: str | None = None, config: dict | None = None
):
    """
    Decorator to track training function.

    Usage:
        @track_training(name="my_model")
        def train_model(X, y):
            # training code
            return model, metrics
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = ExperimentTracker(project=project)

            experiment_name = name or func.__name__
            tracker.start_run(experiment_name, config=config or {})

            try:
                result = func(*args, **kwargs)

                # If function returns (model, metrics), log metrics
                if isinstance(result, tuple) and len(result) == 2:
                    _model, metrics = result
                    tracker.log_metrics(metrics)

                tracker.finish(success=True)
                return result

            except Exception as e:
                logger.error(f"Training failed: {e}")
                tracker.finish(success=False)
                raise

        return wrapper

    return decorator
