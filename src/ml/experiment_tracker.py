"""
Stoic Citadel - Experiment Tracker
===================================

ML experiment tracking and logging:
- Training run logging
- Hyperparameter tracking
- Metric logging
- Artifact management
- W&B integration (optional)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import uuid

logger = logging.getLogger(__name__)

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class ExperimentRun:
    """Single experiment run metadata."""
    run_id: str
    experiment_name: str
    created_at: str
    
    # Configuration
    config: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: Dict[str, List[Dict]] = field(default_factory=dict)
    
    # Status
    status: str = "running"  # running, completed, failed
    finished_at: Optional[str] = None
    error: Optional[str] = None
    
    # Artifacts
    artifacts: List[str] = field(default_factory=list)
    
    # Notes
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentRun":
        return cls(**data)


class ExperimentTracker:
    """
    Experiment tracking for ML workflows.
    
    Supports:
    - Local file-based tracking
    - W&B integration (if available)
    - Metric logging with history
    - Artifact management
    """
    
    def __init__(
        self,
        project_name: str = "stoic-citadel",
        experiments_dir: str = "./experiments",
        use_wandb: bool = False,
        wandb_entity: Optional[str] = None
    ):
        self.project_name = project_name
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_entity = wandb_entity
        
        self._current_run: Optional[ExperimentRun] = None
        self._runs: Dict[str, ExperimentRun] = {}
        
        self._load_existing_runs()
    
    def start_run(
        self,
        experiment_name: str,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> str:
        """
        Start a new experiment run.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration/hyperparameters
            tags: Tags for categorization
            notes: Run notes
            
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())[:8]
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            created_at=datetime.utcnow().isoformat(),
            config=config or {},
            tags=tags or [],
            notes=notes
        )
        
        self._current_run = run
        self._runs[run_id] = run
        
        # Initialize W&B if enabled
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                entity=self.wandb_entity,
                name=f"{experiment_name}_{run_id}",
                config=config,
                tags=tags,
                notes=notes
            )
        
        # Create run directory
        run_dir = self.experiments_dir / experiment_name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Started experiment run: {experiment_name}/{run_id}")
        
        return run_id
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if self._current_run is None:
            logger.warning("No active run, metric not logged")
            return
        
        # Update current value
        self._current_run.metrics[name] = value
        
        # Add to history
        if name not in self._current_run.metric_history:
            self._current_run.metric_history[name] = []
        
        self._current_run.metric_history[name].append({
            "value": value,
            "step": step,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({name: value}, step=step)
        
        # Save periodically
        self._save_run(self._current_run)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log multiple metrics at once.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        """
        if self._current_run is None:
            return
        
        self._current_run.config.update(params)
        
        if self.use_wandb:
            wandb.config.update(params)
        
        self._save_run(self._current_run)
    
    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str = "model"
    ) -> None:
        """
        Log an artifact file.
        """
        if self._current_run is None:
            return
        
        self._current_run.artifacts.append(artifact_path)
        
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"{self._current_run.experiment_name}_{artifact_type}",
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
        
        self._save_run(self._current_run)
    
    def log_backtest_results(
        self,
        results: Dict[str, Any]
    ) -> None:
        """
        Log backtest results with standard metrics.
        """
        metrics_to_log = {
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "sortino_ratio": results.get("sortino_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "win_rate": results.get("win_rate", 0),
            "profit_factor": results.get("profit_factor", 0),
            "total_trades": results.get("total_trades", 0),
            "total_profit_pct": results.get("total_profit_pct", 0),
            "avg_trade_duration_hours": results.get("avg_trade_duration_hours", 0)
        }
        
        self.log_metrics(metrics_to_log)
        
        # Log full results as artifact
        if self._current_run:
            run_dir = self.experiments_dir / self._current_run.experiment_name / self._current_run.run_id
            results_path = run_dir / "backtest_results.json"
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.log_artifact(str(results_path), "backtest_results")
    
    def end_run(
        self,
        status: str = "completed",
        error: Optional[str] = None
    ) -> None:
        """
        End the current run.
        """
        if self._current_run is None:
            return
        
        self._current_run.status = status
        self._current_run.finished_at = datetime.utcnow().isoformat()
        if error:
            self._current_run.error = error
        
        self._save_run(self._current_run)
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info(
            f"Ended run {self._current_run.run_id} with status: {status}"
        )
        
        self._current_run = None
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get run by ID."""
        return self._runs.get(run_id)
    
    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[ExperimentRun]:
        """
        List experiment runs with optional filters.
        """
        runs = list(self._runs.values())
        
        if experiment_name:
            runs = [r for r in runs if r.experiment_name == experiment_name]
        
        if tags:
            runs = [r for r in runs if any(t in r.tags for t in tags)]
        
        if status:
            runs = [r for r in runs if r.status == status]
        
        # Sort by creation date
        runs.sort(key=lambda r: r.created_at, reverse=True)
        
        return runs[:limit]
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare multiple runs.
        """
        runs = [self._runs[rid] for rid in run_ids if rid in self._runs]
        
        if not metrics:
            # Get all metrics from first run
            metrics = list(runs[0].metrics.keys()) if runs else []
        
        comparison = {
            "runs": run_ids,
            "metrics": {}
        }
        
        for metric in metrics:
            comparison["metrics"][metric] = {
                run.run_id: run.metrics.get(metric, None)
                for run in runs
            }
        
        # Find best run per metric
        comparison["best_per_metric"] = {}
        for metric in metrics:
            values = comparison["metrics"][metric]
            valid_values = {k: v for k, v in values.items() if v is not None}
            if valid_values:
                # Maximize most metrics, minimize drawdown
                if "drawdown" in metric.lower():
                    best_run = min(valid_values, key=valid_values.get)
                else:
                    best_run = max(valid_values, key=valid_values.get)
                comparison["best_per_metric"][metric] = best_run
        
        return comparison
    
    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "sharpe_ratio",
        minimize: bool = False
    ) -> Optional[ExperimentRun]:
        """
        Get the best run for an experiment based on a metric.
        """
        runs = self.list_runs(
            experiment_name=experiment_name,
            status="completed"
        )
        
        if not runs:
            return None
        
        # Filter runs with the metric
        runs_with_metric = [r for r in runs if metric in r.metrics]
        
        if not runs_with_metric:
            return None
        
        if minimize:
            return min(runs_with_metric, key=lambda r: r.metrics[metric])
        else:
            return max(runs_with_metric, key=lambda r: r.metrics[metric])
    
    def delete_run(self, run_id: str) -> None:
        """Delete a run."""
        if run_id not in self._runs:
            return
        
        run = self._runs[run_id]
        
        # Remove files
        run_dir = self.experiments_dir / run.experiment_name / run_id
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
        
        del self._runs[run_id]
        logger.info(f"Deleted run: {run_id}")
    
    def _save_run(self, run: ExperimentRun) -> None:
        """Save run to disk."""
        run_dir = self.experiments_dir / run.experiment_name / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        run_file = run_dir / "run.json"
        with open(run_file, 'w') as f:
            json.dump(run.to_dict(), f, indent=2)
    
    def _load_existing_runs(self) -> None:
        """Load existing runs from disk."""
        for experiment_dir in self.experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            
            for run_dir in experiment_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                run_file = run_dir / "run.json"
                if run_file.exists():
                    try:
                        with open(run_file) as f:
                            data = json.load(f)
                        run = ExperimentRun.from_dict(data)
                        self._runs[run.run_id] = run
                    except Exception as e:
                        logger.error(f"Error loading run {run_dir}: {e}")


# Convenience functions for simpler usage
_tracker: Optional[ExperimentTracker] = None


def init(
    project_name: str = "stoic-citadel",
    **kwargs
) -> ExperimentTracker:
    """Initialize global tracker."""
    global _tracker
    _tracker = ExperimentTracker(project_name=project_name, **kwargs)
    return _tracker


def start_run(
    experiment_name: str,
    **kwargs
) -> str:
    """Start a run using global tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker.start_run(experiment_name, **kwargs)


def log_metric(name: str, value: float, step: Optional[int] = None) -> None:
    """Log metric using global tracker."""
    if _tracker:
        _tracker.log_metric(name, value, step)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics using global tracker."""
    if _tracker:
        _tracker.log_metrics(metrics, step)


def log_params(params: Dict[str, Any]) -> None:
    """Log params using global tracker."""
    if _tracker:
        _tracker.log_params(params)


def end_run(status: str = "completed", error: Optional[str] = None) -> None:
    """End run using global tracker."""
    if _tracker:
        _tracker.end_run(status, error)
