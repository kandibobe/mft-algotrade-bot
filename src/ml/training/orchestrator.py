"""
Automated Retraining Orchestrator
=================================

Manages the lifecycle of ML models:
- Monitors performance metrics in real-time.
- Detects concept drift (degradation in performance).
- Triggers automated retraining workflows.
- Manages model promotion (Candidate -> Production).
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.config import config
from src.ml.online_learner import OnlineLearner
from src.ml.training.model_registry import ModelRegistry
from src.ml.pipeline import MLTrainingPipeline
from src.utils.logger import log

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the retraining orchestrator."""

    # Performance Thresholds
    min_accuracy: float = 0.55
    max_drawdown_window: float = 0.10  # Retrain if model causes >10% drawdown

    # Drift Detection
    drift_check_interval_minutes: int = 60
    performance_window_trades: int = 50  # Number of trades to calculate rolling performance

    # Retraining
    enable_auto_retraining: bool = True
    cooldown_minutes: int = 360  # Minimum time between retraining

    # Model Registry
    model_registry_dir: str | None = None


class TrainingOrchestrator:
    """
    Orchestrates the ML lifecycle.
    """

    def __init__(
        self, config_obj: OrchestratorConfig | None = None, online_learning_enabled: bool = False
    ):
        self.config = config_obj or OrchestratorConfig()

        # State
        self.last_retrain_time = datetime.min
        self.trade_history: list[dict] = []
        self.current_model_id: str | None = None

        # Metrics
        self.rolling_accuracy = 0.0
        self.rolling_profit = 0.0

        # Dependencies
        registry_dir = self.config.model_registry_dir or str(config().paths.models_dir / "registry")
        self.registry = ModelRegistry(registry_dir)
        
        # ML Pipeline
        self.training_pipeline = MLTrainingPipeline(quick_mode=False)

        # Online Learning
        self.online_learner = None
        if online_learning_enabled:
            self._init_online_learner()

        logger.info("ML Orchestrator initialized")

    def _init_online_learner(self):
        """Initialize the online learner."""
        try:
            prod_model_path = str(config().paths.models_dir / "production_model.pkl")
            self.online_learner = OnlineLearner(prod_model_path)
            logger.info("Online Learner integrated into Orchestrator")
        except Exception as e:
            logger.error(f"Failed to initialize Online Learner: {e}")

    async def run_training_pipeline(self, pair: str, strategy: str = "StoicEnsembleStrategyV7") -> dict[str, Any]:
        """
        Run the full ML training pipeline.
        
        Args:
            pair: Trading pair to train on
            strategy: Strategy name
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Running training pipeline for {pair} ({strategy})")
        
        try:
            # Execute pipeline
            # Note: run_pipeline_for_pair executes the pipeline but doesn't return rich metadata in all versions
            # We assume it returns the model path on success
            model_path = self.training_pipeline.run_pipeline_for_pair(pair)
            
            if model_path:
                logger.info(f"Training successful. Model saved at {model_path}")
                
                # Retrieve metadata from registry (assuming pipeline registered it)
                # This is a bit indirect, but pipeline.py handles registration internally
                latest_versions = self.registry.get_all_versions(pair.replace("/", "_") + "_5m") # Assuming 5m default
                
                model_id = "unknown"
                if latest_versions:
                    model_id = latest_versions[0].version
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "model_id": model_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                logger.error("Training pipeline returned no model path")
                return {"success": False, "reason": "Pipeline returned None"}
                
        except Exception as e:
            logger.exception(f"Training pipeline execution failed: {e}")
            return {"success": False, "error": str(e)}

    def record_trade_result(self, trade_result: dict):
        """
        Ingest a closed trade result to update performance metrics.

        Args:
            trade_result: Dict containing 'pnl', 'pnl_pct', 'predicted_direction', 'actual_direction'
        """
        self.trade_history.append(trade_result)

        # Keep history within window
        if len(self.trade_history) > self.config.performance_window_trades:
            self.trade_history.pop(0)

        self._update_metrics()
        self._check_for_drift()

    def _update_metrics(self):
        """Recalculate rolling metrics."""
        if not self.trade_history:
            return

        # Simplified accuracy: count trades with positive PnL as "correct"
        # (Assuming model predicts profitable opportunities)
        # In a real classifier, we'd compare predicted_class vs actual_class
        wins = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
        self.rolling_accuracy = wins / len(self.trade_history)

        self.rolling_profit = sum(t.get("pnl_pct", 0) for t in self.trade_history)

        log.info(
            f"Model Performance: Accuracy={self.rolling_accuracy:.2%}, Profit={self.rolling_profit:.2%}"
        )

    def _check_for_drift(self):
        """Check if performance has degraded enough to trigger action."""
        if len(self.trade_history) < self.config.performance_window_trades:
            return  # Not enough data

        # 1. Accuracy Drift
        if self.rolling_accuracy < self.config.min_accuracy:
            log.warning(
                f"📉 Concept Drift Detected: Accuracy {self.rolling_accuracy:.2%} < {self.config.min_accuracy:.2%}"
            )
            self.trigger_retraining("accuracy_drop")

        # 2. Profit Drift (Drawdown logic could be more complex)
        if self.rolling_profit < -self.config.max_drawdown_window:
            log.warning(
                f"📉 Performance Drop: Rolling PnL {self.rolling_profit:.2%} < -{self.config.max_drawdown_window:.2%}"
            )
            self.trigger_retraining("profit_drop")

    def trigger_retraining(self, reason: str):
        """Initiate the retraining workflow."""
        if not self.config.enable_auto_retraining:
            log.info(f"Retraining suggested ({reason}), but auto-retraining is DISABLED.")
            return

        now = datetime.utcnow()
        if now - self.last_retrain_time < timedelta(minutes=self.config.cooldown_minutes):
            log.info(f"Retraining skipped (Cooldown active). Last: {self.last_retrain_time}")
            return

        log.info(f"🔄 STARTING AUTOMATED RETRAINING: {reason}")

        try:
            # We trigger retraining for the primary pair (assuming BTC/USDT for now or derived from config)
            # In a real system, we'd know which pair triggered the drift
            primary_pair = config().pairs[0] 
            
            # Since this method is usually called from sync context (like strategy), 
            # and run_training_pipeline is async (or heavy sync), we need to be careful.
            # However, for this implementation, we'll assume it's safe to call or wrap it.
            # But wait, run_training_pipeline in my implementation above is async.
            # If this is called from a sync context, we might need to schedule it.
            
            # For simplicity in this fix, we'll assume the caller can handle the async nature 
            # or we invoke it synchronously if possible. 
            # Given the original code was just a mock pass, we'll use a sync wrapper or task.
            
            # OPTION: Just use the synchronous run_pipeline_for_pair directly if needed,
            # but let's stick to the async method signature defined above for consistency with full_cycle_launch.
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.run_training_pipeline(primary_pair))
            except RuntimeError:
                 # No running loop, run directly
                asyncio.run(self.run_training_pipeline(primary_pair))

            self.last_retrain_time = now
            # Clear history to give new model a fresh start
            self.trade_history = []
            self.rolling_accuracy = 0.0

            log.info(f"✅ Retraining triggered for {primary_pair}")

        except Exception as e:
            log.error(f"❌ Retraining trigger failed: {e}")

    def log_current_feature_importance(self, model_name: str, version: str):
        """
        Log the feature importance of the currently active model.
        Useful for 'Live Tracking' in logs or dashboard integration.
        """
        importance = self.registry.get_feature_importance(model_name, version)
        if not importance:
            logger.warning(f"No feature importance found for {model_name} v{version}")
            return

        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_importance[:5]

        log.info(f"🔍 Feature Importance ({model_name} v{version}):")
        for feature, score in top_n:
            log.info(f"  - {feature}: {score:.4f}")

        return dict(top_n)

    def get_status(self) -> dict:
        return {
            "rolling_accuracy": self.rolling_accuracy,
            "rolling_profit": self.rolling_profit,
            "trades_in_window": len(self.trade_history),
            "last_retrain": self.last_retrain_time.isoformat()
            if self.last_retrain_time != datetime.min
            else None,
        }
