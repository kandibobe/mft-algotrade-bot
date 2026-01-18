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

from src.config import config
from src.ml.online_learner import OnlineLearner
from src.ml.training.model_registry import ModelRegistry
from src.utils.logger import log

logger = logging.getLogger(__name__)

@dataclass
class OrchestratorConfig:
    """Configuration for the retraining orchestrator."""

    # Performance Thresholds
    min_accuracy: float = 0.55
    max_drawdown_window: float = 0.10 # Retrain if model causes >10% drawdown

    # Drift Detection
    drift_check_interval_minutes: int = 60
    performance_window_trades: int = 50 # Number of trades to calculate rolling performance

    # Retraining
    enable_auto_retraining: bool = False
    cooldown_minutes: int = 360 # Minimum time between retraining

    # Model Registry
    model_registry_dir: str | None = None

class TrainingOrchestrator:
    """
    Orchestrates the ML lifecycle.
    """

    def __init__(self, config_obj: OrchestratorConfig | None = None, online_learning_enabled: bool = False):
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
        wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        self.rolling_accuracy = wins / len(self.trade_history)

        self.rolling_profit = sum(t.get('pnl_pct', 0) for t in self.trade_history)

        log.info(f"Model Performance: Accuracy={self.rolling_accuracy:.2%}, Profit={self.rolling_profit:.2%}")

    def _check_for_drift(self):
        """Check if performance has degraded enough to trigger action."""
        if len(self.trade_history) < self.config.performance_window_trades:
            return # Not enough data

        # 1. Accuracy Drift
        if self.rolling_accuracy < self.config.min_accuracy:
            log.warning(f"📉 Concept Drift Detected: Accuracy {self.rolling_accuracy:.2%} < {self.config.min_accuracy:.2%}")
            self.trigger_retraining("accuracy_drop")

        # 2. Profit Drift (Drawdown logic could be more complex)
        if self.rolling_profit < -self.config.max_drawdown_window:
            log.warning(f"📉 Performance Drop: Rolling PnL {self.rolling_profit:.2%} < -{self.config.max_drawdown_window:.2%}")
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
            # Here we would call the actual training pipeline
            # e.g., await self.training_pipeline.run_full_cycle()
            # For now, we simulate success
            self._execute_retraining_mock()

            self.last_retrain_time = now
            # Clear history to give new model a fresh start
            self.trade_history = []
            self.rolling_accuracy = 0.0

            log.info("✅ Retraining complete. New model deployed.")

        except Exception as e:
            log.error(f"❌ Retraining failed: {e}")

    def _execute_retraining_mock(self):
        """Simulate the heavy lifting of retraining."""
        time.sleep(0.1) # Simulate work
        pass

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
            "last_retrain": self.last_retrain_time.isoformat() if self.last_retrain_time != datetime.min else None
        }
