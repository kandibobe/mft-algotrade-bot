"""
Online Learning (Model Retraining) Module
=========================================

Implements online learning capabilities for trading models with:
1. Continuous model updates on new data
2. Model drift detection and handling
3. A/B testing for new model versions
4. Gradual rollout of improved models

Problems addressed:
- Model doesn't update on new data
- Model drift over time
- No A/B testing for new models

Author: Stoic Citadel Team
License: MIT
"""

import logging
import pickle
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try to import river for streaming ML
try:
    from river import drift, linear_model, metrics, optimizers, losses

    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("River library not available. Install with: pip install river")

    # Define dummy classes for type hints
    class linear_model:
        class LogisticRegression:
            pass

    class metrics:
        class Accuracy:
            pass

    class drift:
        class ADWIN:
            pass

    class optimizers:
        class SGD:
            pass

    class losses:
        class Log:
            pass


# Scikit-learn for fallback
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""

    # Model paths
    base_model_path: str = "user_data/models/production_model.pkl"
    online_model_path: str = "user_data/models/online_model.pkl"

    # Online learning parameters
    learning_rate: float = 0.01
    use_river: bool = RIVER_AVAILABLE  # Use river if available
    fallback_to_sklearn: bool = True  # Fallback to sklearn if river not available

    # Model replacement criteria
    improvement_threshold: float = 0.05  # 5% better accuracy
    min_samples_for_comparison: int = 100
    confidence_level: float = 0.95  # Statistical confidence for replacement

    # A/B testing
    ab_test_traffic_pct: float = 0.1  # 10% traffic to online model
    ab_test_min_samples: int = 1000  # Minimum samples before evaluation

    # Drift detection
    enable_drift_detection: bool = True
    drift_detection_window: int = 1000
    drift_confidence: float = 0.99

    # Performance tracking
    metrics_window_size: int = 1000
    save_interval: int = 100  # Save model every N updates


def load_model(model_path: str) -> Any:
    """Load a model from disk."""
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


def save_model(model: Any, model_path: str) -> bool:
    """Save a model to disk."""
    try:
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        return False


class OnlineLearner:
    """
    Online Learning system for trading models.

    Features:
    1. Continuous online learning with new data
    2. Model drift detection using ADWIN or similar
    3. A/B testing between production and online models
    4. Gradual rollout of improved models
    5. Performance metrics tracking

    Usage:
        learner = OnlineLearner("user_data/models/prod_model.pkl")

        # Update with new data
        learner.update_online(X_new, y_true)

        # Get prediction (with A/B testing)
        prediction = learner.predict(X)

        # Check if online model should replace production
        if learner.should_replace_prod_model():
            learner.replace_production_model()
    """

    def __init__(self, base_model_path: str, config: Optional[OnlineLearningConfig] = None):
        """
        Initialize Online Learner.

        Args:
            base_model_path: Path to production model
            config: Configuration for online learning
        """
        self.config = config or OnlineLearningConfig()
        self.base_model_path = base_model_path

        # Load production model
        self.prod_model = load_model(base_model_path)
        if self.prod_model is None:
            logger.warning(f"Could not load production model from {base_model_path}")
            # Create a dummy model for testing
            self.prod_model = self._create_dummy_model()

        # Initialize online model
        self.online_model = self._initialize_online_model()

        # Initialize metrics
        self.prod_metrics = self._initialize_metrics()
        self.online_metrics = self._initialize_metrics()

        # Drift detection
        self.drift_detector = self._initialize_drift_detector()

        # Performance tracking
        self.prod_performance_history: List[float] = []
        self.online_performance_history: List[float] = []
        self.update_count = 0
        self.ab_test_samples = 0

        # A/B testing state
        self.ab_test_active = False
        self.ab_test_results: Dict[str, Any] = {}

        logger.info(f"Online Learner initialized with {'river' if RIVER_AVAILABLE else 'sklearn'}")

    def _create_dummy_model(self) -> Any:
        """Create a dummy model for testing if no production model exists."""
        logger.warning("Creating dummy model for testing")
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        # Fit on dummy data
        X_dummy = np.random.randn(10, 5)
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        return model

    def _initialize_online_model(self) -> Any:
        """Initialize online learning model."""
        if self.config.use_river and RIVER_AVAILABLE:
            # Use river for streaming ML
            try:
                model = linear_model.LogisticRegression(
                    optimizer=optimizers.SGD(learning_rate=self.config.learning_rate),
                    loss=losses.Log(),
                )
                logger.info("Initialized river online model")
            except Exception as e:
                logger.warning(f"Failed to initialize river model: {e}. Falling back to sklearn.")
                model = SGDClassifier(
                    loss="log_loss",
                    learning_rate="constant",
                    eta0=self.config.learning_rate,
                    random_state=42,
                )
                logger.info("Initialized sklearn online model (fallback)")
        else:
            # Use sklearn SGDClassifier for online learning
            model = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=self.config.learning_rate,
                random_state=42,
            )
            logger.info("Initialized sklearn online model (fallback)")

        return model

    def _initialize_metrics(self) -> Any:
        """Initialize metrics tracker."""
        if self.config.use_river and RIVER_AVAILABLE:
            return metrics.Accuracy()
        else:
            # Custom metrics tracker for sklearn
            class SklearnMetrics:
                def __init__(self):
                    self.correct = 0
                    self.total = 0
                    self.predictions = []
                    self.true_labels = []

                def update(self, y_true, y_pred):
                    self.predictions.append(y_pred)
                    self.true_labels.append(y_true)
                    self.total += 1
                    if y_pred == y_true:
                        self.correct += 1

                def get(self):
                    if self.total == 0:
                        return 0.0
                    return self.correct / self.total

                def reset(self):
                    self.correct = 0
                    self.total = 0
                    self.predictions = []
                    self.true_labels = []

            return SklearnMetrics()

    def _initialize_drift_detector(self) -> Optional[Any]:
        """Initialize drift detector."""
        if not self.config.enable_drift_detection:
            return None

        if self.config.use_river and RIVER_AVAILABLE:
            return drift.ADWIN(delta=self.config.drift_confidence)
        else:
            # Simple drift detector based on performance degradation
            class SimpleDriftDetector:
                def __init__(self, window_size=1000, threshold=0.05):
                    self.window_size = window_size
                    self.threshold = threshold
                    self.performance_history = []

                def update(self, performance: float):
                    self.performance_history.append(performance)
                    if len(self.performance_history) > self.window_size:
                        self.performance_history.pop(0)

                def detect_drift(self) -> bool:
                    if len(self.performance_history) < self.window_size:
                        return False

                    # Check if recent performance is significantly worse
                    recent = np.mean(self.performance_history[-self.window_size // 2 :])
                    older = np.mean(self.performance_history[: self.window_size // 2])

                    return (older - recent) > self.threshold

            return SimpleDriftDetector(
                window_size=self.config.drift_detection_window,
                threshold=self.config.improvement_threshold,
            )

    def update_online(self, X: Union[np.ndarray, Dict[str, float]], y_true: int):
        """
        Update online model with new data.

        Args:
            X: Feature vector (numpy array or dict for river)
            y_true: True label (0 or 1)
        """
        self.update_count += 1

        # Get predictions
        y_pred_prod = self._predict_prod(X)
        y_pred_online = self._predict_online(X)

        # Update metrics
        self.prod_metrics.update(y_true, y_pred_prod)
        self.online_metrics.update(y_true, y_pred_online)

        # Update online model
        self._learn_online(X, y_true)

        # Update drift detector
        if self.drift_detector is not None:
            if hasattr(self.drift_detector, "update"):
                # River-style drift detector
                self.drift_detector.update(y_pred_prod == y_true)
            else:
                # Custom drift detector
                self.drift_detector.update(self.prod_metrics.get())

        # Track performance history
        self.prod_performance_history.append(self.prod_metrics.get())
        self.online_performance_history.append(self.online_metrics.get())

        # Keep history within window
        if len(self.prod_performance_history) > self.config.metrics_window_size:
            self.prod_performance_history.pop(0)
            self.online_performance_history.pop(0)

        # Save model periodically
        if self.update_count % self.config.save_interval == 0:
            self.save_online_model()

        # Log progress
        if self.update_count % 100 == 0:
            logger.info(
                f"Update {self.update_count}: "
                f"Prod Acc={self.prod_metrics.get():.3f}, "
                f"Online Acc={self.online_metrics.get():.3f}"
            )

    def _predict_prod(self, X: Union[np.ndarray, Dict[str, float]]) -> int:
        """Predict using production model."""
        try:
            if isinstance(X, dict):
                # Convert dict to numpy array for sklearn
                X_array = np.array([list(X.values())])
            else:
                X_array = X.reshape(1, -1) if len(X.shape) == 1 else X

            if hasattr(self.prod_model, "predict"):
                return int(self.prod_model.predict(X_array)[0])
            else:
                # Assume it's a river model
                return int(self.prod_model.predict_one(X))
        except Exception as e:
            logger.error(f"Production model prediction error: {e}")
            return 0  # Conservative fallback

    def _predict_online(self, X: Union[np.ndarray, Dict[str, float]]) -> int:
        """Predict using online model."""
        try:
            if RIVER_AVAILABLE and self.config.use_river:
                # Check if model has predict_one method (river)
                if hasattr(self.online_model, 'predict_one'):
                    return int(self.online_model.predict_one(X))
                else:
                    # Fallback to sklearn predict
                    if isinstance(X, dict):
                        X_array = np.array([list(X.values())])
                    else:
                        X_array = X.reshape(1, -1) if len(X.shape) == 1 else X
                    return int(self.online_model.predict(X_array)[0])
            else:
                # sklearn model
                if isinstance(X, dict):
                    X_array = np.array([list(X.values())])
                else:
                    X_array = X.reshape(1, -1) if len(X.shape) == 1 else X

                return int(self.online_model.predict(X_array)[0])
        except Exception as e:
            logger.error(f"Online model prediction error: {e}")
            return 0  # Conservative fallback

    def _learn_online(self, X: Union[np.ndarray, Dict[str, float]], y_true: int):
        """Update online model with new sample."""
        try:
            if RIVER_AVAILABLE and self.config.use_river:
                # Check if model has learn_one method (river)
                if hasattr(self.online_model, 'learn_one'):
                    self.online_model.learn_one(X, y_true)
                else:
                    # Fallback to sklearn partial_fit
                    if isinstance(X, dict):
                        X_array = np.array([list(X.values())])
                    else:
                        X_array = X.reshape(1, -1) if len(X.shape) == 1 else X

                    y_array = np.array([y_true])

                    # Check if model has been fitted before
                    if not hasattr(self.online_model, "classes_"):
                        self.online_model.partial_fit(X_array, y_array, classes=[0, 1])
                    else:
                        self.online_model.partial_fit(X_array, y_array)
            else:
                # sklearn partial_fit
                if isinstance(X, dict):
                    X_array = np.array([list(X.values())])
                else:
                    X_array = X.reshape(1, -1) if len(X.shape) == 1 else X

                y_array = np.array([y_true])

                # Check if model has been fitted before
                if not hasattr(self.online_model, "classes_"):
                    self.online_model.partial_fit(X_array, y_array, classes=[0, 1])
                else:
                    self.online_model.partial_fit(X_array, y_array)
        except Exception as e:
            logger.error(f"Online learning error: {e}")

    def predict(self, X: Union[np.ndarray, Dict[str, float]], use_ab_test: bool = True) -> int:
        """
        Get prediction with optional A/B testing.

        Args:
            X: Feature vector
            use_ab_test: Whether to use A/B testing

        Returns:
            Prediction (0 or 1)
        """
        if use_ab_test and self.ab_test_active:
            self.ab_test_samples += 1
            return self.gradual_rollout(X)
        else:
            return self._predict_prod(X)

    def gradual_rollout(
        self, X: Union[np.ndarray, Dict[str, float]], traffic_pct: Optional[float] = None
    ) -> int:
        """
        A/B test: Route percentage of traffic to online model.

        Args:
            X: Feature vector
            traffic_pct: Percentage of traffic to route to online model (0-1)

        Returns:
            Prediction from either production or online model
        """
        if traffic_pct is None:
            traffic_pct = self.config.ab_test_traffic_pct

        if random.random() < traffic_pct:
            # Route to online model
            return self._predict_online(X)
        else:
            # Route to production model
            return self._predict_prod(X)

    def should_replace_prod_model(self) -> bool:
        """
        Decide if online model beats production model.

        Returns:
            True if online model should replace production model
        """
        # Need minimum samples for comparison
        if self.update_count < self.config.min_samples_for_comparison:
            return False

        prod_acc = self.prod_metrics.get()
        online_acc = self.online_metrics.get()

        # Check if online model is significantly better
        if online_acc > prod_acc + self.config.improvement_threshold:
            logger.info(
                f"Online model better: {online_acc:.3f} vs {prod_acc:.3f} "
                f"(improvement: {online_acc - prod_acc:.3f})"
            )
            return True

        # Check for model drift
        if self.drift_detector is not None:
            if hasattr(self.drift_detector, "drift_detected"):
                # River-style drift detector
                if self.drift_detector.drift_detected:
                    logger.info("Model drift detected")
                    return True
            else:
                # Custom drift detector
                if self.drift_detector.detect_drift():
                    logger.info("Model drift detected (performance degradation)")
                    return True

        return False

    def replace_production_model(self) -> bool:
        """
        Replace production model with online model.

        Returns:
            True if successful
        """
        try:
            # Save current production model as backup
            backup_path = self.base_model_path.replace(".pkl", f"_backup_{int(time.time())}.pkl")
            save_model(self.prod_model, backup_path)

            # Replace production model with online model
            self.prod_model = self.online_model

            # Save new production model
            success = save_model(self.prod_model, self.base_model_path)

            if success:
                # Reset online model for continued learning
                self.online_model = self._initialize_online_model()
                self.online_metrics = self._initialize_metrics()
                self.update_count = 0

                logger.info(
                    f"Production model replaced with online model. Backup saved to {backup_path}"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to replace production model: {e}")
            return False

    def start_ab_test(self, traffic_pct: Optional[float] = None) -> bool:
        """
        Start A/B test between production and online models.

        Args:
            traffic_pct: Percentage of traffic to route to online model

        Returns:
            True if A/B test started successfully
        """
        if traffic_pct is not None:
            self.config.ab_test_traffic_pct = traffic_pct

        self.ab_test_active = True
        self.ab_test_samples = 0
        self.ab_test_results = {
            "start_time": datetime.now(),
            "traffic_pct": self.config.ab_test_traffic_pct,
            "prod_wins": 0,
            "online_wins": 0,
            "ties": 0,
        }

        logger.info(
            f"A/B test started with {self.config.ab_test_traffic_pct*100:.1f}% traffic to online model"
        )
        return True

    def stop_ab_test(self) -> Dict[str, Any]:
        """
        Stop A/B test and return results.

        Returns:
            Dictionary with A/B test results
        """
        if not self.ab_test_active:
            return {"error": "no_active_ab_test"}

        self.ab_test_active = False
        self.ab_test_results["end_time"] = datetime.now()
        self.ab_test_results["duration"] = (
            self.ab_test_results["end_time"] - self.ab_test_results["start_time"]
        ).total_seconds()
        self.ab_test_results["total_samples"] = self.ab_test_samples

        # Calculate win rates
        total = (
            self.ab_test_results["prod_wins"]
            + self.ab_test_results["online_wins"]
            + self.ab_test_results["ties"]
        )
        if total > 0:
            self.ab_test_results["prod_win_rate"] = self.ab_test_results["prod_wins"] / total
            self.ab_test_results["online_win_rate"] = self.ab_test_results["online_wins"] / total
            self.ab_test_results["tie_rate"] = self.ab_test_results["ties"] / total
        else:
            self.ab_test_results["prod_win_rate"] = 0.0
            self.ab_test_results["online_win_rate"] = 0.0
            self.ab_test_results["tie_rate"] = 0.0

        logger.info(
            f"A/B test stopped. Results: "
            f"Prod wins: {self.ab_test_results['prod_wins']}, "
            f"Online wins: {self.ab_test_results['online_wins']}, "
            f"Ties: {self.ab_test_results['ties']}"
        )

        return self.ab_test_results

    def save_online_model(self) -> bool:
        """
        Save online model to disk.

        Returns:
            True if successful
        """
        return save_model(self.online_model, self.config.online_model_path)

    def load_online_model(self) -> bool:
        """
        Load online model from disk.

        Returns:
            True if successful
        """
        loaded_model = load_model(self.config.online_model_path)
        if loaded_model is not None:
            self.online_model = loaded_model
            logger.info(f"Online model loaded from {self.config.online_model_path}")
            return True
        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for both models.

        Returns:
            Dictionary with performance statistics
        """
        return {
            "production_model": {
                "accuracy": float(self.prod_metrics.get()),
                "update_count": self.update_count,
                "performance_history": (
                    self.prod_performance_history[-100:] if self.prod_performance_history else []
                ),
                "avg_performance": (
                    float(np.mean(self.prod_performance_history))
                    if self.prod_performance_history
                    else 0.0
                ),
            },
            "online_model": {
                "accuracy": float(self.online_metrics.get()),
                "update_count": self.update_count,
                "performance_history": (
                    self.online_performance_history[-100:]
                    if self.online_performance_history
                    else []
                ),
                "avg_performance": (
                    float(np.mean(self.online_performance_history))
                    if self.online_performance_history
                    else 0.0
                ),
            },
            "comparison": {
                "improvement": float(self.online_metrics.get() - self.prod_metrics.get()),
                "should_replace": self.should_replace_prod_model(),
                "drift_detected": self._check_drift_detected(),
            },
        }

    def _check_drift_detected(self) -> bool:
        """Check if drift has been detected."""
        if self.drift_detector is None:
            return False

        if hasattr(self.drift_detector, "drift_detected"):
            return self.drift_detector.drift_detected
        elif hasattr(self.drift_detector, "detect_drift"):
            return self.drift_detector.detect_drift()

        return False

    def reset_online_model(self) -> None:
        """Reset online model to initial state."""
        self.online_model = self._initialize_online_model()
        self.online_metrics = self._initialize_metrics()
        logger.info("Online model reset to initial state")

    def batch_update(self, X_batch: np.ndarray, y_batch: np.ndarray) -> None:
        """
        Update online model with a batch of data.

        Args:
            X_batch: Batch of features
            y_batch: Batch of labels
        """
        for X, y in zip(X_batch, y_batch):
            self.update_online(X, y)

    def evaluate_on_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Dict[str, float]:
        """
        Evaluate both models on a batch of data.

        Args:
            X_batch: Batch of features
            y_batch: Batch of labels

        Returns:
            Dictionary with evaluation metrics
        """
        prod_correct = 0
        online_correct = 0
        total = len(X_batch)

        for X, y in zip(X_batch, y_batch):
            prod_pred = self._predict_prod(X)
            online_pred = self._predict_online(X)

            if prod_pred == y:
                prod_correct += 1
            if online_pred == y:
                online_correct += 1

        return {
            "prod_accuracy": prod_correct / total if total > 0 else 0.0,
            "online_accuracy": online_correct / total if total > 0 else 0.0,
            "total_samples": total,
            "prod_correct": prod_correct,
            "online_correct": online_correct,
        }
