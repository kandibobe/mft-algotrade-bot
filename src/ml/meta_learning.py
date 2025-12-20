"""
Meta-Learning Ensemble for Trading Strategies
=============================================

Implements meta-learning ensemble that learns optimal weights for base models
using Logistic Regression as meta-learner.

Features:
1. Learns optimal weights for base models based on historical performance
2. Provides confidence scores based on model agreement
3. Supports online learning and weight updates
4. Handles missing predictions gracefully

Author: Stoic Citadel Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning ensemble."""
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    meta_model_path: str = "user_data/models/meta_ensemble.pkl"
    retrain_interval: int = 1000  # Number of predictions before retraining
    min_samples_for_training: int = 100
    use_probabilities: bool = True  # Use predict_proba instead of predict


class MetaLearningEnsemble:
    """
    Meta-Learning Ensemble that learns optimal weights for base models.
    
    Architecture:
    1. Base models make predictions (probabilities)
    2. Meta-learner (LogisticRegression) learns to combine these predictions
    3. Final prediction = weighted combination of base model predictions
    4. Confidence = agreement between base models
    
    Usage:
        ensemble = MetaLearningEnsemble(base_models)
        ensemble.train_meta_model(X_train, y_train, X_val, y_val)
        predictions, confidence = ensemble.predict_with_confidence(X_new)
    """
    
    def __init__(
        self,
        base_models: List[Any],
        config: Optional[MetaLearningConfig] = None
    ):
        """
        Initialize meta-learning ensemble.
        
        Args:
            base_models: List of base model instances with predict_proba method
            config: Configuration for meta-learning
        """
        self.base_models = base_models
        self.config = config or MetaLearningConfig()
        self.meta_model = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        self.is_trained = False
        self.training_history: List[Dict] = []
        self.prediction_count = 0
        
        # Validate base models
        for i, model in enumerate(self.base_models):
            if not hasattr(model, 'predict_proba'):
                logger.warning(f"Base model {i} doesn't have predict_proba method")
    
    def train_meta_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train meta-model on base model predictions.
        
        Args:
            X_train: Training features for base models
            y_train: Training labels (1 for buy, 0 for sell/hold)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        if len(X_train) < self.config.min_samples_for_training:
            logger.warning(
                f"Insufficient training samples: {len(X_train)} < "
                f"{self.config.min_samples_for_training}"
            )
            return {"error": "insufficient_samples"}
        
        # Get base model predictions on training data
        logger.info(f"Training meta-model with {len(X_train)} samples")
        
        base_preds_train = self._get_base_predictions(X_train)
        
        # Train meta-model
        self.meta_model.fit(base_preds_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_metrics = self._evaluate_metrics(base_preds_train, y_train, "train")
        
        # Validation metrics if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            base_preds_val = self._get_base_predictions(X_val)
            val_metrics = self._evaluate_metrics(base_preds_val, y_val, "val")
        
        # Combine metrics
        metrics = {
            **train_metrics,
            **val_metrics,
            "base_model_count": len(self.base_models),
            "training_samples": len(X_train),
            "meta_model_coef": self.meta_model.coef_.tolist() if hasattr(self.meta_model, 'coef_') else []
        }
        
        # Save training history
        self.training_history.append({
            "timestamp": pd.Timestamp.now(),
            "metrics": metrics
        })
        
        logger.info(f"Meta-model trained. Training accuracy: {metrics.get('train_accuracy', 0):.3f}")
        
        return metrics
    
    def train_with_validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Train meta-model with automatic train/val/test split.
        
        Args:
            X: Features for base models
            y: Labels
            test_size: Size of test split (if None, uses config)
            
        Returns:
            Dictionary with training metrics
        """
        if test_size is None:
            test_size = self.config.test_size
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        val_size = self.config.val_size / (self.config.val_size + self.config.test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        # Train on train set
        metrics = self.train_meta_model(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        if len(X_test) > 0:
            test_preds, test_conf = self.predict_with_confidence(X_test)
            test_accuracy = accuracy_score(y_test, test_preds > 0.5)
            test_auc = roc_auc_score(y_test, test_preds) if len(np.unique(y_test)) > 1 else 0.5
            
            metrics.update({
                "test_accuracy": float(test_accuracy),
                "test_auc": float(test_auc),
                "test_samples": len(X_test)
            })
        
        return metrics
    
    def predict_with_confidence(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence score.
        
        Args:
            X: Features for base models
            
        Returns:
            Tuple of (predictions, confidence_scores)
            - predictions: Final meta-model predictions (probabilities)
            - confidence: Agreement between base models (0-1)
        """
        if not self.is_trained:
            logger.warning("Meta-model not trained. Using equal weights.")
            return self._predict_with_equal_weights(X)
        
        # Get base model predictions
        base_preds = self._get_base_predictions(X)
        
        # Meta-model prediction
        if self.config.use_probabilities and hasattr(self.meta_model, 'predict_proba'):
            final_pred = self.meta_model.predict_proba(base_preds)[:, 1]
        else:
            final_pred = self.meta_model.predict(base_preds)
        
        # Confidence = agreement between base models
        confidence = self._calculate_confidence(base_preds)
        
        # Update prediction count
        self.prediction_count += len(X)
        
        # Check if retraining is needed
        if self.prediction_count >= self.config.retrain_interval:
            logger.info(f"Prediction count {self.prediction_count} reached retraining interval")
            # In production, you would trigger retraining here
            self.prediction_count = 0
        
        return final_pred, confidence
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict without confidence scores.
        
        Args:
            X: Features for base models
            
        Returns:
            Final predictions (probabilities)
        """
        predictions, _ = self.predict_with_confidence(X)
        return predictions
    
    def get_model_weights(self) -> np.ndarray:
        """
        Get current weights of base models from meta-learner.
        
        Returns:
            Array of weights for each base model
        """
        if not self.is_trained or not hasattr(self.meta_model, 'coef_'):
            # Return equal weights if not trained
            return np.ones(len(self.base_models)) / len(self.base_models)
        
        # LogisticRegression coefficients (absolute values normalized)
        weights = np.abs(self.meta_model.coef_[0])
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save meta-model to disk.
        
        Args:
            path: Path to save model (if None, uses config.meta_model_path)
            
        Returns:
            Path where model was saved
        """
        if path is None:
            path = self.config.meta_model_path
        
        # Create directory if it doesn't exist
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'meta_model': self.meta_model,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'prediction_count': self.prediction_count
            }, f)
        
        logger.info(f"Meta-model saved to {path}")
        return path
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load meta-model from disk.
        
        Args:
            path: Path to load model from (if None, uses config.meta_model_path)
            
        Returns:
            True if successful, False otherwise
        """
        if path is None:
            path = self.config.meta_model_path
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.meta_model = data['meta_model']
            self.config = data.get('config', self.config)
            self.is_trained = data.get('is_trained', False)
            self.training_history = data.get('training_history', [])
            self.prediction_count = data.get('prediction_count', 0)
            
            logger.info(f"Meta-model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load meta-model from {path}: {e}")
            return False
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all base models.
        
        Args:
            X: Features for base models
            
        Returns:
            2D array where each column is a base model's predictions
        """
        base_preds = []
        
        for i, model in enumerate(self.base_models):
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]  # Probability of positive class
                else:
                    pred = model.predict(X)
                base_preds.append(pred)
            except Exception as e:
                logger.error(f"Base model {i} prediction error: {e}")
                # Use neutral predictions for failed models
                base_preds.append(np.full(len(X), 0.5))
        
        return np.column_stack(base_preds)
    
    def _calculate_confidence(self, base_preds: np.ndarray) -> np.ndarray:
        """
        Calculate confidence based on agreement between base models.
        
        Args:
            base_preds: Base model predictions (2D array)
            
        Returns:
            Confidence scores (0-1)
        """
        if base_preds.shape[1] == 0:
            return np.zeros(base_preds.shape[0])
        
        # Method 1: Standard deviation (lower std = higher agreement)
        std_dev = np.std(base_preds, axis=1)
        mean_pred = np.mean(base_preds, axis=1)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.where(mean_pred > 0, std_dev / mean_pred, 0)
        
        # Confidence = 1 - normalized coefficient of variation
        confidence = 1 - np.clip(cv, 0, 1)
        
        # Method 2: Percentage of models agreeing on direction
        # (predictions > 0.5)
        if base_preds.shape[1] > 1:
            buy_votes = np.sum(base_preds > 0.5, axis=1)
            sell_votes = np.sum(base_preds < 0.5, axis=1)
            total_votes = buy_votes + sell_votes
            
            with np.errstate(divide='ignore', invalid='ignore'):
                agreement = np.where(total_votes > 0, 
                                   np.maximum(buy_votes, sell_votes) / total_votes,
                                   0.5)
            
            # Combine both confidence measures
            confidence = 0.7 * confidence + 0.3 * agreement
        
        return np.clip(confidence, 0, 1)
    
    def _predict_with_equal_weights(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using equal weights (fallback when meta-model not trained).
        
        Args:
            X: Features for base models
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        base_preds = self._get_base_predictions(X)
        
        # Equal weighted average
        final_pred = np.mean(base_preds, axis=1)
        
        # Confidence
        confidence = self._calculate_confidence(base_preds)
        
        return final_pred, confidence
    
    def _evaluate_metrics(
        self,
        base_preds: np.ndarray,
        y_true: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Evaluate model performance metrics.
        
        Args:
            base_preds: Base model predictions
            y_true: True labels
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            return {}
        
        # Get meta-model predictions
        if self.config.use_probabilities and hasattr(self.meta_model, 'predict_proba'):
            y_pred_proba = self.meta_model.predict_proba(base_preds)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.meta_model.predict(base_preds)
            y_pred_proba = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.5
        
        # Base model accuracies
        base_accuracies = []
        for i in range(base_preds.shape[1]):
            base_pred = (base_preds[:, i] > 0.5).astype(int)
            base_acc = accuracy_score(y_true, base_pred)
            base_accuracies.append(base_acc)
        
        # Create metric dictionary
        metrics = {
            f"{prefix}_accuracy": float(accuracy),
            f"{prefix}_auc": float(auc),
            f"{prefix}_base_accuracies": [float(acc) for acc in base_accuracies],
            f"{prefix}_base_avg_accuracy": float(np.mean(base_accuracies))
        }
        
        return metrics
    
    def update_with_feedback(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        incremental: bool = True
    ) -> Dict[str, Any]:
        """
        Update meta-model with new feedback data.
        
        Args:
            X: New features
            y_true: New labels
            incremental: If True, use partial_fit (if available)
            
        Returns:
            Dictionary with update metrics
        """
        if len(X) == 0:
            return {"error": "no_data"}
        
        # Get base model predictions
        base_preds = self._get_base_predictions(X)
        
        # Check if meta-model supports incremental learning
        if incremental and hasattr(self.meta_model, 'partial_fit'):
            try:
                self.meta_model.partial_fit(base_preds, y_true)
                update_method = "incremental"
            except Exception as e:
                logger.warning(f"Incremental learning failed: {e}. Using full retrain.")
                incremental = False
        
        # Full retrain if incremental not available or failed
        if not incremental:
            # Combine with existing training data (if available)
            # In production, you would maintain a rolling window of recent data
            # For simplicity, we just retrain on new data
            if len(X) >= self.config.min_samples_for_training:
                self.meta_model.fit(base_preds, y_true)
                update_method = "full_retrain"
            else:
                logger.warning("Insufficient data for full retrain")
                return {"error": "insufficient_data"}
        
        # Evaluate on the new data
        metrics = self._evaluate_metrics(base_preds, y_true, "update")
        
        # Update training history
        self.training_history.append({
            "timestamp": pd.Timestamp.now(),
            "metrics": metrics,
            "update_method": update_method,
            "samples": len(X)
        })
        
        logger.info(f"Meta-model updated with {len(X)} samples using {update_method}")
        
        return {
            **metrics,
            "update_method": update_method,
            "samples": len(X),
            "model_weights": self.get_model_weights().tolist()
        }
