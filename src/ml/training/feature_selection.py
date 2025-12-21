"""
Feature Selection Module
=========================

Advanced feature selection using SHAP values, Permutation Importance,
and statistical methods to eliminate noise and prevent overfitting.

Key insight: More features != better model.
Garbage in = Garbage out. Remove weak and correlated features.

Methods:
1. SHAP Values - Explains each feature's contribution
2. Permutation Importance - Measures impact of shuffling each feature
3. Correlation Filter - Removes redundant features
4. Noise Threshold - Removes features weaker than random noise
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""

    # Method: 'shap', 'permutation', 'both'
    method: str = "permutation"

    # Correlation threshold for removing redundant features
    correlation_threshold: float = 0.95

    # Minimum importance threshold (relative to noise)
    # Features with importance < noise_multiplier * random_feature_importance are removed
    noise_multiplier: float = 1.5

    # Maximum number of features to keep (None = no limit)
    max_features: Optional[int] = None

    # Minimum number of features to keep
    min_features: int = 5

    # Cross-validation folds for stability
    cv_folds: int = 5

    # Random state for reproducibility
    random_state: int = 42

    # Output path for selected features
    output_path: str = "user_data/models/selected_features.json"


class FeatureSelector:
    """
    Intelligent feature selection to prevent overfitting.

    Problem: Throwing all indicators (RSI, MACD, CCI, Bollinger...) into
    the model leads to overfitting on noise.

    Solution: Use SHAP/Permutation Importance to keep only features
    that actually help prediction.

    Usage:
        selector = FeatureSelector(config)
        X_filtered, selected_features = selector.fit_transform(X, y, model)
        selector.save_selected_features()
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """Initialize feature selector."""
        self.config = config or FeatureSelectionConfig()
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.dropped_features: Dict[str, str] = {}  # feature -> reason
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, model: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fit feature selector and transform data.

        Args:
            X: Feature DataFrame
            y: Target Series
            model: Optional trained model (will train one if not provided)

        Returns:
            (filtered_X, selected_features)
        """
        logger.info(f"Starting feature selection on {X.shape[1]} features")

        # Step 1: Remove highly correlated features
        X_uncorrelated, corr_dropped = self._remove_correlated(X)
        for feat in corr_dropped:
            self.dropped_features[feat] = "high_correlation"

        logger.info(f"After correlation filter: {X_uncorrelated.shape[1]} features")

        # Step 2: Calculate feature importance
        if model is None:
            model = self._create_default_model()
            model.fit(X_uncorrelated, y)

        if self.config.method in ["shap", "both"]:
            shap_importance = self._calculate_shap_importance(X_uncorrelated, model)
        else:
            shap_importance = {}

        if self.config.method in ["permutation", "both"]:
            perm_importance = self._calculate_permutation_importance(X_uncorrelated, y, model)
        else:
            perm_importance = {}

        # Combine importances
        self.feature_importance = self._combine_importances(shap_importance, perm_importance)

        # Step 3: Add noise feature and filter
        X_with_noise, noise_importance = self._add_noise_feature(X_uncorrelated, y, model)

        # Step 4: Filter features below noise threshold
        noise_threshold = noise_importance * self.config.noise_multiplier
        logger.info(f"Noise threshold: {noise_threshold:.4f}")

        filtered_features = []
        for feat, importance in self.feature_importance.items():
            if importance >= noise_threshold:
                filtered_features.append((feat, importance))
            else:
                self.dropped_features[feat] = (
                    f"below_noise_threshold ({importance:.4f} < {noise_threshold:.4f})"
                )

        # Sort by importance
        filtered_features.sort(key=lambda x: x[1], reverse=True)

        # Step 5: Apply max/min feature limits
        if self.config.max_features:
            filtered_features = filtered_features[: self.config.max_features]

        if len(filtered_features) < self.config.min_features:
            # Keep top min_features regardless of threshold
            all_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            filtered_features = all_features[: self.config.min_features]
            logger.warning(
                f"Only {len(filtered_features)} features above threshold, "
                f"keeping top {self.config.min_features}"
            )

        self.selected_features = [f[0] for f in filtered_features]

        logger.info(f"Selected {len(self.selected_features)} features")
        logger.info(f"Top 10 features: {self.selected_features[:10]}")

        return X[self.selected_features], self.selected_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using previously selected features.

        Args:
            X: Feature DataFrame

        Returns:
            Filtered DataFrame with only selected features
        """
        if not self.selected_features:
            raise ValueError("Must call fit_transform() first")

        missing = set(self.selected_features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features in input: {missing}")

        return X[self.selected_features]

    def _remove_correlated(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        # Calculate correlation matrix
        self.correlation_matrix = X.corr().abs()

        # Find pairs above threshold
        upper_tri = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )

        # Drop columns with high correlation
        to_drop = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > self.config.correlation_threshold):
                to_drop.append(column)

        logger.info(f"Removing {len(to_drop)} correlated features")

        return X.drop(columns=to_drop), to_drop

    def _create_default_model(self):
        """Create default model for importance calculation."""
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=self.config.random_state, n_jobs=-1
        )

    def _calculate_shap_importance(self, X: pd.DataFrame, model: Any) -> Dict[str, float]:
        """Calculate SHAP-based feature importance."""
        try:
            import shap

            logger.info("Calculating SHAP values...")

            # Use TreeExplainer for tree-based models
            if hasattr(model, "estimators_"):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X)

            # Calculate SHAP values (sample if too large)
            sample_size = min(1000, len(X))
            X_sample = X.sample(sample_size, random_state=self.config.random_state)

            shap_values = explainer.shap_values(X_sample)

            # Handle multi-class (use absolute mean)
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)

            # Calculate mean absolute SHAP value per feature
            importance = np.abs(shap_values).mean(axis=0)

            return dict(zip(X.columns, importance))

        except ImportError:
            logger.warning("SHAP not installed, skipping SHAP importance")
            return {}
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            return {}

    def _calculate_permutation_importance(
        self, X: pd.DataFrame, y: pd.Series, model: Any
    ) -> Dict[str, float]:
        """Calculate permutation-based feature importance."""
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import cross_val_score

        logger.info("Calculating permutation importance...")

        try:
            # Use cross-validated permutation importance for stability
            result = permutation_importance(
                model,
                X,
                y,
                n_repeats=10,
                random_state=self.config.random_state,
                n_jobs=-1,
                scoring="accuracy",
            )

            importance = result.importances_mean

            return dict(zip(X.columns, importance))

        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            # Fall back to model's built-in importance
            if hasattr(model, "feature_importances_"):
                return dict(zip(X.columns, model.feature_importances_))
            return {}

    def _combine_importances(
        self, shap_importance: Dict[str, float], perm_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine SHAP and permutation importance scores."""
        all_features = set(shap_importance.keys()) | set(perm_importance.keys())

        combined = {}
        for feat in all_features:
            shap_val = shap_importance.get(feat, 0)
            perm_val = perm_importance.get(feat, 0)

            # Normalize each to 0-1 range
            if shap_importance:
                max_shap = max(shap_importance.values()) or 1
                shap_val = shap_val / max_shap

            if perm_importance:
                max_perm = max(perm_importance.values()) or 1
                perm_val = perm_val / max_perm

            # Average (or take one if other is missing)
            if shap_importance and perm_importance:
                combined[feat] = (shap_val + perm_val) / 2
            elif shap_importance:
                combined[feat] = shap_val
            else:
                combined[feat] = perm_val

        return combined

    def _add_noise_feature(
        self, X: pd.DataFrame, y: pd.Series, model: Any
    ) -> Tuple[pd.DataFrame, float]:
        """
        Add random noise feature to establish importance baseline.

        Features less important than random noise should be removed.
        """
        np.random.seed(self.config.random_state)

        # Add random feature
        X_with_noise = X.copy()
        X_with_noise["_random_noise_"] = np.random.randn(len(X))

        # Retrain model
        model_with_noise = self._create_default_model()
        model_with_noise.fit(X_with_noise, y)

        # Get noise feature importance
        if hasattr(model_with_noise, "feature_importances_"):
            noise_idx = list(X_with_noise.columns).index("_random_noise_")
            noise_importance = model_with_noise.feature_importances_[noise_idx]
        else:
            noise_importance = 0.01  # Default threshold

        logger.info(f"Random noise feature importance: {noise_importance:.4f}")

        return X_with_noise.drop(columns=["_random_noise_"]), noise_importance

    def save_selected_features(self, path: Optional[str] = None):
        """
        Save selected features to JSON file.

        Args:
            path: Output path (uses config.output_path if not specified)
        """
        output_path = Path(path or self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "selected_features": self.selected_features,
            "feature_importance": self.feature_importance,
            "dropped_features": self.dropped_features,
            "config": {
                "method": self.config.method,
                "correlation_threshold": self.config.correlation_threshold,
                "noise_multiplier": self.config.noise_multiplier,
                "max_features": self.config.max_features,
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Selected features saved to {output_path}")

    def load_selected_features(self, path: Optional[str] = None) -> List[str]:
        """
        Load selected features from JSON file.

        Args:
            path: Input path (uses config.output_path if not specified)

        Returns:
            List of selected feature names
        """
        input_path = Path(path or self.config.output_path)

        with open(input_path, "r") as f:
            data = json.load(f)

        self.selected_features = data["selected_features"]
        self.feature_importance = data.get("feature_importance", {})
        self.dropped_features = data.get("dropped_features", {})

        logger.info(f"Loaded {len(self.selected_features)} features from {input_path}")

        return self.selected_features

    def get_feature_report(self) -> str:
        """Generate human-readable feature selection report."""
        report = []
        report.append("=" * 60)
        report.append("FEATURE SELECTION REPORT")
        report.append("=" * 60)
        report.append(f"\nMethod: {self.config.method}")
        report.append(f"Correlation threshold: {self.config.correlation_threshold}")
        report.append(f"Noise multiplier: {self.config.noise_multiplier}")

        report.append(f"\n📊 SELECTED FEATURES ({len(self.selected_features)}):")
        report.append("-" * 40)
        for i, feat in enumerate(self.selected_features[:20], 1):
            importance = self.feature_importance.get(feat, 0)
            report.append(f"  {i:2d}. {feat:30s} (importance: {importance:.4f})")

        if len(self.selected_features) > 20:
            report.append(f"  ... and {len(self.selected_features) - 20} more")

        report.append(f"\n🗑️ DROPPED FEATURES ({len(self.dropped_features)}):")
        report.append("-" * 40)
        for feat, reason in list(self.dropped_features.items())[:10]:
            report.append(f"  - {feat}: {reason}")

        if len(self.dropped_features) > 10:
            report.append(f"  ... and {len(self.dropped_features) - 10} more")

        return "\n".join(report)


class RecursiveFeatureEliminator:
    """
    Recursive Feature Elimination with Cross-Validation.

    Starts with all features and recursively removes the least important
    until optimal number is found.
    """

    def __init__(
        self,
        min_features: int = 5,
        step: int = 1,
        cv_folds: int = 5,
        scoring: str = "accuracy",
        random_state: int = 42,
    ):
        """Initialize RFE."""
        self.min_features = min_features
        self.step = step
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.selected_features: List[str] = []
        self.cv_scores: Dict[int, float] = {}

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, model: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform RFECV to find optimal features.

        Args:
            X: Feature DataFrame
            y: Target Series
            model: Optional model to use

        Returns:
            (filtered_X, selected_features)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFECV

        logger.info("Starting Recursive Feature Elimination with CV...")

        if model is None:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1
            )

        rfecv = RFECV(
            estimator=model,
            step=self.step,
            cv=self.cv_folds,
            scoring=self.scoring,
            min_features_to_select=self.min_features,
            n_jobs=-1,
        )

        rfecv.fit(X, y)

        self.selected_features = list(X.columns[rfecv.support_])
        self.cv_scores = dict(enumerate(rfecv.cv_results_["mean_test_score"], 1))

        logger.info(f"Optimal number of features: {rfecv.n_features_}")
        logger.info(f"Selected features: {self.selected_features}")

        return X[self.selected_features], self.selected_features
