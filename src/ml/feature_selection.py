"""
SHAP Feature Selection
=======================

Implements automated feature selection using SHAP (SHapley Additive exPlanations).
Helps identify and remove noisy features that don't contribute to model performance.
"""

import logging
import pandas as pd
import numpy as np
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

class FeatureSelectorSHAP:
    """
    Automated feature selection using SHAP values.
    """

    def __init__(self, model: Any = None):
        self.model = model
        self.selected_features = []

    def fit_select(self, X: pd.DataFrame, y: pd.Series, top_n: int = 20) -> pd.DataFrame:
        """
        Analyze feature importance and select top N features.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Skipping automated selection.")
            return X

        logger.info(f"Running SHAP feature selection on {X.shape[1]} features...")
        
        try:
            # Use TreeExplainer for tree-based models (XGBoost, LightGBM)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            # For multiclass, shap_values is a list. Take mean absolute value across all classes.
            if isinstance(shap_values, list):
                importance = np.mean([np.abs(v).mean(0) for v in shap_values], axis=0)
            else:
                importance = np.abs(shap_values).mean(0)
                
            feature_importance = pd.Series(importance, index=X.columns)
            self.selected_features = feature_importance.sort_values(ascending=False).head(top_n).index.tolist()
            
            logger.info(f"Selected top {len(self.selected_features)} features using SHAP.")
            return X[self.selected_features]
            
        except Exception as e:
            logger.error(f"SHAP selection failed: {e}")
            return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply selection to new data."""
        if not self.selected_features:
            return X
        return X[[c for c in self.selected_features if c in X.columns]]
