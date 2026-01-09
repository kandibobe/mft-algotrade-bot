"""
SHAP Explainability Utility
===========================

Provides model explainability using SHAP (SHapley Additive exPlanations).
Helps understand why the model made a specific prediction.

Usage:
    from src.ml.explainability import generate_shap_report
    generate_shap_report(model, X_train, X_test)
"""

import logging
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import log

logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Wrapper for SHAP explainability.
    """
    def __init__(self, model: Any, X_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        try:
            # Tree-based models (XGBoost, LightGBM, CatBoost, RandomForest)
            if hasattr(self.model, "predict_proba"): 
                 # General case, but TreeExplainer is faster for trees
                 # Here we try TreeExplainer first as most used models are trees
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception:
                    # Fallback to KernelExplainer (slow but generic)
                    # Use a sample of background data to speed it up
                    background = shap.sample(self.X_train, 100)
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            else:
                self.explainer = shap.Explainer(self.model)
                
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")

    def explain_local(self, X_instance: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Explain a single prediction (Local Interpretability).
        """
        if not self.explainer:
            return None
            
        try:
            shap_values = self.explainer(X_instance)
            
            # Handle classification (prob for class 0, 1)
            # shap_values.values shape might be (N, features, 2) for binary classifier
            vals = shap_values.values
            if len(vals.shape) == 3: # (N, features, classes)
                vals = vals[:, :, 1] # Take SHAP for class 1 (positive/buy)
            
            return pd.DataFrame(vals, columns=X_instance.columns)
        except Exception as e:
            logger.error(f"Error calculating local SHAP values: {e}")
            return None

    def plot_summary(self, X_test: pd.DataFrame, save_path: str = "user_data/plots/shap_summary.png"):
        """
        Generate global feature importance summary plot.
        """
        if not self.explainer:
            return

        try:
            shap_values = self.explainer(X_test)
            
            # Handle dimensionality for plot
            vals = shap_values.values
            if len(vals.shape) == 3:
                # Plot for positive class
                plt.figure(figsize=(12, 8))
                shap.summary_plot(vals[:, :, 1], X_test, show=False)
            else:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test, show=False)
            
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            log.info("shap_summary_plot_saved", path=save_path)
            
        except Exception as e:
            logger.error(f"Error generating SHAP summary plot: {e}")

def generate_shap_report(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, output_dir: str = "user_data/reports"):
    """
    Convenience function to generate full SHAP report.
    """
    explainer = ModelExplainer(model, X_train)
    
    # Global Summary
    explainer.plot_summary(X_test, save_path=f"{output_dir}/shap_summary.png")
    
    # Explain recent predictions
    recent_data = X_test.tail(10)
    shap_vals = explainer.explain_local(recent_data)
    
    if shap_vals is not None:
        shap_csv = f"{output_dir}/recent_shap_values.csv"
        shap_vals.to_csv(shap_csv)
        log.info("shap_values_saved", path=shap_csv)

if __name__ == "__main__":
    # Test stub
    print("SHAP Utility Module")
