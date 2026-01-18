#!/usr/bin/env python3
"""
Weekly Feature Importance Analysis (SHAP)
=========================================

Calculates SHAP values for the production model and generates a report.
Run this weekly to identify and remove noisy features.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ml.model_loader import ModelLoader
from src.ml.explainability import generate_shap_report
from src.data.loader import DataLoader
from src.config.unified_config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger("shap_analysis")

def run_weekly_analysis():
    """
    Main entry point for weekly feature analysis.
    """
    setup_logging(level="INFO")
    logger.info("Starting weekly feature importance analysis...")

    # 1. Load configuration and model
    try:
        config = load_config()
        model_loader = ModelLoader()
        
        # Get the latest production model
        # Assuming we have a way to get 'production' model, otherwise latest
        model, metadata = model_loader.load_latest_model()
        if model is None:
            logger.error("No model found for analysis")
            return

        model_name = metadata.get('model_type', 'unknown_model')
        logger.info(f"Loaded model: {model_name} (Trained at: {metadata.get('trained_at')})")

        # 2. Load recent data for analysis
        # We use a sample of recent data to calculate SHAP values
        data_loader = DataLoader()
        pairs = config.pairs
        if not pairs:
            pairs = ["BTC/USDT"]
            
        logger.info(f"Loading recent data for pairs: {pairs}")
        
        # Just use the first pair for general feature importance analysis
        # In a real scenario, you might want to aggregate across multiple pairs
        df = data_loader.load_pair_data(pairs[0], config.timeframe)
        
        # 3. Preprocessing (simplified - ideally use the same pipeline as training)
        # Assuming features are already in the dataframe or we need to extract them
        # For SHAP, we need the feature matrix X
        # Here we assume df contains features used by the model
        feature_names = metadata.get('feature_names', [])
        if not feature_names:
            # Fallback: try to get from importance csv if exists
            importance_path = Path(model_loader.models_dir) / f"{metadata.get('model_file', '')}.csv"
            if importance_path.exists():
                importance_df = pd.read_csv(importance_path)
                feature_names = importance_df['feature'].tolist()

        if not feature_names:
            logger.error("Could not determine feature names for SHAP analysis")
            return

        # Ensure all features exist in df
        available_features = [f for f in feature_names if f in df.columns]
        X = df[available_features].tail(1000) # Use last 1000 candles for analysis
        
        # 4. Generate SHAP report
        output_dir = Path("user_data/reports/feature_importance") / datetime.now().strftime("%Y%m%d")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating SHAP report in {output_dir}...")
        
        # Split into background and test for SHAP
        X_train_sample = X.head(500)
        X_test_sample = X.tail(500)
        
        generate_shap_report(model, X_train_sample, X_test_sample, output_dir=str(output_dir))
        
        logger.info("Analysis complete. Review results in user_data/reports/feature_importance/")
        
    except Exception as e:
        logger.exception(f"Feature importance analysis failed: {e}")

if __name__ == "__main__":
    run_weekly_analysis()
