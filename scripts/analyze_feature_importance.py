import sys
import os
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from pathlib import Path

def analyze_feature_importance(models_dir="user_data/models"):
    """
    Analyze feature importance of the latest FreqAI model.
    """
    # Find all joblib files
    files = list(Path(models_dir).rglob("*model.joblib"))
    if not files:
        # Try broader search
        files = list(Path(models_dir).rglob("*.joblib"))
        
    if not files:
        print(f"No models found in {models_dir}")
        return

    # Sort by modification time to get the latest
    latest_model = max(files, key=os.path.getmtime)
    print(f"Loading latest model: {latest_model}")

    try:
        model = joblib.load(latest_model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Unwrap pipeline if necessary
    estimator = model
    if hasattr(model, "named_steps"):
        # Assuming the last step is the regressor/classifier
        step_name = list(model.named_steps.keys())[-1]
        estimator = model.named_steps[step_name]
        print(f"Extracted estimator '{step_name}' from pipeline.")

    # Analyze XGBoost
    if isinstance(estimator, (xgb.XGBModel, xgb.XGBRegressor, xgb.XGBClassifier)):
        print("XGBoost model detected.")
        importance = estimator.feature_importances_
        
        # Try to retrieve feature names
        feature_names = None
        
        # 1. Try from booster
        try:
            feature_names = estimator.get_booster().feature_names
        except:
            pass
            
        # 2. Try from estimator attribute (if fitted with pandas)
        if feature_names is None and hasattr(estimator, "feature_names_in_"):
            feature_names = estimator.feature_names_in_
            
        # 3. Fallback
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Create DataFrame
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
        df_imp = df_imp.sort_values("importance", ascending=False)
        
        print("\nTop 15 Features by Importance:")
        print(df_imp.head(15))
        
        # Check specific features of interest
        print("\n--- Specific Feature Checks ---")
        for key in ['dist_ema', 'vol_ratio', 'rsi']:
            matches = df_imp[df_imp['feature'].str.contains(key, case=False)]
            if not matches.empty:
                print(f"Found related to '{key}':")
                print(matches.head(3))
            else:
                print(f"No features found matching '{key}'")

        # Plot
        try:
            plt.figure(figsize=(12, 8))
            # Plot top 20
            top_n = df_imp.head(20)
            plt.barh(top_n["feature"], top_n["importance"])
            plt.gca().invert_yaxis()
            plt.title(f"Feature Importance - {latest_model.name}")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            
            output_file = "user_data/feature_importance.png"
            plt.savefig(output_file)
            print(f"\n[SUCCESS] Feature importance plot saved to {output_file}")
        except Exception as e:
            print(f"Could not save plot: {e}")
        
    else:
        print(f"Model type {type(estimator)} is not automatically supported by this script.")
        if hasattr(estimator, "feature_importances_"):
             print("Raw feature importances:", estimator.feature_importances_)

if __name__ == "__main__":
    analyze_feature_importance()
