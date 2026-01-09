"""
Verify SHAP Explainer
====================
Check if SHAP utilities work with mock data and models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.ml.explainability import generate_shap_report

def verify():
    print("--- Verifying SHAP Explainer ---")
    
    # 1. Create Mock Data
    X = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    y = np.random.randint(0, 2, 100)
    
    # 2. Train a fast mock model
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    print("[OK] Mock model trained")
    
    # 3. Generate Report
    output_dir = "user_data/reports/test_shap"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Generating SHAP report in {output_dir}...")
        generate_shap_report(model, X, X.tail(20), output_dir=output_dir)
        
        # 4. Check files
        summary_plot = Path(f"{output_dir}/shap_summary.png")
        values_csv = Path(f"{output_dir}/recent_shap_values.csv")
        
        if summary_plot.exists():
            print(f"[OK] Summary plot created: {summary_plot}")
        else:
            print("[FAIL] Summary plot NOT found")
            
        if values_csv.exists():
            print(f"[OK] SHAP values CSV created: {values_csv}")
        else:
            print("[FAIL] SHAP values CSV NOT found")

    except Exception as e:
        print(f"[ERROR] SHAP Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
