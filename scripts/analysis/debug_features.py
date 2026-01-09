
import logging
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.ml.training.feature_engineering import FeatureEngineer, FeatureConfig

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("src.ml.training.feature_engineering")
logger.setLevel(logging.DEBUG)

def debug_features():
    try:
        # Load data
        print("Loading data...")
        df = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather")
        print(f"Data shape: {df.shape}")
        
        # Initialize Feature Engineer
        config = FeatureConfig(
            include_price_features=True,
            include_volume_features=True, 
            include_time_features=True,
            scale_features=False, # Disable scaling for debugging to see raw values
            remove_correlated=False
        )
        engineer = FeatureEngineer(config)
        
        # Generate features
        print("Generating features...")
        features = engineer.prepare_data(df, use_cache=False)
        
        # Check problematic columns
        cols_to_check = ['intraday_return', 'close_open_ratio', 'vwap_diff']
        
        for col in cols_to_check:
            if col in features.columns:
                series = features[col]
                var = series.var()
                mean = series.mean()
                print(f"\n--- {col} ---")
                print(f"Variance: {var}")
                print(f"Mean: {mean}")
                print(f"Head: {series.head().tolist()}")
                print(f"Zeros count: {(series == 0).sum()}")
                print(f"NaN count: {series.isna().sum()}")
                
                if var < 1e-12:
                    print("!!! LOW VARIANCE DETECTED !!!")
            else:
                print(f"\nMissing column: {col}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_features()
