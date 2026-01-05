import pandas as pd
import os

data_dir = 'user_data/data/binance'
files = [f for f in os.listdir(data_dir) if f.endswith('.feather')]

print(f"Found {len(files)} feather files.")

for f in files:
    path = os.path.join(data_dir, f)
    try:
        df = pd.read_feather(path)
        print(f"\n--- {f} ---")
        print(f"Shape: {df.shape}")
        if 'date' in df.columns:
            print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        else:
            print("Columns:", df.columns)
    except Exception as e:
        print(f"Error reading {f}: {e}")
