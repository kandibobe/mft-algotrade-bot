import pandas as pd
import sys

try:
    df = pd.read_feather('user_data/data/binance/BTC_USDT-5m.feather')
    print(f'Rows: {len(df)}')
    print(f'Date range: {df["date"].min()} to {df["date"].max()}')
    print(f'Days: {(df["date"].max() - df["date"].min()).days}')
    
    # Also check the size
    import os
    size = os.path.getsize('user_data/data/binance/BTC_USDT-5m.feather')
    print(f'File size: {size / 1024 / 1024:.2f} MB')
    
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
