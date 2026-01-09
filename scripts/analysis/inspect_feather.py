import pandas as pd
import logging

try:
    df = pd.read_feather("user_data/data/binance/BTC_USDT-1h.feather")
    print(df.head())
    print("\nDescription:")
    print(df.describe())
    
    # Check variance of raw returns
    df['returns'] = df['close'].pct_change()
    print("\nReturns Variance:", df['returns'].var())
except Exception as e:
    print(e)
