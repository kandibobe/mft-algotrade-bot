import pandas as pd
import os

def check_data_range():
    data_path = "user_data/data/binance/BTC_USDT-5m.feather"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
    
    df = pd.read_feather(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check ETH data too
    eth_path = "user_data/data/binance/ETH_USDT-5m.feather"
    if os.path.exists(eth_path):
        df_eth = pd.read_feather(eth_path)
        print(f"\nETH Data shape: {df_eth.shape}")
        print(f"ETH Date range: {df_eth['date'].min()} to {df_eth['date'].max()}")

if __name__ == "__main__":
    check_data_range()
