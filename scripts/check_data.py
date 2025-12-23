import pandas as pd
import numpy as np

def main():
    path = 'user_data/data/binance/BTC_USDT-5m.feather'
    df = pd.read_feather(path)
    
    print('=== DATA INTEGRITY CHECK ===')
    print(f'Shape: {df.shape}')
    print(f'Date range: {df["date"].min()} to {df["date"].max()}')
    print(f'Number of days: {(df["date"].max() - df["date"].min()).days}')
    
    print('\n=== NaN COUNT ===')
    nan_counts = df.isna().sum()
    print(nan_counts)
    
    print('\n=== GAPS CHECK ===')
    df_sorted = df.sort_values('date').reset_index(drop=True)
    df_sorted['date_diff'] = df_sorted['date'].diff()
    # Expected difference is 5 minutes (300 seconds)
    gaps = df_sorted[df_sorted['date_diff'] > pd.Timedelta('5 minutes')]
    print(f'Number of gaps >5min: {len(gaps)}')
    
    if len(gaps) > 0:
        print('First 5 gaps:')
        for idx, row in gaps.head().iterrows():
            print(f"  Gap at {row['date']}: {row['date_diff']}")
    
    # Check for duplicates
    duplicates = df_sorted.duplicated(subset=['date'], keep=False)
    print(f'\nDuplicate timestamps: {duplicates.sum()}')
    
    # Check data completeness
    total_expected = (df['date'].max() - df['date'].min()).total_seconds() / 300 + 1
    actual = len(df)
    completeness = actual / total_expected * 100
    print(f'\nData completeness: {completeness:.2f}% (expected {total_expected:.0f} candles, got {actual})')
    
    # Check for zero or negative volumes
    zero_volume = (df['volume'] <= 0).sum()
    print(f'\nZero or negative volume rows: {zero_volume}')
    
    # Check for zero price ranges (high == low)
    zero_range = (df['high'] == df['low']).sum()
    print(f'Zero price range rows: {zero_range}')
    
    return df

if __name__ == '__main__':
    main()
