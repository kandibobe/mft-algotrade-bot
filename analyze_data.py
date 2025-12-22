"""
Analyze available trading data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_file(filepath):
    """Analyze a single data file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath}")
    print('='*60)
    
    try:
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif filepath.endswith('.feather'):
            df = pd.read_feather(filepath)
        else:
            print(f"Unsupported format: {filepath}")
            return None
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for date column
        date_col = None
        for col in ['date', 'timestamp', 'time', 'datetime']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
            
            # Calculate timeframe
            if len(df) > 1:
                timeframe = df[date_col].iloc[1] - df[date_col].iloc[0]
                print(f"Timeframe: {timeframe}")
            
            # Set index
            df = df.set_index(date_col)
            df = df.sort_index()
        
        # Check OHLCV columns
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in ohlcv_cols if col not in df.columns]
        if missing:
            print(f"Missing OHLCV columns: {missing}")
        
        # Basic statistics
        if 'close' in df.columns:
            print(f"\nClose price statistics:")
            print(f"  Min: ${df['close'].min():.2f}")
            print(f"  Max: ${df['close'].max():.2f}")
            print(f"  Mean: ${df['close'].mean():.2f}")
            print(f"  Std: ${df['close'].std():.2f}")
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                print(f"\nReturns statistics:")
                print(f"  Mean: {returns.mean():.6f}")
                print(f"  Std: {returns.std():.6f}")
                print(f"  Sharpe (annualized): {returns.mean()/returns.std()*np.sqrt(252):.2f}")
        
        # Missing values
        missing_vals = df.isnull().sum()
        if missing_vals.sum() > 0:
            print(f"\nMissing values:")
            for col, count in missing_vals[missing_vals > 0].items():
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Data quality checks
        print(f"\nData quality checks:")
        if all(col in df.columns for col in ['high', 'low']):
            invalid_hl = (df['high'] < df['low']).sum()
            print(f"  High < Low: {invalid_hl} rows")
        
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = ((df['close'] > df['high']) | (df['close'] < df['low']) | 
                           (df['open'] > df['high']) | (df['open'] < df['low'])).sum()
            print(f"  Invalid OHLC: {invalid_ohlc} rows")
        
        return df
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def main():
    """Main analysis function."""
    data_dir = Path("user_data/data")
    
    # List all data files
    data_files = []
    for ext in ['*.parquet', '*.feather']:
        data_files.extend(list(data_dir.rglob(ext)))
    
    print(f"Found {len(data_files)} data files:")
    for f in data_files:
        print(f"  - {f.relative_to(data_dir)}")
    
    # Analyze each file
    all_data = {}
    for filepath in data_files:
        df = analyze_file(str(filepath))
        if df is not None:
            all_data[filepath.name] = df
    
    # Summary
    print(f"\n{'='*60}")
    print("DATA ANALYSIS SUMMARY")
    print('='*60)
    
    total_rows = sum(len(df) for df in all_data.values())
    print(f"Total rows across all files: {total_rows:,}")
    
    # Recommend best dataset
    if all_data:
        best_file = max(all_data.items(), key=lambda x: len(x[1]))
        print(f"\nRecommended dataset: {best_file[0]}")
        print(f"  Rows: {len(best_file[1]):,}")
        print(f"  Columns: {best_file[1].shape[1]}")
        
        # Check if suitable for ML
        if len(best_file[1]) >= 1000:
            print(f"  Suitability for ML: GOOD (>1000 samples)")
        elif len(best_file[1]) >= 500:
            print(f"  Suitability for ML: MARGINAL (500-1000 samples)")
        else:
            print(f"  Suitability for ML: POOR (<500 samples)")
    
    return all_data

if __name__ == "__main__":
    all_data = main()
