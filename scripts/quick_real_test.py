"""
Quick test of the pipeline on real data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ml.training.advanced_pipeline import AdvancedTradingPipeline

def main():
    print("="*80)
    print("QUICK REAL DATA PIPELINE TEST")
    print("="*80)
    
    # Load data
    data_path = Path("user_data/data/binance/BTC_USDT-5m.feather")  # Use 5m for speed
    print(f"Loading data from {data_path}")
    
    df = pd.read_feather(data_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    df.columns = df.columns.str.lower()
    df = df.sort_index()
    
    # Take only last 5000 samples for speed
    df = df.iloc[-5000:]
    print(f"Using {len(df)} samples from {df.index[0]} to {df.index[-1]}")
    
    # Create pipeline with default config
    pipeline = AdvancedTradingPipeline()
    
    # Run pipeline
    print("\nRunning pipeline...")
    results = pipeline.run(df)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    if results:
        print(f"Data samples: {len(results.get('data_processed', []))}")
        print(f"Features generated: {results.get('features', pd.DataFrame()).shape[1] if isinstance(results.get('features'), pd.DataFrame) else 'N/A'}")
        print(f"Features selected: {results.get('X_selected', pd.DataFrame()).shape[1] if isinstance(results.get('X_selected'), pd.DataFrame) else 'N/A'}")
        
        y = results.get('y')
        if y is not None:
            print(f"Buy signals: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
        
        metrics = results.get('trading_metrics', {})
        if metrics:
            print(f"\nTrading Performance:")
            print(f"  Precision: {metrics.get('precision', 0):.2%}")
            print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        
        # Feature importance
        importance_df = results.get('feature_importance')
        if importance_df is not None and len(importance_df) > 0:
            print(f"\nTop 5 Features:")
            for idx, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
