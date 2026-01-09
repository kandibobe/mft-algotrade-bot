"""
Verify Strategy Integration
===========================
Check if Alpha Score and Alternative Data Fetcher are correctly integrated 
into the Stoic Citadel strategies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from user_data.strategies.StoicEnsembleStrategyV5 import StoicEnsembleStrategyV5
from src.config.unified_config import load_config
from src.utils.regime_detection import calculate_regime

def verify():
    print("--- Verifying Strategy Integration ---")
    
    # 1. Mock Dataframe
    df = pd.DataFrame({
        'date': pd.date_range(start='2026-01-01', periods=100, freq='5min'),
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 5000, 100)
    })
    
    # 2. Load Strategy
    config = {
        'runmode': 'dry_run',
        'pairs': ['BTC/USDT'],
        'timeframe': '5m',
        'user_data_dir': 'user_data'
    }
    
    try:
        strategy = StoicEnsembleStrategyV5(config)
        print("[OK] Strategy initialized")

        # Mock Data Provider (dp) to avoid AttributeError
        strategy.dp = MagicMock()
        strategy.dp.runmode.value = "dry_run"
        # Return a mock dataframe for informative pairs
        inf_df = pd.DataFrame({
            'date': pd.date_range(start='2026-01-01', periods=100, freq='1h'),
            'close': np.random.randn(100) + 100
        })
        strategy.dp.get_pair_dataframe.return_value = inf_df
        strategy.dp.ticker.return_value = {'bid': 100, 'ask': 100.1, 'quoteVolume': 1000000}
        
        # Check if fetcher exists
        if hasattr(strategy, 'alt_data_fetcher'):
            print("[OK] AlternativeDataFetcher found in strategy")
        else:
            print("[FAIL] AlternativeDataFetcher NOT found in strategy")
            return

        # 3. Simulate populate_indicators
        # We manually set last_alt_data to test integration
        strategy.last_alt_data = {'alpha_score': 75}
        
        # Mock StoicLogic.calculate_atr to avoid parameter errors in some versions
        from src.strategies.core_logic import StoicLogic
        if not hasattr(StoicLogic, 'calculate_atr'):
             # If it's missing or signature is different, we ensures it works
             pass

        # Prepare dataframe with required columns for Technicals
        df['close'] = df['close'].astype(float)
        
        processed_df = strategy.populate_indicators(df, {'pair': 'BTC/USDT'})
        
        print(f"DEBUG: Processed DF columns: {processed_df.columns.tolist()}")
        if 'alpha_score' in processed_df.columns:
            print(f"[OK] 'alpha_score' column found in DataFrame. Value: {processed_df['alpha_score'].iloc[-1]}")
        else:
            print("[FAIL] 'alpha_score' column NOT found in processed DataFrame")

        # 4. Check confirm_trade_entry logic
        # Should return False if alpha_score < 20 (we'll set it to 10 for test)
        strategy.last_alt_data = {'alpha_score': 10}
        can_entry = strategy.confirm_trade_entry(
            pair='BTC/USDT', order_type='limit', amount=1.0, rate=100.0,
            time_in_force='gtc', current_time=None, entry_tag=None, side='long'
        )
        
        if not can_entry:
            print("[OK] confirm_trade_entry correctly BLOCKED trade with low Alpha Score (10)")
        else:
            print("[FAIL] confirm_trade_entry FAILED to block trade with low Alpha Score")

    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
