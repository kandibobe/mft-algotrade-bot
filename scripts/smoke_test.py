#!/usr/bin/env python3
"""
Stoic Citadel - Smoke Test Script
==================================

Quick validation that all core modules work correctly.
Run this after any significant changes.

Usage:
    python scripts/smoke_test.py

Exit codes:
    0 - All tests passed
    1 - Some tests failed
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("\n[TEST] Testing imports...")
    
    try:
        from src.data.loader import get_ohlcv, load_csv
        from src.data.validator import validate_ohlcv
        from src.data.downloader import download_data
        print("  [OK] src.data modules")
    except ImportError as e:
        print(f"  [FAIL] src.data: {e}")
        return False
    
    try:
        from src.utils.indicators import (
            calculate_rsi, calculate_macd, calculate_all_indicators
        )
        print("  [OK] src.utils.indicators")
    except ImportError as e:
        print(f"  [FAIL] src.utils.indicators: {e}")
        return False
    
    try:
        from src.utils.risk import (
            calculate_position_size_fixed_risk,
            calculate_sharpe_ratio
        )
        print("  [OK] src.utils.risk")
    except ImportError as e:
        print(f"  [FAIL] src.utils.risk: {e}")
        return False
    
    try:
        from src.utils.regime_detection import (
            calculate_regime_score,
            get_regime_parameters
        )
        print("  [OK] src.utils.regime_detection")
    except ImportError as e:
        print(f"  [FAIL] src.utils.regime_detection: {e}")
        return False
    
    try:
        from src.strategies.strategy_config import StrategyConfig
        print("  [OK] src.strategies")
    except ImportError as e:
        print(f"  [FAIL] src.strategies: {e}")
        return False
    
    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\n[TEST] Testing data loading...")
    
    try:
        from src.data.loader import load_csv
        from src.data.validator import validate_ohlcv
        
        fixture_path = project_root / 'tests/fixtures/sample_data/BTC_USDT-5m.csv'
        
        if not fixture_path.exists():
            print(f"  [WARN] Fixture not found: {fixture_path}")
            return True  # Not a failure, just missing fixture
        
        df = load_csv(fixture_path)
        print(f"  [OK] Loaded {len(df)} rows")
        
        is_valid, issues = validate_ohlcv(df)
        if is_valid:
            print("  [OK] Data validation passed")
        else:
            print(f"  [FAIL] Validation issues: {issues}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Data loading failed: {e}")
        return False


def test_indicators():
    """Test indicator calculations."""
    print("\n[TEST] Testing indicators...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.utils.indicators import (
            calculate_rsi, calculate_macd, calculate_ema,
            calculate_bollinger_bands, calculate_all_indicators
        )
        
        # Generate test data
        np.random.seed(42)
        n = 200
        close = pd.Series(100 + np.random.randn(n).cumsum())
        
        # Test individual indicators
        rsi = calculate_rsi(close)
        assert not rsi.isna().all(), "RSI all NaN"
        assert 0 <= rsi.dropna().min() <= 100, "RSI out of range"
        print("  [OK] RSI calculation")
        
        macd = calculate_macd(close)
        assert 'macd' in macd and 'signal' in macd
        print("  [OK] MACD calculation")
        
        bb = calculate_bollinger_bands(close)
        assert 'upper' in bb and 'lower' in bb
        print("  [OK] Bollinger Bands calculation")
        
        # Test combined calculation
        df = pd.DataFrame({
            'open': close + np.random.randn(n),
            'high': close + abs(np.random.randn(n)),
            'low': close - abs(np.random.randn(n)),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        result = calculate_all_indicators(df)
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'atr' in result.columns
        print("  [OK] All indicators calculation")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Indicator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk():
    """Test risk calculations."""
    print("\n[TEST] Testing risk management...")
    
    try:
        from src.utils.risk import (
            calculate_position_size_fixed_risk,
            calculate_max_drawdown,
            calculate_sharpe_ratio
        )
        import pandas as pd
        import numpy as np
        
        # Test position sizing
        position = calculate_position_size_fixed_risk(
            account_balance=10000,
            risk_per_trade=0.02,
            entry_price=50000,
            stop_loss_price=47500
        )
        assert position > 0, "Position size should be positive"
        print(f"  [OK] Position sizing: {position:.4f}")
        
        # Test drawdown
        equity = pd.Series([100, 110, 105, 120, 100, 115])
        max_dd, _, _ = calculate_max_drawdown(equity)
        assert 0 <= max_dd <= 1, "Max DD should be 0-1"
        print(f"  [OK] Max drawdown: {max_dd:.2%}")
        
        # Test Sharpe
        returns = pd.Series(np.random.randn(100) * 0.01)
        sharpe = calculate_sharpe_ratio(returns)
        print(f"  [OK] Sharpe ratio: {sharpe:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Risk test failed: {e}")
        return False


def test_regime():
    """Test regime detection."""
    print("\n[TEST] Testing regime detection...")
    
    try:
        from src.utils.regime_detection import (
            calculate_regime_score,
            get_regime_parameters
        )
        import pandas as pd
        import numpy as np
        
        # Generate test data
        np.random.seed(42)
        n = 200
        close = pd.Series(100 + np.random.randn(n).cumsum())
        high = close + abs(np.random.randn(n))
        low = close - abs(np.random.randn(n))
        volume = pd.Series(np.random.randint(1000, 10000, n))
        
        regime_data = calculate_regime_score(high, low, close, volume)
        assert 'regime_score' in regime_data.columns
        print("  [OK] Regime score calculation")
        
        # Test parameter adjustment
        score = regime_data['regime_score'].iloc[-1]
        params = get_regime_parameters(score)
        assert 'mode' in params
        assert 'risk_per_trade' in params
        print(f"  [OK] Regime mode: {params['mode']}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Regime test failed: {e}")
        return False


def test_config():
    """Test configuration system."""
    print("\n[TEST] Testing configuration...")
    
    try:
        from src.strategies.strategy_config import StrategyConfig
        
        # Test default config
        config = StrategyConfig()
        assert config.validate()
        print("  [OK] Default config valid")
        
        # Test config export
        config_dict = config.to_dict()
        assert 'risk_per_trade' in config_dict
        print("  [OK] Config export")
        
        # Test YAML loading if file exists
        yaml_path = project_root / 'config/strategy_config.yaml'
        if yaml_path.exists():
            loaded = StrategyConfig.from_file(str(yaml_path))
            assert loaded.validate()
            print("  [OK] YAML config loading")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Config test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("")
    print("============================================================")
    print("           STOIC CITADEL - SMOKE TEST                       ")
    print("============================================================")
    
    results = []
    
    results.append(('Imports', test_imports()))
    results.append(('Data Loading', test_data_loading()))
    results.append(('Indicators', test_indicators()))
    results.append(('Risk Management', test_risk()))
    results.append(('Regime Detection', test_regime()))
    results.append(('Configuration', test_config()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        color_status = status
        print(f"  {name}: {color_status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("")
        print("ALL SMOKE TESTS PASSED!")
        print("")
        print("Stoic Citadel is ready for use.")
        print("")
        return 0
    else:
        print("")
        print("Some tests failed. Please review the errors above.")
        print("")
        return 1


if __name__ == '__main__':
    sys.exit(main())
