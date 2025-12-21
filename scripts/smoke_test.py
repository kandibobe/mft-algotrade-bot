#!/usr/bin/env python3
"""
Stoic Citadel - Enhanced Smoke Test Script
===========================================

Quick validation that all core modules work correctly.
Run this after any significant changes.

Features:
- Tests for all critical modules (CircuitBreaker, InferenceService, MarketRegimeFilter, etc.)
- Colorful output with emojis ✅ ❌
- Command-line arguments for selective testing
- Timing measurements for each test
- Environment checks (Redis, config files)
- Detailed error reporting with optional traceback

Usage:
    python scripts/smoke_test.py                    # Run all tests
    python scripts/smoke_test.py --test imports     # Run only import tests
    python scripts/smoke_test.py --verbose          # Show detailed output
    python scripts/smoke_test.py --list             # List all available tests

Exit codes:
    0 - All tests passed
    1 - Some tests failed
"""

import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Global settings
VERBOSE = False
USE_COLOR = True  # Try to use color if supported


def color_text(text: str, color_code: str) -> str:
    """Add ANSI color codes if USE_COLOR is True."""
    if not USE_COLOR:
        return text
    return f"\033[{color_code}m{text}\033[0m"


def print_success(msg: str):
    """Print success message with green checkmark."""
    print(f"{color_text('✅', '32')} {msg}")


def print_failure(msg: str):
    """Print failure message with red cross."""
    print(f"{color_text('❌', '31')} {msg}")


def print_warning(msg: str):
    """Print warning message with yellow exclamation."""
    print(f"{color_text('⚠️', '33')} {msg}")


def print_info(msg: str):
    """Print info message."""
    print(f"{color_text('ℹ️', '36')} {msg}")


def print_header(msg: str):
    """Print section header."""
    print(f"\n{color_text('='*60, '1;37')}")
    print(f"{color_text(msg, '1;37')}")
    print(f"{color_text('='*60, '1;37')}")


def test_imports() -> Tuple[bool, str]:
    """Test that all modules can be imported."""
    print_header("Testing Imports")
    
    modules_to_test = [
        ("src.data.loader", ["get_ohlcv", "load_csv"], False),
        ("src.data.validator", ["validate_ohlcv"], False),
        ("src.data.downloader", ["download_data"], False),
        ("src.utils.indicators", ["calculate_rsi", "calculate_macd", "calculate_all_indicators"], False),
        ("src.utils.risk", ["calculate_position_size_fixed_risk", "calculate_sharpe_ratio"], False),
        ("src.utils.regime_detection", ["calculate_regime_score", "get_regime_parameters"], False),
        ("src.strategies.strategy_config", ["StrategyConfig"], False),
        # New modules from the task
        ("src.order_manager.circuit_breaker", ["CircuitBreaker", "CircuitBreakerConfig"], False),
        ("src.ml.inference_service", ["MLInferenceService", "PredictionRequest"], False),
        ("src.strategies.market_regime", ["MarketRegimeFilter", "RegimeFilterConfig"], False),
        ("user_data.strategies.StoicEnsembleStrategyV3", ["StoicEnsembleStrategyV3"], True),  # Optional: freqtrade dependency
    ]
    
    all_passed = True
    failures = []
    
    for module_name, attributes, optional in modules_to_test:
        try:
            # Dynamic import
            module = __import__(module_name, fromlist=attributes)
            for attr in attributes:
                if hasattr(module, attr):
                    if VERBOSE:
                        print_success(f"  {module_name}.{attr} imported successfully")
                else:
                    raise AttributeError(f"Attribute {attr} not found in {module_name}")
            print_success(f"{module_name}")
        except (ImportError, AttributeError) as e:
            if optional:
                print_warning(f"{module_name}: {str(e)} (optional)")
            else:
                all_passed = False
                error_msg = f"{module_name}: {str(e)}"
                print_failure(f"{module_name}: {str(e)}")
                failures.append(error_msg)
            if VERBOSE:
                traceback.print_exc()
    
    return all_passed, "All imports passed" if all_passed else f"Import failures: {', '.join(failures)}"


def test_circuit_breaker() -> Tuple[bool, str]:
    """Test CircuitBreaker functionality."""
    print_header("Testing Circuit Breaker")
    
    try:
        from src.order_manager.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        
        # Create config
        config = CircuitBreakerConfig(
            max_daily_loss_pct=5.0,
            max_drawdown_pct=15.0,
            max_consecutive_losses=5,
            max_orders_per_minute=10
        )
        
        # Create breaker
        breaker = CircuitBreaker(config)
        
        # Test properties
        assert not breaker.is_tripped, "New breaker should not be tripped"
        assert breaker.is_operational, "New breaker should be operational"
        
        # Test balance update
        breaker.update_balance(current=10000, peak=11000)
        drawdown = breaker.calculate_drawdown()
        assert drawdown < 0, "Drawdown should be negative when current < peak"
        
        # Test trade recording
        breaker.record_trade(pnl=100)
        breaker.record_trade(pnl=-50)
        
        # Test order recording
        breaker.record_order()
        
        print_success("CircuitBreaker basic functionality")
        return True, "CircuitBreaker tests passed"
        
    except Exception as e:
        print_failure(f"CircuitBreaker test failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"CircuitBreaker test failed: {e}"


def test_inference_service() -> Tuple[bool, str]:
    """Test ML Inference Service imports and basic functionality."""
    print_header("Testing ML Inference Service")
    
    try:
        from src.ml.inference_service import (
            MLInferenceService, 
            PredictionRequest,
            MLModelConfig
        )
        
        # Test that classes can be instantiated
        config = MLModelConfig(
            model_name="test_model",
            model_path="test.pkl",
            feature_columns=["feature1", "feature2"]
        )
        
        # Create service (without Redis for smoke test)
        service = MLInferenceService(models={"test_model": config})
        
        # Create prediction request
        request = PredictionRequest(
            request_id="test_123",
            model_name="test_model",
            features={"feature1": 0.5, "feature2": 0.3}
        )
        
        print_success("ML Inference Service imports and basic instantiation")
        return True, "InferenceService tests passed"
        
    except Exception as e:
        print_failure(f"InferenceService test failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"InferenceService test failed: {e}"


def test_market_regime_filter() -> Tuple[bool, str]:
    """Test Market Regime Filter."""
    print_header("Testing Market Regime Filter")
    
    try:
        from src.strategies.market_regime import MarketRegimeFilter, RegimeFilterConfig
        import pandas as pd
        import numpy as np
        
        # Create config
        config = RegimeFilterConfig(
            ema_period=200,
            adx_trend_threshold=25.0,
            allow_trade_in_sideways=False
        )
        
        # Create filter
        filter = MarketRegimeFilter(config)
        
        # Create test dataframe
        np.random.seed(42)
        n = 300
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, n),
            'high': np.random.uniform(50000, 52000, n),
            'low': np.random.uniform(38000, 40000, n),
            'close': np.random.uniform(40000, 50000, n),
            'volume': np.random.uniform(1000, 10000, n)
        }, index=dates)
        
        # Test regime detection
        regime, details = filter.detect_regime(df)
        assert regime is not None, "Regime should be detected"
        
        # Test trade decision
        should_trade, reason = filter.should_trade(df, side='buy')
        # Just check that it returns a boolean and string
        
        print_success(f"MarketRegimeFilter detected regime: {regime.value}")
        return True, f"MarketRegimeFilter tests passed (regime: {regime.value})"
        
    except Exception as e:
        print_failure(f"MarketRegimeFilter test failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"MarketRegimeFilter test failed: {e}"


def test_strategy() -> Tuple[bool, str]:
    """Test that strategy can be loaded."""
    print_header("Testing Strategy")
    
    try:
        from user_data.strategies.StoicEnsembleStrategyV3 import StoicEnsembleStrategyV3
        
        # Try to create strategy instance
        strategy = StoicEnsembleStrategyV3()
        
        # Check basic attributes
        assert hasattr(strategy, 'timeframe'), "Strategy should have timeframe attribute"
        assert hasattr(strategy, 'stoploss'), "Strategy should have stoploss attribute"
        
        print_success(f"Strategy loaded: {strategy.__class__.__name__}")
        return True, f"Strategy {strategy.__class__.__name__} loaded successfully"
        
    except ImportError as e:
        # Freqtrade not installed
        print_warning(f"Strategy test skipped: {e}")
        return True, "Strategy test skipped (freqtrade not installed)"
    except Exception as e:
        print_failure(f"Strategy test failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"Strategy test failed: {e}"


def test_data_loading() -> Tuple[bool, str]:
    """Test data loading functionality."""
    print_header("Testing Data Loading")
    
    try:
        from src.data.loader import load_csv
        from src.data.validator import validate_ohlcv
        
        fixture_path = project_root / 'tests/fixtures/sample_data/BTC_USDT-5m.csv'
        
        if not fixture_path.exists():
            print_warning(f"Fixture not found: {fixture_path}")
            return True, "Data loading skipped (fixture not found)"
        
        df = load_csv(fixture_path)
        assert len(df) > 0, "Dataframe should not be empty"
        
        is_valid, issues = validate_ohlcv(df)
        if is_valid:
            print_success(f"Data loaded: {len(df)} rows, validation passed")
            return True, f"Data loading passed ({len(df)} rows)"
        else:
            print_failure(f"Data validation issues: {issues}")
            return False, f"Data validation failed: {issues}"
        
    except Exception as e:
        print_failure(f"Data loading failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"Data loading failed: {e}"


def test_indicators() -> Tuple[bool, str]:
    """Test indicator calculations."""
    print_header("Testing Indicators")
    
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
        print_success("RSI calculation")
        
        macd = calculate_macd(close)
        assert 'macd' in macd and 'signal' in macd
        print_success("MACD calculation")
        
        bb = calculate_bollinger_bands(close)
        assert 'upper' in bb and 'lower' in bb
        print_success("Bollinger Bands calculation")
        
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
        print_success("All indicators calculation")
        
        return True, "Indicator tests passed"
        
    except Exception as e:
        print_failure(f"Indicator test failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"Indicator test failed: {e}"


def test_risk() -> Tuple[bool, str]:
    """Test risk calculations."""
    print_header("Testing Risk Management")
    
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
        print_success(f"Position sizing: {position:.4f}")
        
        # Test drawdown
        equity = pd.Series([100, 110, 105, 120, 100, 115])
        max_dd, _, _ = calculate_max_drawdown(equity)
        assert 0 <= max_dd <= 1, "Max DD should be 0-1"
        print_success(f"Max drawdown: {max_dd:.2%}")
        
        # Test Sharpe
        returns = pd.Series(np.random.randn(100) * 0.01)
        sharpe = calculate_sharpe_ratio(returns)
        print_success(f"Sharpe ratio: {sharpe:.2f}")
        
        return True, "Risk tests passed"
        
    except Exception as e:
        print_failure(f"Risk test failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"Risk test failed: {e}"


def test_regime_detection() -> Tuple[bool, str]:
    """Test regime detection."""
    print_header("Testing Regime Detection")
    
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
        print_success("Regime score calculation")
        
        # Test parameter adjustment
        score = regime_data['regime_score'].iloc[-1]
        params = get_regime_parameters(score)
        assert 'mode' in params
        assert 'risk_per_trade' in params
        print_success(f"Regime mode: {params['mode']}")
        
        return True, "Regime detection tests passed"
        
    except Exception as e:
        print_failure(f"Regime detection test failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"Regime detection test failed: {e}"


def test_config() -> Tuple[bool, str]:
    """Test configuration system."""
    print_header("Testing Configuration")
    
    try:
        from src.strategies.strategy_config import StrategyConfig
        
        # Test default config
        config = StrategyConfig()
        assert config.validate()
        print_success("Default config valid")
        
        # Test config export
        config_dict = config.to_dict()
        assert 'risk_per_trade' in config_dict
        print_success("Config export")
        
        # Test YAML loading if file exists
        yaml_path = project_root / 'config/strategy_config.yaml'
        if yaml_path.exists():
            loaded = StrategyConfig.from_file(str(yaml_path))
            assert loaded.validate()
            print_success("YAML config loading")
        else:
            print_warning("YAML config file not found, skipping loading test")
        
        return True, "Config tests passed"
        
    except Exception as e:
        print_failure(f"Config test failed: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False, f"Config test failed: {e}"


def test_environment() -> Tuple[bool, str]:
    """Test environment dependencies."""
    print_header("Testing Environment")
    
    checks = []
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 9):
        print_success(f"Python version: {python_version}")
        checks.append(("Python >= 3.9", True))
    else:
        print_failure(f"Python version too old: {python_version} (need >= 3.9)")
        checks.append(("Python >= 3.9", False))
    
    # Check essential directories
    essential_dirs = [
        ("src", project_root / "src"),
        ("config", project_root / "config"),
        ("user_data/strategies", project_root / "user_data" / "strategies"),
    ]
    
    for name, path in essential_dirs:
        if path.exists():
            print_success(f"Directory exists: {name}")
            checks.append((f"Directory {name}", True))
        else:
            print_warning(f"Directory missing: {name}")
            checks.append((f"Directory {name}", False))
    
    # Check config files
    config_files = [
        ("strategy_config.yaml", project_root / "config" / "strategy_config.yaml"),
    ]
    
    for name, path in config_files:
        if path.exists():
            print_success(f"Config file exists: {name}")
            checks.append((f"Config {name}", True))
        else:
            print_warning(f"Config file missing: {name}")
            checks.append((f"Config {name}", False))
    
    # Summary
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    if passed == total:
        return True, f"Environment checks passed ({passed}/{total})"
    else:
        return False, f"Environment checks failed ({passed}/{total})"


def run_all_tests(selected_tests: Optional[List[str]] = None) -> Tuple[int, int, List[Tuple[str, bool, str, float]]]:
    """Run all tests and return results."""
    # Define all available tests
    all_tests = [
        ("imports", test_imports),
        ("circuit_breaker", test_circuit_breaker),
        ("inference_service", test_inference_service),
        ("market_regime_filter", test_market_regime_filter),
        ("strategy", test_strategy),
        ("data_loading", test_data_loading),
        ("indicators", test_indicators),
        ("risk", test_risk),
        ("regime_detection", test_regime_detection),
        ("config", test_config),
        ("environment", test_environment),
    ]
    
    # Filter if selected_tests provided
    if selected_tests:
        tests_to_run = [(name, fn) for name, fn in all_tests if name in selected_tests]
        if not tests_to_run:
            print_failure(f"No tests matched: {selected_tests}")
            return 0, 0, []
    else:
        tests_to_run = all_tests
    
    print_header("STOIC CITADEL - SMOKE TEST")
    print_info(f"Running {len(tests_to_run)} tests...")
    
    results = []
    passed = 0
    failed = 0
    
    for test_name, test_fn in tests_to_run:
        start_time = time.time()
        try:
            success, message = test_fn()
            elapsed = time.time() - start_time
            results.append((test_name, success, message, elapsed))
            
            if success:
                passed += 1
                print_success(f"{test_name}: {message} ({elapsed:.2f}s)")
            else:
                failed += 1
                print_failure(f"{test_name}: {message} ({elapsed:.2f}s)")
                
        except Exception as e:
            elapsed = time.time() - start_time
            failed += 1
            error_msg = f"Test crashed: {str(e)}"
            results.append((test_name, False, error_msg, elapsed))
            print_failure(f"{test_name}: {error_msg} ({elapsed:.2f}s)")
            if VERBOSE:
                traceback.print_exc()
    
    return passed, failed, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stoic Citadel Smoke Test")
    parser.add_argument("--test", action="append", help="Run specific test(s)")
    parser.add_argument("--list", action="store_true", help="List all available tests")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()
    
    global VERBOSE, USE_COLOR
    VERBOSE = args.verbose
    USE_COLOR = not args.no_color
    
    # List tests if requested
    if args.list:
        print("Available tests:")
        tests = [
            "imports", "circuit_breaker", "inference_service", "market_regime_filter",
            "strategy", "data_loading", "indicators", "risk", "regime_detection",
            "config", "environment"
        ]
        for test in tests:
            print(f"  - {test}")
        return 0
    
    # Run tests
    passed, failed, results = run_all_tests(args.test)
    
    # Print summary
    print_header("TEST SUMMARY")
    print_info(f"Total tests: {passed + failed}")
    print_success(f"Passed: {passed}")
    if failed > 0:
        print_failure(f"Failed: {failed}")
    
    # Show failed test details
    if failed > 0:
        print_header("FAILED TESTS")
        for test_name, success, message, elapsed in results:
            if not success:
                print_failure(f"{test_name}: {message}")
    
    # Exit code
    if failed == 0:
        print_header("✅ ALL TESTS PASSED")
        print_info("Stoic Citadel is ready for use.")
        return 0
    else:
        print_header("❌ SOME TESTS FAILED")
        print_info("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
