"""
Pytest configuration and fixtures for Stoic Citadel tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_ohlcv():
    """
    Generate sample OHLCV data for testing.
    
    Returns a DataFrame with 200 rows of realistic price data.
    """
    np.random.seed(42)
    n = 200
    
    # Generate realistic price data with trend
    base_price = 100
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    close = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    # Create timestamps
    dates = pd.date_range('2024-01-01', periods=n, freq='5min')
    
    return pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def sample_ohlcv_short():
    """
    Generate short sample OHLCV data (50 rows).
    """
    np.random.seed(42)
    n = 50
    
    close = pd.Series(100 + np.random.randn(n).cumsum())
    
    return pd.DataFrame({
        'open': close + np.random.randn(n) * 0.5,
        'high': close + abs(np.random.randn(n)),
        'low': close - abs(np.random.randn(n)),
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })


@pytest.fixture
def sample_equity_curve():
    """
    Generate sample equity curve for risk testing.
    """
    np.random.seed(42)
    n = 100
    
    # Random returns with slight upward bias
    returns = np.random.randn(n) * 0.02 + 0.001
    equity = 10000 * np.exp(np.cumsum(returns))
    
    return pd.Series(equity)


@pytest.fixture
def sample_returns():
    """
    Generate sample returns series.
    """
    np.random.seed(42)
    n = 252  # One year of daily returns
    
    returns = pd.Series(np.random.randn(n) * 0.02)
    
    return returns


@pytest.fixture
def fixture_dir():
    """
    Return path to test fixtures directory.
    """
    return Path(__file__).parent / 'fixtures'


@pytest.fixture
def sample_data_path(fixture_dir):
    """
    Return path to sample CSV data.
    """
    return fixture_dir / 'sample_data' / 'BTC_USDT-5m.csv'


# ============================================================================
# MARKERS
# ============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "strategy: marks tests as strategy tests"
    )


# ============================================================================
# HOOKS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Automatically add markers based on test location.
    """
    for item in items:
        # Add markers based on path
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_strategies" in str(item.fspath):
            item.add_marker(pytest.mark.strategy)
        else:
            item.add_marker(pytest.mark.unit)
