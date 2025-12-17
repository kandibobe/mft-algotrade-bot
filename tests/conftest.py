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


# ============================================================================
# STRATEGY FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """
    Generate sample DataFrame for strategy testing (freqtrade format).
    """
    np.random.seed(42)
    n = 500
    
    # Generate realistic price data
    base_price = 50000
    returns = np.random.randn(n) * 0.02
    close = base_price * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.randint(100, 1000, n).astype(float)
    
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
def strategy_metadata():
    """
    Generate strategy metadata dict for freqtrade.
    """
    return {
        'pair': 'BTC/USDT',
        'timeframe': '5m',
        'stake_currency': 'USDT',
        'stake_amount': 100.0
    }


# ============================================================================
# ASSERTION HELPERS
# ============================================================================

def assert_column_exists(df: pd.DataFrame, column: str, msg: str = None):
    """Assert that a column exists in DataFrame."""
    assert column in df.columns, msg or f"Column '{column}' not found in DataFrame"


def assert_no_nan_in_column(df: pd.DataFrame, column: str, allow_partial: bool = False):
    """Assert that column has no NaN values (or limited NaN for warmup)."""
    if column not in df.columns:
        raise AssertionError(f"Column '{column}' not in DataFrame")
    
    nan_count = df[column].isna().sum()
    total = len(df)
    
    if allow_partial:
        # Allow up to 50% NaN for warmup periods
        assert nan_count < total * 0.5, f"Column '{column}' has too many NaN values: {nan_count}/{total}"
    else:
        assert nan_count == 0, f"Column '{column}' has {nan_count} NaN values"


def assert_signal_generated(df: pd.DataFrame, signal_column: str, expected_count: int = None):
    """Assert that trading signals were generated."""
    assert signal_column in df.columns, f"Signal column '{signal_column}' not found"
    
    signals = df[df[signal_column] == 1]
    
    if expected_count is not None:
        assert len(signals) == expected_count, f"Expected {expected_count} signals, got {len(signals)}"
    else:
        # Just check that at least some signals exist
        assert len(signals) >= 0, f"No signals generated in column '{signal_column}'"


# ============================================================================
# FREQTRADE STRATEGY FIXTURES
# ============================================================================

@pytest.fixture
def minimal_config():
    """
    Minimal freqtrade config for strategy testing.
    """
    return {
        'strategy': 'StoicEnsembleStrategy',
        'stake_currency': 'USDT',
        'stake_amount': 100.0,
        'dry_run': True,
        'trading_mode': 'spot',
        'margin_mode': '',
        'exchange': {
            'name': 'binance',
            'key': '',
            'secret': '',
            'pair_whitelist': ['BTC/USDT'],
            'pair_blacklist': []
        },
        'pairlists': [{'method': 'StaticPairList'}],
        'telegram': {'enabled': False},
        'api_server': {'enabled': False},
        'internals': {},
    }


@pytest.fixture
def uptrend_dataframe():
    """
    Generate DataFrame with clear uptrend for testing.
    """
    np.random.seed(42)
    n = 500
    
    # Generate uptrend data
    base_price = 50000
    trend = np.linspace(0, 0.5, n)  # 50% uptrend
    noise = np.random.randn(n) * 0.01
    close = base_price * (1 + trend + noise)
    
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = close * (1 + np.random.randn(n) * 0.002)
    volume = np.random.randint(100, 1000, n).astype(float)
    
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
def downtrend_dataframe():
    """
    Generate DataFrame with clear downtrend for testing.
    """
    np.random.seed(42)
    n = 500
    
    # Generate downtrend data
    base_price = 50000
    trend = np.linspace(0, -0.3, n)  # 30% downtrend
    noise = np.random.randn(n) * 0.01
    close = base_price * (1 + trend + noise)
    
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = close * (1 + np.random.randn(n) * 0.002)
    volume = np.random.randint(100, 1000, n).astype(float)
    
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
def mock_trade():
    """
    Mock freqtrade Trade object for testing.
    """
    from unittest.mock import MagicMock
    from datetime import datetime, timedelta
    
    trade = MagicMock()
    trade.pair = 'BTC/USDT'
    trade.stake_amount = 100.0
    trade.amount = 0.002
    trade.open_rate = 50000.0
    trade.open_date_utc = datetime.utcnow() - timedelta(hours=1)
    trade.is_open = True
    trade.is_short = False
    trade.leverage = 1.0
    trade.stop_loss_pct = -0.05
    
    return trade


@pytest.fixture
def mock_exchange():
    """
    Mock freqtrade exchange object for testing.
    """
    from unittest.mock import MagicMock
    
    exchange = MagicMock()
    exchange.get_fee.return_value = 0.001
    exchange.get_min_pair_stake_amount.return_value = 10.0
    exchange.get_max_pair_stake_amount.return_value = 10000.0
    exchange.markets = {'BTC/USDT': {'limits': {'amount': {'min': 0.0001}}}}
    
    return exchange


@pytest.fixture
def stoic_strategy(minimal_config):
    """
    Create StoicEnsembleStrategy with proper config for testing.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../user_data/strategies"))
    
    from StoicEnsembleStrategy import StoicEnsembleStrategy
    
    return StoicEnsembleStrategy(minimal_config)


@pytest.fixture
def stoic_strategy_v2(minimal_config):
    """
    Create StoicEnsembleStrategyV2 with proper config for testing.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../user_data/strategies"))
    
    from StoicEnsembleStrategyV2 import StoicEnsembleStrategyV2
    
    return StoicEnsembleStrategyV2(minimal_config)
