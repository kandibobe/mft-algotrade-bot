"""
Pytest Configuration and Shared Fixtures
=========================================

Provides reusable test fixtures for Stoic Citadel testing suite.

Author: Stoic Citadel Team
License: MIT
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import MagicMock


# ==============================================================================
# DATAFRAME FIXTURES
# ==============================================================================


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Generate a realistic sample OHLCV dataframe for testing.

    Returns:
        pd.DataFrame: 200 rows of synthetic market data
    """
    rows = 200
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=rows // 288),
        periods=rows,
        freq="5min",
    )

    # Generate synthetic price data with realistic characteristics
    np.random.seed(42)  # Reproducible tests
    close_prices = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    close_prices = np.maximum(close_prices, 50)  # Prevent negative prices

    high_prices = close_prices + np.random.rand(rows) * 2
    low_prices = close_prices - np.random.rand(rows) * 2
    open_prices = close_prices + np.random.randn(rows) * 0.5

    volume = np.random.randint(1000, 10000, rows)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    ).set_index("date")


@pytest.fixture
def uptrend_dataframe() -> pd.DataFrame:
    """
    Generate a dataframe with clear uptrend for testing trend filters.

    Returns:
        pd.DataFrame: Synthetic uptrending market data
    """
    rows = 200
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=rows, freq="5min")

    # Generate uptrending data
    np.random.seed(42)
    trend = np.linspace(100, 150, rows)  # Steady uptrend
    noise = np.random.randn(rows) * 0.5
    close_prices = trend + noise

    high_prices = close_prices + np.random.rand(rows) * 1
    low_prices = close_prices - np.random.rand(rows) * 1
    open_prices = close_prices + np.random.randn(rows) * 0.3
    volume = np.random.randint(5000, 15000, rows)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    ).set_index("date")


@pytest.fixture
def downtrend_dataframe() -> pd.DataFrame:
    """
    Generate a dataframe with clear downtrend for testing filters.

    Returns:
        pd.DataFrame: Synthetic downtrending market data
    """
    rows = 200
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=rows, freq="5min")

    # Generate downtrending data
    np.random.seed(42)
    trend = np.linspace(150, 100, rows)  # Steady downtrend
    noise = np.random.randn(rows) * 0.5
    close_prices = trend + noise

    high_prices = close_prices + np.random.rand(rows) * 1
    low_prices = close_prices - np.random.rand(rows) * 1
    open_prices = close_prices + np.random.randn(rows) * 0.3
    volume = np.random.randint(5000, 15000, rows)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    ).set_index("date")


@pytest.fixture
def sideways_dataframe() -> pd.DataFrame:
    """
    Generate a dataframe with sideways market for testing range strategies.

    Returns:
        pd.DataFrame: Synthetic ranging market data
    """
    rows = 200
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=rows, freq="5min")

    # Generate sideways data
    np.random.seed(42)
    close_prices = 125 + np.random.randn(rows) * 2  # Oscillate around 125

    high_prices = close_prices + np.random.rand(rows) * 1
    low_prices = close_prices - np.random.rand(rows) * 1
    open_prices = close_prices + np.random.randn(rows) * 0.3
    volume = np.random.randint(5000, 15000, rows)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        }
    ).set_index("date")


# ==============================================================================
# EXCHANGE & TRADING FIXTURES
# ==============================================================================


@pytest.fixture
def mock_exchange():
    """
    Create a mock exchange object for testing without real API calls.

    Returns:
        MagicMock: Mock exchange with common methods
    """
    exchange = MagicMock()
    exchange.name = "binance"
    exchange.get_fee.return_value = 0.001  # 0.1% fee
    exchange.get_min_pair_stake_amount.return_value = 10.0
    exchange.get_max_pair_stake_amount.return_value = 100000.0
    exchange.fetch_ticker.return_value = {
        "bid": 100.0,
        "ask": 100.5,
        "last": 100.25,
    }
    return exchange


@pytest.fixture
def mock_trade():
    """
    Create a mock trade object for testing exit logic.

    Returns:
        MagicMock: Mock trade object
    """
    trade = MagicMock()
    trade.pair = "BTC/USDT"
    trade.open_rate = 100.0
    trade.open_date_utc = datetime.utcnow() - timedelta(hours=2)
    trade.stake_amount = 100.0
    trade.amount = 1.0
    trade.stop_loss = 95.0
    trade.is_open = True
    return trade


@pytest.fixture
def default_conf() -> Dict[str, Any]:
    """
    Provide a default Freqtrade configuration for testing.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "stake_currency": "USDT",
        "dry_run": True,
        "exchange": {
            "name": "binance",
            "key": "test_key",
            "secret": "test_secret",
            "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
            "pair_blacklist": [],
        },
        "max_open_trades": 3,
        "stake_amount": "unlimited",
        "tradable_balance_ratio": 0.99,
        "timeframe": "5m",
        "dry_run_wallet": 1000,
        "stoploss": -0.05,
        "trailing_stop": True,
        "trailing_stop_positive": 0.01,
    }


# ==============================================================================
# STRATEGY FIXTURES
# ==============================================================================


@pytest.fixture
def strategy_metadata() -> Dict[str, Any]:
    """
    Provide metadata dictionary for strategy testing.

    Returns:
        Dict[str, Any]: Strategy metadata
    """
    return {"pair": "BTC/USDT", "timeframe": "5m"}


@pytest.fixture
def populated_dataframe(sample_dataframe):
    """
    Provide a dataframe with basic indicators already populated.

    This is useful for testing entry/exit logic without running
    the full populate_indicators method.

    Returns:
        pd.DataFrame: Dataframe with basic indicators
    """
    df = sample_dataframe.copy()

    # Add simple indicators for testing
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_100"] = df["close"].ewm(span=100).mean()
    df["ema_200"] = df["close"].ewm(span=200).mean()
    df["rsi"] = 50  # Neutral RSI
    df["adx"] = 25  # Moderate trend
    df["volume_mean"] = df["volume"].rolling(20).mean()

    return df


# ==============================================================================
# PYTEST CONFIGURATION
# ==============================================================================


def pytest_configure(config):
    """
    Configure pytest with custom markers.
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "backtest: marks tests that run backtests")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def assert_column_exists(dataframe: pd.DataFrame, column: str) -> None:
    """
    Assert that a column exists in the dataframe.

    Args:
        dataframe: The dataframe to check
        column: The column name to verify

    Raises:
        AssertionError: If column doesn't exist
    """
    assert column in dataframe.columns, f"Column '{column}' not found in dataframe"


def assert_signal_generated(dataframe: pd.DataFrame, signal_column: str) -> None:
    """
    Assert that at least one signal was generated.

    Args:
        dataframe: The dataframe to check
        signal_column: The signal column name ('enter_long', 'exit_long', etc.)

    Raises:
        AssertionError: If no signals generated
    """
    assert signal_column in dataframe.columns, f"Signal column '{signal_column}' not found"
    assert dataframe[signal_column].sum() > 0, f"No signals generated in '{signal_column}'"


def assert_no_nan_in_column(dataframe: pd.DataFrame, column: str, skip_first: int = 0) -> None:
    """
    Assert that a column has no NaN values (except optionally the first N rows).

    Args:
        dataframe: The dataframe to check
        column: The column name to verify
        skip_first: Number of initial rows to skip (for indicators with lookback)

    Raises:
        AssertionError: If NaN values found
    """
    if skip_first > 0:
        values = dataframe[column].iloc[skip_first:]
    else:
        values = dataframe[column]

    nan_count = values.isna().sum()
    assert nan_count == 0, f"Column '{column}' has {nan_count} NaN values"
