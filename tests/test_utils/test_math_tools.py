
import numpy as np
import pandas as pd

from src.utils.math_tools import calculate_efficiency_ratio, calculate_hurst


def test_hurst_random_walk():
    """Test Hurst on random walk (should be approx 0.5)."""
    np.random.seed(42)
    # Generate geometric brownian motion / random walk
    returns = np.random.normal(0, 0.01, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    series = pd.Series(prices)

    hurst = calculate_hurst(series, window=100)

    # Last value should be defined
    assert not pd.isna(hurst.iloc[-1])

    # Check average Hurst of random walk is close to 0.5
    # It fluctuates, but should be within 0.4 - 0.6 range mostly
    mean_hurst = hurst.mean()
    assert 0.4 < mean_hurst < 0.6

def test_hurst_trending():
    """Test Hurst on persistent data (AR(1) with pos corr)."""
    np.random.seed(42)
    n = 1000
    returns = np.zeros(n)
    for i in range(1, n):
        # Strong autocorrelation (0.8) -> Trend persistence
        returns[i] = 0.8 * returns[i-1] + np.random.normal(0, 0.01)

    prices = 100 * np.exp(np.cumsum(returns))
    series = pd.Series(prices)

    hurst = calculate_hurst(series, window=100)

    # Should be > 0.5
    # Note: Short window R/S is biased, but should detect strong persistence
    assert hurst.mean() > 0.55

def test_hurst_mean_reverting():
    """Test Hurst on mean reverting data (AR(1) with neg corr)."""
    np.random.seed(42)
    n = 1000
    returns = np.zeros(n)
    for i in range(1, n):
        # Strong anti-correlation (-0.5) -> Mean Reversion
        returns[i] = -0.5 * returns[i-1] + np.random.normal(0, 0.01)

    prices = 100 * np.exp(np.cumsum(returns))
    series = pd.Series(prices)

    hurst = calculate_hurst(series, window=100)

    # Should be < 0.5
    assert hurst.mean() < 0.45

def test_efficiency_ratio():
    """Test Efficiency Ratio."""
    # Perfect trend
    prices = pd.Series(range(100))
    er = calculate_efficiency_ratio(prices, window=10)
    # ER should be 1.0
    assert np.allclose(er.iloc[-1], 1.0)

    # Chop
    prices_chop = pd.Series([10, 11, 10, 11, 10, 11, 10, 11, 10, 11])
    er_chop = calculate_efficiency_ratio(prices_chop, window=5)
    # Change = 0 or 1, Volatility = 5
    # ER should be low
    assert er_chop.iloc[-1] < 0.5
