"""
Tests for technical indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_stochastic,
    calculate_all_indicators
)


# Generate test data
@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    
    # Generate random walk prices
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    dates = pd.date_range('2024-01-01', periods=n, freq='5min')
    
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


class TestEMA:
    """Tests for EMA calculation."""
    
    def test_ema_basic(self, sample_ohlcv):
        """Test basic EMA calculation."""
        ema = calculate_ema(sample_ohlcv['close'], period=20)
        
        assert len(ema) == len(sample_ohlcv)
        assert not ema.isna().all()
    
    def test_ema_periods(self, sample_ohlcv):
        """Test that longer period EMA is smoother."""
        ema_short = calculate_ema(sample_ohlcv['close'], period=10)
        ema_long = calculate_ema(sample_ohlcv['close'], period=50)
        
        # Longer EMA should have smaller variance
        assert ema_long.std() < ema_short.std()
    
    def test_ema_values_reasonable(self, sample_ohlcv):
        """Test that EMA values are within reasonable range."""
        ema = calculate_ema(sample_ohlcv['close'], period=20)
        close = sample_ohlcv['close']
        
        # EMA should be within price range (approximately)
        assert ema.min() > close.min() * 0.9
        assert ema.max() < close.max() * 1.1


class TestRSI:
    """Tests for RSI calculation."""
    
    def test_rsi_range(self, sample_ohlcv):
        """Test that RSI is within 0-100 range."""
        rsi = calculate_rsi(sample_ohlcv['close'], period=14)
        
        # Skip NaN values
        rsi_valid = rsi.dropna()
        
        assert rsi_valid.min() >= 0
        assert rsi_valid.max() <= 100
    
    def test_rsi_mean_around_50(self, sample_ohlcv):
        """Test that RSI mean is approximately 50 for random walk."""
        rsi = calculate_rsi(sample_ohlcv['close'], period=14)
        
        # For random walk, RSI should be around 50
        assert 30 < rsi.mean() < 70
    
    def test_rsi_trending_up_high(self):
        """Test that RSI is high for consistently rising prices with noise."""
        np.random.seed(42)
        n = 100
        # Create strongly rising prices with small noise
        trend = np.linspace(0, 50, n)
        noise = np.random.randn(n) * 0.5
        rising_prices = pd.Series(100 + trend + noise)
        
        rsi = calculate_rsi(rising_prices, period=14)
        
        # RSI should be above 50 for uptrend (last few values)
        assert rsi.iloc[-10:].mean() > 50
    
    def test_rsi_trending_down_low(self):
        """Test that RSI is low for consistently falling prices with noise."""
        np.random.seed(42)
        n = 100
        # Create strongly falling prices with small noise
        trend = np.linspace(0, -50, n)
        noise = np.random.randn(n) * 0.5
        falling_prices = pd.Series(100 + trend + noise)
        
        rsi = calculate_rsi(falling_prices, period=14)
        
        # RSI should be below 50 for downtrend (last few values)
        assert rsi.iloc[-10:].mean() < 50


class TestMACD:
    """Tests for MACD calculation."""
    
    def test_macd_returns_dict(self, sample_ohlcv):
        """Test that MACD returns expected keys."""
        macd = calculate_macd(sample_ohlcv['close'])
        
        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
    
    def test_macd_histogram_relationship(self, sample_ohlcv):
        """Test that histogram = macd - signal."""
        macd = calculate_macd(sample_ohlcv['close'])
        
        expected_hist = macd['macd'] - macd['signal']
        
        # Should be equal (within floating point tolerance)
        np.testing.assert_array_almost_equal(
            macd['histogram'].values,
            expected_hist.values
        )
    
    def test_macd_lengths(self, sample_ohlcv):
        """Test that all MACD components have same length."""
        macd = calculate_macd(sample_ohlcv['close'])
        
        assert len(macd['macd']) == len(sample_ohlcv)
        assert len(macd['signal']) == len(sample_ohlcv)
        assert len(macd['histogram']) == len(sample_ohlcv)


class TestATR:
    """Tests for ATR calculation."""
    
    def test_atr_positive(self, sample_ohlcv):
        """Test that ATR is always positive."""
        atr = calculate_atr(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        atr_valid = atr.dropna()
        assert (atr_valid >= 0).all()
    
    def test_atr_reasonable_range(self, sample_ohlcv):
        """Test that ATR is in reasonable range relative to price."""
        atr = calculate_atr(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        close = sample_ohlcv['close']
        
        # ATR as percentage of price should be small
        atr_pct = atr / close
        assert atr_pct.mean() < 0.1  # Less than 10% on average


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""
    
    def test_bb_returns_dict(self, sample_ohlcv):
        """Test that BB returns expected keys."""
        bb = calculate_bollinger_bands(sample_ohlcv['close'])
        
        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        assert 'width' in bb
    
    def test_bb_relationships(self, sample_ohlcv):
        """Test band relationships: upper > middle > lower."""
        bb = calculate_bollinger_bands(sample_ohlcv['close'])
        
        # After warmup period
        valid = ~bb['upper'].isna()
        
        assert (bb['upper'][valid] >= bb['middle'][valid]).all()
        assert (bb['middle'][valid] >= bb['lower'][valid]).all()
    
    def test_bb_middle_is_sma(self, sample_ohlcv):
        """Test that middle band is SMA."""
        bb = calculate_bollinger_bands(sample_ohlcv['close'], period=20)
        sma = sample_ohlcv['close'].rolling(20).mean()
        
        np.testing.assert_array_almost_equal(
            bb['middle'].dropna().values,
            sma.dropna().values
        )


class TestStochastic:
    """Tests for Stochastic oscillator calculation."""
    
    def test_stoch_range(self, sample_ohlcv):
        """Test that Stochastic is within 0-100 range."""
        stoch = calculate_stochastic(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        k_valid = stoch['k'].dropna()
        d_valid = stoch['d'].dropna()
        
        assert k_valid.min() >= 0
        assert k_valid.max() <= 100
        assert d_valid.min() >= 0
        assert d_valid.max() <= 100
    
    def test_stoch_d_is_smoothed_k(self, sample_ohlcv):
        """Test that %D is smoothed %K."""
        stoch = calculate_stochastic(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close'],
            d_period=3
        )
        
        # %D should be less volatile than %K
        assert stoch['d'].std() < stoch['k'].std()


class TestCalculateAllIndicators:
    """Tests for combined indicator calculation."""
    
    def test_adds_expected_columns(self, sample_ohlcv):
        """Test that all expected indicator columns are added."""
        result = calculate_all_indicators(sample_ohlcv)
        
        expected_cols = [
            'ema_9', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'atr', 'bb_upper', 'bb_middle', 'bb_lower',
            'stoch_k', 'stoch_d'
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_preserves_original_columns(self, sample_ohlcv):
        """Test that original OHLCV columns are preserved."""
        result = calculate_all_indicators(sample_ohlcv)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns
    
    def test_no_all_nan_columns(self, sample_ohlcv):
        """Test that no indicator column is all NaN."""
        result = calculate_all_indicators(sample_ohlcv)
        
        for col in result.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                # Should have some valid values
                assert not result[col].isna().all(), f"Column {col} is all NaN"
