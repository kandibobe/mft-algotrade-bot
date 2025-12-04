"""
Stoic Citadel - Technical Indicators Library
=============================================

Vectorized implementations of common technical indicators.
All functions operate on pandas Series/DataFrames for efficiency.

Usage:
    from src.utils.indicators import calculate_rsi, calculate_macd
    
    df['rsi'] = calculate_rsi(df['close'], period=14)
    macd_data = calculate_macd(df['close'])
    df['macd'] = macd_data['macd']
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_ema(
    series: pd.Series,
    period: int = 20,
    adjust: bool = False
) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series (typically close)
        period: EMA period
        adjust: Adjust for gaps (default False for speed)
        
    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=adjust).mean()


def calculate_sma(
    series: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        series: Price series
        period: SMA period
        
    Returns:
        SMA series
    """
    return series.rolling(window=period).mean()


def calculate_rsi(
    series: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Vectorized implementation using exponential weighted moving average.
    
    Args:
        series: Price series (typically close)
        period: RSI period (default 14)
        
    Returns:
        RSI series (0-100 scale)
    """
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Calculate average gain/loss using EWM (Wilder's smoothing)
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)  # Fill NaN with neutral value


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        series: Price series
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
        
    Returns:
        Dict with keys: 'macd', 'signal', 'histogram'
    """
    # Calculate EMAs
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)
    
    # MACD line
    macd = ema_fast - ema_slow
    
    # Signal line
    signal = calculate_ema(macd, signal_period)
    
    # Histogram
    histogram = macd - signal
    
    return {
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period
        
    Returns:
        ATR series
    """
    # Previous close
    prev_close = close.shift(1)
    
    # True Range components
    tr1 = high - low  # Current high-low
    tr2 = (high - prev_close).abs()  # High - Previous close
    tr3 = (low - prev_close).abs()  # Low - Previous close
    
    # True Range is max of all three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR is smoothed TR (Wilder's smoothing)
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return atr


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        series: Price series (typically close)
        period: Moving average period
        num_std: Number of standard deviations
        
    Returns:
        Dict with keys: 'upper', 'middle', 'lower', 'width', 'percent_b'
    """
    # Middle band (SMA)
    middle = series.rolling(window=period).mean()
    
    # Standard deviation
    std = series.rolling(window=period).std()
    
    # Upper and lower bands
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # Bandwidth (volatility measure)
    width = (upper - lower) / middle
    
    # %B (position within bands, 0=lower, 1=upper)
    percent_b = (series - lower) / (upper - lower)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width,
        'percent_b': percent_b
    }


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        k_period: %K period (default 14)
        d_period: %D smoothing period (default 3)
        
    Returns:
        Dict with keys: 'k' (fast), 'd' (slow)
    """
    # Highest high and lowest low over period
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # %K calculation
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    
    # %D is smoothed %K
    d = k.rolling(window=d_period).mean()
    
    return {
        'k': k.fillna(50),
        'd': d.fillna(50)
    }


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    
    Note: This is cumulative VWAP (from start of data).
    For session VWAP, data should be filtered to session.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        
    Returns:
        VWAP series
    """
    # Typical price
    typical_price = (high + low + close) / 3
    
    # Cumulative volume-price and volume
    cum_vp = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    
    # VWAP
    vwap = cum_vp / cum_vol.replace(0, np.nan)
    
    return vwap


def calculate_obv(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        OBV series
    """
    # Price direction
    direction = np.sign(close.diff())
    
    # OBV: cumulative sum of signed volume
    obv = (direction * volume).cumsum()
    
    return obv


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Dict[str, pd.Series]:
    """
    Calculate Average Directional Index.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ADX period
        
    Returns:
        Dict with keys: 'adx', 'plus_di', 'minus_di'
    """
    # True Range
    atr = calculate_atr(high, low, close, period)
    
    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    
    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Smooth DM
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)
    
    # DX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    
    # ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return {
        'adx': adx.fillna(0),
        'plus_di': plus_di.fillna(0),
        'minus_di': minus_di.fillna(0)
    }


def calculate_all_indicators(
    df: pd.DataFrame,
    include_advanced: bool = True
) -> pd.DataFrame:
    """
    Calculate all standard indicators and add to dataframe.
    
    Expects DataFrame with columns: open, high, low, close, volume
    
    Args:
        df: OHLCV DataFrame
        include_advanced: Include ADX, VWAP, OBV (default True)
        
    Returns:
        DataFrame with indicator columns added
    """
    result = df.copy()
    
    # Standardize column names
    result.columns = result.columns.str.lower()
    
    # EMAs
    result['ema_9'] = calculate_ema(result['close'], 9)
    result['ema_21'] = calculate_ema(result['close'], 21)
    result['ema_50'] = calculate_ema(result['close'], 50)
    result['ema_100'] = calculate_ema(result['close'], 100)
    result['ema_200'] = calculate_ema(result['close'], 200)
    
    # RSI
    result['rsi'] = calculate_rsi(result['close'], 14)
    
    # MACD
    macd = calculate_macd(result['close'])
    result['macd'] = macd['macd']
    result['macd_signal'] = macd['signal']
    result['macd_hist'] = macd['histogram']
    
    # ATR
    result['atr'] = calculate_atr(
        result['high'], result['low'], result['close'], 14
    )
    
    # Bollinger Bands
    bb = calculate_bollinger_bands(result['close'])
    result['bb_upper'] = bb['upper']
    result['bb_middle'] = bb['middle']
    result['bb_lower'] = bb['lower']
    result['bb_width'] = bb['width']
    result['bb_percent_b'] = bb['percent_b']
    
    # Stochastic
    stoch = calculate_stochastic(
        result['high'], result['low'], result['close']
    )
    result['stoch_k'] = stoch['k']
    result['stoch_d'] = stoch['d']
    
    if include_advanced:
        # ADX
        adx = calculate_adx(
            result['high'], result['low'], result['close']
        )
        result['adx'] = adx['adx']
        result['plus_di'] = adx['plus_di']
        result['minus_di'] = adx['minus_di']
        
        # VWAP
        result['vwap'] = calculate_vwap(
            result['high'], result['low'], result['close'], result['volume']
        )
        
        # OBV
        result['obv'] = calculate_obv(result['close'], result['volume'])
    
    # Volume SMA
    result['volume_sma'] = calculate_sma(result['volume'], 20)
    
    logger.info(f"Calculated {len(result.columns) - len(df.columns)} indicators")
    
    return result
