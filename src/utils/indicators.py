import numpy as np
import pandas as pd


def zscore_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate the Z-score of the close price.
    """
    df['zscore'] = (df['close'] - df['close'].rolling(window=window).mean()) / df['close'].rolling(window=window).std()
    return df

def vwma(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate the Volume Weighted Moving Average (VWMA).
    """
    df[f'vwma_{window}'] = (df['close'] * df['volume']).rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
    return df

def calculate_ema(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.DataFrame:
    """Calculate Exponential Moving Average."""
    df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
    return df

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD."""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    df['bb_upper'] = rolling_mean + (rolling_std * num_std)
    df['bb_lower'] = rolling_mean - (rolling_std * num_std)
    df['bb_middle'] = rolling_mean
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    df[f'atr_{period}'] = true_range.rolling(window=period).mean()
    return df

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate ADX (Simplified)."""
    # Requires ATR
    if f'atr_{period}' not in df.columns:
        df = calculate_atr(df, period)

    up = df['high'].diff()
    down = -df['low'].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period).mean() / df[f'atr_{period}']
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period).mean() / df[f'atr_{period}']

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df[f'adx_{period}'] = dx.ewm(alpha=1/period).mean()
    return df

def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate On-Balance Volume."""
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return df

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    return df

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate VWAP (simplified, cumulative)."""
    v = df['volume'].values
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = df.assign(vwap=(tp * v).cumsum() / v.cumsum())['vwap']
    return df

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all standard indicators."""
    df = calculate_ema(df, 20)
    df = calculate_ema(df, 50)
    df = calculate_ema(df, 200)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_atr(df)
    df = calculate_adx(df)
    df = calculate_obv(df)
    df = calculate_stochastic(df)
    df = calculate_vwap(df)
    return df
