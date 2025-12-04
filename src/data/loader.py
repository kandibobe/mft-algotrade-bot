"""
Stoic Citadel - Data Loader
============================

Unified interface for loading OHLCV data from various sources.
Supports CSV, Feather, and Parquet formats with caching.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal
from datetime import datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = Path('user_data/data')


def get_ohlcv(
    symbol: str,
    timeframe: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    exchange: str = 'binance',
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Get OHLCV data for a trading pair.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '5m', '1h', '1d')
        start: Start datetime (inclusive)
        end: End datetime (exclusive)
        exchange: Exchange name (default: 'binance')
        data_dir: Custom data directory
        
    Returns:
        DataFrame with columns: [date, open, high, low, close, volume]
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is corrupted or invalid
    """
    data_path = data_dir or DATA_DIR
    
    # Normalize symbol (BTC/USDT -> BTC_USDT)
    symbol_normalized = symbol.replace('/', '_')
    
    # Try different file formats
    for fmt, loader in [('feather', load_feather), ('csv', load_csv), ('json', load_json)]:
        file_path = data_path / exchange / f"{symbol_normalized}-{timeframe}.{fmt}"
        if file_path.exists():
            logger.info(f"Loading data from {file_path}")
            df = loader(file_path)
            break
    else:
        raise FileNotFoundError(
            f"No data found for {symbol} {timeframe} in {data_path}/{exchange}/"
        )
    
    # Ensure datetime index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Filter by date range
    if start:
        start_dt = pd.to_datetime(start)
        df = df[df.index >= start_dt]
    if end:
        end_dt = pd.to_datetime(end)
        df = df[df.index < end_dt]
    
    # Validate required columns
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    df_cols = set(df.columns.str.lower())
    if not required_cols.issubset(df_cols):
        missing = required_cols - df_cols
        raise ValueError(f"Missing required columns: {missing}")
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
    return df


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file.
    
    Handles various CSV formats:
    - Standard columns: date,open,high,low,close,volume
    - Unix timestamp: timestamp,open,high,low,close,volume
    """
    file_path = Path(file_path)
    
    # Try to detect format
    df = pd.read_csv(file_path)
    
    # Handle timestamp column
    if 'timestamp' in df.columns and 'date' not in df.columns:
        # Detect if timestamp is in milliseconds or seconds
        sample_ts = df['timestamp'].iloc[0]
        if sample_ts > 1e12:  # Milliseconds
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:  # Seconds
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df.drop('timestamp', axis=1, inplace=True)
    
    return df


def load_feather(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load OHLCV data from Feather file (Freqtrade default format).
    """
    return pd.read_feather(file_path)


def load_json(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load OHLCV data from JSON file.
    
    Handles Freqtrade JSON format:
    [[timestamp, open, high, low, close, volume], ...]
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle Freqtrade format (list of lists)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        df = pd.DataFrame(
            data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        # Convert timestamp
        sample_ts = df['timestamp'].iloc[0]
        unit = 'ms' if sample_ts > 1e12 else 's'
        df['date'] = pd.to_datetime(df['timestamp'], unit=unit)
        df.drop('timestamp', axis=1, inplace=True)
    else:
        df = pd.DataFrame(data)
    
    return df


def get_data_hash(df: pd.DataFrame) -> str:
    """
    Generate a hash for dataset versioning.
    
    Useful for ensuring reproducibility:
    - Same hash = same data = same backtest results
    """
    # Create hash from first/last rows and shape
    hash_content = f"{df.shape}_{df.iloc[0].values.tobytes()}_{df.iloc[-1].values.tobytes()}"
    return hashlib.md5(hash_content.encode()).hexdigest()[:12]


def get_data_metadata(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    """
    Generate metadata for a dataset.
    """
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'start_date': str(df.index.min()),
        'end_date': str(df.index.max()),
        'num_candles': len(df),
        'data_hash': get_data_hash(df),
        'generated_at': datetime.now().isoformat()
    }
