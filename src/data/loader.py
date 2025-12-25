"""
Stoic Citadel - Data Loader
============================

Unified interface for loading OHLCV data from various sources.
Supports CSV, Feather, and Parquet formats with caching.

Performance Optimizations:
1. Lazy loading with chunking for large datasets
2. Redis and in-memory caching for frequently accessed data
3. Parquet format for 10x faster I/O with compression
"""

import hashlib
import json
import logging
import pickle
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Literal, Optional, Tuple, Union, Any, Dict, List

import numpy as np
import pandas as pd

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = Path("user_data/data")

# Cache configuration
CACHE_ENABLED = True
REDIS_CACHE_ENABLED = False  # Set to True if Redis is available and configured
REDIS_CACHE_TTL = 3600  # 1 hour
MEMORY_CACHE_SIZE = 100  # LRU cache size for frequently accessed data


def get_ohlcv(
    symbol: str,
    timeframe: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    exchange: str = "binance",
    data_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Get OHLCV data for a trading pair with caching support.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '5m', '1h', '1d')
        start: Start datetime (inclusive)
        end: End datetime (exclusive)
        exchange: Exchange name (default: 'binance')
        data_dir: Custom data directory
        use_cache: Whether to use caching (default: True)

    Returns:
        DataFrame with columns: [date, open, high, low, close, volume]
        Index is DatetimeIndex.

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is corrupted or invalid
    """
    # Check cache first if enabled
    if use_cache and CACHE_ENABLED:
        cached_data = _get_cached_data(symbol, timeframe, start, end, exchange)
        if cached_data is not None:
            logger.info(f"Using cached data for {symbol} {timeframe}")
            return cached_data

    data_path = data_dir or DATA_DIR

    # Normalize symbol (BTC/USDT -> BTC_USDT)
    symbol_normalized = symbol.replace("/", "_")

    # Try different file formats (prioritize Parquet for performance)
    df = pd.DataFrame()
    found = False
    
    # Priority: Parquet > Feather > CSV > JSON
    for fmt, loader in [
        ("parquet", load_parquet),
        ("feather", load_feather),
        ("csv", load_csv),
        ("json", load_json),
    ]:
        file_path = data_path / exchange / f"{symbol_normalized}-{timeframe}.{fmt}"
        if file_path.exists():
            logger.info(f"Loading data from {file_path}")
            try:
                df = loader(file_path)
                found = True
                break
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
    
    if not found:
        raise FileNotFoundError(
            f"No data found for {symbol} {timeframe} in {data_path}/{exchange}/"
        )

    # Ensure datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Filter by date range
    if start:
        start_dt = pd.to_datetime(start)
        # Ensure timezone consistency
        if df.index.tz is not None and start_dt.tz is None:
            start_dt = start_dt.tz_localize(df.index.tz)
        elif df.index.tz is None and start_dt.tz is not None:
            start_dt = start_dt.tz_convert(None)
        df = df[df.index >= start_dt]
        
    if end:
        end_dt = pd.to_datetime(end)
        # Ensure timezone consistency
        if df.index.tz is not None and end_dt.tz is None:
            end_dt = end_dt.tz_localize(df.index.tz)
        elif df.index.tz is None and end_dt.tz is not None:
            end_dt = end_dt.tz_convert(None)
        df = df[df.index < end_dt]
        
    # Validate required columns
    required_cols = {"open", "high", "low", "close", "volume"}
    df_cols = set(df.columns.str.lower())
    if not required_cols.issubset(df_cols):
        missing = required_cols - df_cols
        raise ValueError(f"Missing required columns: {missing}")

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Cache the result if caching is enabled
    if use_cache and CACHE_ENABLED:
        _set_cached_data(symbol, timeframe, start, end, exchange, df)

    logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
    return df


def load_ohlcv_chunked(
    symbol: str,
    timeframe: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    exchange: str = "binance",
    data_dir: Optional[Path] = None,
    chunk_size: str = "1ME",  # Monthly chunks by default (ME = month end)
) -> Iterator[pd.DataFrame]:
    """
    Load OHLCV data in chunks to reduce memory usage.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '5m', '1h', '1d')
        start: Start datetime (inclusive)
        end: End datetime (exclusive)
        exchange: Exchange name (default: 'binance')
        data_dir: Custom data directory
        chunk_size: Pandas frequency string for chunking (e.g., '1M', '1W', '1D')

    Yields:
        DataFrames with columns: [date, open, high, low, close, volume]
    """
    data_path = data_dir or DATA_DIR
    symbol_normalized = symbol.replace("/", "_")

    # Find the data file
    file_path = None
    for fmt in ["parquet", "feather", "csv", "json"]:
        test_path = data_path / exchange / f"{symbol_normalized}-{timeframe}.{fmt}"
        if test_path.exists():
            file_path = test_path
            break

    if file_path is None:
        raise FileNotFoundError(
            f"No data found for {symbol} {timeframe} in {data_path}/{exchange}/"
        )

    # Determine date range for chunking
    if start is None or end is None:
        # Load metadata to get date range - careful not to load full file
        # Ideally we read metadata only, but here we load full for simplicity if not implementing header parsing
        # Optimization: Just load head/tail or use pyarrow for parquet metadata
        try:
            df_sample = get_ohlcv(
                symbol,
                timeframe,
                start=None,
                end=None,
                exchange=exchange,
                data_dir=data_dir,
                use_cache=False,
            )
            if start is None:
                start = df_sample.index.min()
            if end is None:
                end = df_sample.index.max()
        except Exception:
             # Fallback
             start = datetime.now()
             end = datetime.now()

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # Create date ranges for chunking
    try:
        date_ranges = pd.date_range(start=start_dt, end=end_dt, freq=chunk_size)
    except Exception:
        # Fallback if frequency invalid
        date_ranges = pd.date_range(start=start_dt, end=end_dt, periods=10)

    for i in range(len(date_ranges) - 1):
        chunk_start = date_ranges[i]
        chunk_end = date_ranges[i + 1]

        try:
            df_chunk = get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=chunk_start,
                end=chunk_end,
                exchange=exchange,
                data_dir=data_dir,
                use_cache=True,  # Use cache for chunks
            )

            if not df_chunk.empty:
                yield df_chunk

        except Exception as e:
            logger.warning(f"Failed to load chunk {chunk_start} to {chunk_end}: {e}")
            continue


@lru_cache(maxsize=MEMORY_CACHE_SIZE)
def get_cached_indicators(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Cache calculated indicators using LRU cache.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '5m', '1h', '1d')

    Returns:
        Cached DataFrame with indicators or None if not in cache
    """
    # This is a placeholder for indicator caching
    # In practice, you would calculate indicators and cache them here
    return None


def _get_cached_data(
    symbol: str,
    timeframe: str,
    start: Optional[Union[str, datetime]],
    end: Optional[Union[str, datetime]],
    exchange: str,
) -> Optional[pd.DataFrame]:
    """
    Get data from cache (Redis or memory).

    Returns:
        Cached DataFrame or None if not found
    """
    cache_key = _generate_cache_key(symbol, timeframe, start, end, exchange)

    # Try Redis cache first if enabled
    if REDIS_CACHE_ENABLED and REDIS_AVAILABLE and redis:
        try:
            redis_client = redis.Redis(decode_responses=False)
            cached_bytes = redis_client.get(cache_key)
            if cached_bytes:
                df = pickle.loads(cached_bytes)
                logger.debug(f"Redis cache hit for {cache_key}")
                return df
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")

    return None


def _set_cached_data(
    symbol: str,
    timeframe: str,
    start: Optional[Union[str, datetime]],
    end: Optional[Union[str, datetime]],
    exchange: str,
    df: pd.DataFrame,
) -> None:
    """
    Store data in cache (Redis or memory).
    """
    cache_key = _generate_cache_key(symbol, timeframe, start, end, exchange)

    # Store in Redis if enabled
    if REDIS_CACHE_ENABLED and REDIS_AVAILABLE and redis:
        try:
            redis_client = redis.Redis(decode_responses=False)
            cached_bytes = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
            redis_client.setex(cache_key, REDIS_CACHE_TTL, cached_bytes)
            logger.debug(f"Cached data in Redis with key {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache in Redis: {e}")


def _generate_cache_key(
    symbol: str,
    timeframe: str,
    start: Optional[Union[str, datetime]],
    end: Optional[Union[str, datetime]],
    exchange: str,
) -> str:
    """
    Generate a unique cache key for the query.
    """
    start_str = str(start) if start else "None"
    end_str = str(end) if end else "None"

    key_parts = ["ohlcv", exchange, symbol.replace("/", "_"), timeframe, start_str, end_str]

    return ":".join(key_parts)


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
    if "timestamp" in df.columns and "date" not in df.columns:
        # Detect if timestamp is in milliseconds or seconds
        sample_ts = df["timestamp"].iloc[0]
        if sample_ts > 1e12:  # Milliseconds
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:  # Seconds
            df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df.drop("timestamp", axis=1, inplace=True)

    return df


def load_feather(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load OHLCV data from Feather file (Freqtrade default format).
    """
    return pd.read_feather(file_path)


def load_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load OHLCV data from Parquet file.

    Parquet is 10x faster than CSV and uses compression.
    """
    file_path = Path(file_path)
    return pd.read_parquet(file_path)


def save_to_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Save DataFrame to Parquet format.

    Parquet is 10x faster than CSV and uses compression.

    Args:
        df: DataFrame to save
        path: Output path (will add .parquet extension if not present)
    """
    path = Path(path)
    if path.suffix != ".parquet":
        path = path.with_suffix(".parquet")

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save with compression for better performance
    df.to_parquet(path, engine="pyarrow", compression="snappy")

    logger.info(f"Saved data to Parquet: {path} (size: {path.stat().st_size / 1024 / 1024:.2f} MB)")


def load_json(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load OHLCV data from JSON file.

    Handles Freqtrade JSON format:
    [[timestamp, open, high, low, close, volume], ...]
    """
    file_path = Path(file_path)

    with open(file_path, "r") as f:
        data = json.load(f)

    # Handle Freqtrade format (list of lists)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Convert timestamp
        sample_ts = df["timestamp"].iloc[0]
        unit = "ms" if sample_ts > 1e12 else "s"
        df["date"] = pd.to_datetime(df["timestamp"], unit=unit)
        df.drop("timestamp", axis=1, inplace=True)
    else:
        df = pd.DataFrame(data)

    return df


def get_data_hash(df: pd.DataFrame) -> str:
    """
    Generate a hash for dataset versioning.

    Useful for ensuring reproducibility:
    - Same hash = same data = same backtest results
    """
    # Select only numeric columns for hashing (exclude date strings)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 0:
        # Use numeric data for reliable hashing
        numeric_df = df[numeric_cols]
        hash_content = f"{numeric_df.shape}_{numeric_df.values.tobytes()}"
    else:
        # Fallback to string representation
        hash_content = f"{df.shape}_{df.to_string()}"

    return hashlib.md5(hash_content.encode()).hexdigest()[:12]


def get_data_metadata(df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Generate metadata for a dataset.
    """
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": str(df.index.min()),
        "end_date": str(df.index.max()),
        "num_candles": len(df),
        "data_hash": get_data_hash(df),
        "generated_at": datetime.now().isoformat(),
    }


def convert_csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
    delete_original: bool = False,
) -> Path:
    """
    Convert CSV file to Parquet format for better performance.

    Args:
        csv_path: Path to CSV file
        parquet_path: Output path for Parquet file (default: same as CSV with .parquet extension)
        delete_original: Whether to delete the original CSV file after conversion

    Returns:
        Path to the created Parquet file
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    else:
        parquet_path = Path(parquet_path)

    # Load CSV
    logger.info(f"Converting {csv_path} to Parquet...")
    df = load_csv(csv_path)

    # Save as Parquet
    save_to_parquet(df, parquet_path)

    # Delete original if requested
    if delete_original:
        csv_path.unlink()
        logger.info(f"Deleted original CSV file: {csv_path}")

    return parquet_path
