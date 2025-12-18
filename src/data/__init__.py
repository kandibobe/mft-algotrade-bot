"""
Stoic Citadel - Data Pipeline Module
=====================================

Provides unified interface for:
- Loading OHLCV data (CSV, Feather, JSON)
- Downloading historical data
- Async data fetching (ccxt.async_support)
- Data validation and integrity checks
- Caching and versioning

Usage:
    # Sync loading from files
    from src.data import get_ohlcv, download_data
    df = get_ohlcv('BTC/USDT', '5m', start='2024-01-01', end='2024-02-01')

    # Async fetching from exchange
    from src.data import AsyncDataFetcher
    async with AsyncDataFetcher() as fetcher:
        df = await fetcher.fetch_ohlcv('BTC/USDT', '1h')
"""

from .loader import get_ohlcv, load_csv, load_feather
from .downloader import download_data
from .validator import validate_ohlcv, check_data_integrity

# Lazy import for async fetcher (requires ccxt)
def __getattr__(name):
    if name in ("AsyncDataFetcher", "AsyncOrderExecutor", "FetcherConfig", "fetch_ohlcv_async"):
        from .async_fetcher import (
            AsyncDataFetcher,
            AsyncOrderExecutor,
            FetcherConfig,
            fetch_ohlcv_async,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'get_ohlcv',
    'load_csv',
    'load_feather',
    'download_data',
    'validate_ohlcv',
    'check_data_integrity',
    'AsyncDataFetcher',
    'AsyncOrderExecutor',
    'FetcherConfig',
    'fetch_ohlcv_async',
]

__version__ = '2.0.0'
