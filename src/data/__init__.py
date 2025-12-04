"""
Stoic Citadel - Data Pipeline Module
=====================================

Provides unified interface for:
- Loading OHLCV data (CSV, Feather, JSON)
- Downloading historical data
- Data validation and integrity checks
- Caching and versioning

Usage:
    from src.data import get_ohlcv, download_data
    
    df = get_ohlcv('BTC/USDT', '5m', start='2024-01-01', end='2024-02-01')
"""

from .loader import get_ohlcv, load_csv, load_feather
from .downloader import download_data
from .validator import validate_ohlcv, check_data_integrity

__all__ = [
    'get_ohlcv',
    'load_csv',
    'load_feather',
    'download_data',
    'validate_ohlcv',
    'check_data_integrity'
]

__version__ = '1.0.0'
