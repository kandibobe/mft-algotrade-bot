"""
Data Loader and Converter
=========================

Handles loading historical data from various formats (JSON, Feather, Parquet).
Provides unified access to market data for backtesting and ML training.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)

class DataLoader:
    """Unified data loader for trading data."""

    def __init__(self, data_dir: str | None = None):
        cfg = config()
        self.data_dir = Path(data_dir or cfg.paths.data_dir)

    def load_pair_data(
        self,
        pair: str,
        timeframe: str,
        exchange: str = "binance",
        format: str = "feather",
    ) -> pd.DataFrame:
        """
        Load historical data for a pair.

        Args:
            pair: Trading pair (e.g. "BTC/USDT")
            timeframe: Candle timeframe
            exchange: Exchange name
            format: Data format (feather, parquet, json)

        Returns:
            DataFrame with OHLCV data
        """
        pair_slug = pair.replace("/", "_")
        file_path = self.data_dir / exchange / f"{pair_slug}-{timeframe}.{format}"

        if not file_path.exists():
            # Try without exchange subfolder for generic data
            file_path = self.data_dir / f"{pair_slug}-{timeframe}.{format}"
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        if format == "feather":
            df = pd.read_feather(file_path)
        elif format == "parquet":
            df = pd.read_parquet(file_path)
        elif format == "json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return df


def get_ohlcv(pair: str, timeframe: str, exchange: str = "binance", format: str = "feather") -> pd.DataFrame:
    """
    Standalone function to get OHLCV data.
    Wraps DataLoader for backward compatibility.
    """
    loader = DataLoader()
    return loader.load_pair_data(pair, timeframe, exchange, format)


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(path)


def load_feather(path: str) -> pd.DataFrame:
    """Load Feather data."""
    return pd.read_feather(path)


def get_data_hash(df: pd.DataFrame) -> str:
    """Get hash of dataframe content."""
    import hashlib
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


def get_data_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Get metadata about dataframe."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "start_date": df.index.min() if not df.empty else None,
        "end_date": df.index.max() if not df.empty else None,
    }


def get_ohlcv(pair: str, timeframe: str, exchange: str = "binance", format: str = "feather") -> pd.DataFrame:
    """
    Standalone function to get OHLCV data.
    Wraps DataLoader for backward compatibility.
    """
    loader = DataLoader()
    return loader.load_pair_data(pair, timeframe, exchange, format)


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(path)


def load_feather(path: str) -> pd.DataFrame:
    """Load Feather data."""
    return pd.read_feather(path)
