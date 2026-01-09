"""
Market Data Downloader
======================

Downloads historical OHLCV data from exchanges.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from src.config import config

logger = logging.getLogger(__name__)


def download_historical_data(
    pairs: list[str],
    timeframe: str,
    days: int = 30,
    exchange: str = "binance",
    data_dir: str | None = None,
    config_path: str | None = None,
) -> bool:
    """
    Download data using Freqtrade CLI.

    Args:
        pairs: List of pairs
        timeframe: Candle timeframe
        days: Number of days to download
        exchange: Exchange name
        data_dir: Output directory
        config_path: Path to freqtrade config

    Returns:
        True if successful
    """
    cfg = config()
    actual_data_dir = Path(data_dir or cfg.paths.data_dir)
    actual_config_path = config_path or str(cfg.paths.user_data_dir / "config/config_backtest.json")

    cmd = [
        "freqtrade",
        "download-data",
        "--exchange",
        exchange,
        "--pairs",
    ]
    cmd.extend(pairs)
    cmd.extend(
        [
            "--timeframes",
            timeframe,
            "--days",
            str(days),
            "--datadir",
            str(actual_data_dir),
            "--config",
            actual_config_path,
        ]
    )

    logger.info(f"Downloading data for {len(pairs)} pairs...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        return False


# Alias for backward compatibility
download_data = download_historical_data


# Alias for backward compatibility
download_data = download_historical_data
