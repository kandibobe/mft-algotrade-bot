"""
Freqtrade Data Converter
========================

Converts Freqtrade JSON/H5 data to Feather format for high-performance loading.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)


def convert_freqtrade_data(data_dir: str | None = None):
    """
    Scan data directory and convert all .json files to .feather.
    """
    cfg = config()
    actual_data_dir = Path(data_dir or cfg.paths.data_dir / "binance")

    if not actual_data_dir.exists():
        logger.error(f"Data directory not found: {actual_data_dir}")
        return

    json_files = list(actual_data_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to convert in {actual_data_dir}")

    for json_file in json_files:
        try:
            feather_file = json_file.with_suffix(".feather")
            if feather_file.exists():
                continue

            logger.info(f"Converting {json_file.name}...")
            df = pd.read_json(json_file)

            # Freqtrade format usually needs some renaming/sorting
            # but we assume standard format for now
            df.to_feather(feather_file)
        except Exception as e:
            logger.error(f"Failed to convert {json_file.name}: {e}")
