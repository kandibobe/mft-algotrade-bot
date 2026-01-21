"""
Freqtrade Data to Parquet Converter
===================================

Converts Freqtrade Feather data to Parquet format for Feature Store.
Handles recursive directory scanning and efficient columnar storage.
"""

import logging
from pathlib import Path

import pandas as pd
import click

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@click.command()
@click.option("--data-dir", default="user_data/data/binance", help="Source directory containing Freqtrade data")
@click.option("--output-dir", default="user_data/data/parquet", help="Destination directory for Parquet files")
@click.option("--timeframe", default="1m", help="Timeframe to process (e.g., 1m, 5m, 1h)")
def convert_data(data_dir: str, output_dir: str, timeframe: str):
    """
    Convert Freqtrade Feather data to Parquet format.
    """
    source_path = Path(data_dir)
    dest_path = Path(output_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting conversion from {source_path} to {dest_path}")

    # Pattern match for timeframe specific files
    pattern = f"*-{timeframe}.feather"
    files = list(source_path.glob(pattern))

    if not files:
        logger.warning(f"No files found matching pattern {pattern} in {source_path}")
        return

    logger.info(f"Found {len(files)} files to convert.")

    for file_path in files:
        try:
            symbol = file_path.name.replace(f"-{timeframe}.feather", "")
            
            # Read Data
            df = pd.read_feather(file_path)

            if df.empty:
                logger.warning(f"Empty data in {file_path.name}")
                continue

            # Ensure correct types
            if "date" in df.columns:
                df.rename(columns={"date": "timestamp"}, inplace=True)
            
            if "timestamp" not in df.columns:
                logger.error(f"Timestamp column missing in {file_path.name}")
                continue

            # Add metadata columns
            df["symbol"] = symbol
            df["timeframe"] = timeframe

            # Save as Parquet
            output_file = dest_path / f"{symbol}_{timeframe}.parquet"
            df.to_parquet(output_file, compression="snappy", index=False)
            
            logger.info(f"✅ Converted {symbol} ({len(df)} rows) -> {output_file.name}")

        except Exception as e:
            logger.error(f"❌ Failed to convert {file_path.name}: {e}")

    logger.info("Conversion complete.")

if __name__ == "__main__":
    convert_data()