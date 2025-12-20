#!/usr/bin/env python3
"""
Script to download missing historical data for trading pairs.

This script is intended to be run in the Python virtual environment.
It checks for missing data and downloads it using the Freqtrade data download command.

Usage:
    python scripts/fix_missing_data.py --pairs BTC/USDT ETH/USDT --days 30
"""

import subprocess
import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_data(pairs: list[str], days: int, timeframe: str = "5m", exchange: str = "binance"):
    """Downloads historical data for specified pairs using Freqtrade CLI.
    """
    logger.info(f"Attempting to download data for pairs: {', '.join(pairs)} for {days} days.")
    
    # Freqtrade CLI command for data download within the Docker container
    # Assumes 'stoic_freqtrade' is the name of the Freqtrade Docker container
    command = [
        "docker", "exec", "stoic_freqtrade", "freqtrade", "download-data",
        "--exchange", exchange,
        "--timeframe", timeframe,
        "--days", str(days),
        "--erase"
    ]
    
    for pair in pairs:
        command.extend(["--pairs", pair])

    try:
        # Execute the command
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info("Data download successful.")
        logger.debug(process.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Data download failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Docker command not found. Is Docker installed and running?")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Download missing historical data for trading pairs."
    )
    parser.add_argument(
        "--pairs", 
        nargs="+", 
        required=True, 
        help="List of trading pairs (e.g., BTC/USDT ETH/USDT)"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="Number of days of historical data to download"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5m",
        help="Timeframe for the data (e.g., 5m, 1h)"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange name (e.g., binance, bybit)"
    )

    args = parser.parse_args()

    logger.info("Starting data download process...")
    if download_data(args.pairs, args.days, args.timeframe, args.exchange):
        logger.info("All specified data downloaded successfully.")
    else:
        logger.error("Failed to download all specified data.")

if __name__ == "__main__":
    main()
