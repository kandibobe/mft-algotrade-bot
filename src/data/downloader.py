"""
Stoic Citadel - Data Downloader
================================

Wrapper around Freqtrade's data download functionality.
Provides programmatic access to download historical data.
"""

import subprocess
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def download_data(
    pairs: List[str],
    timeframes: List[str] = ['5m', '1h'],
    exchange: str = 'binance',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config_path: str = 'user_data/config/config_backtest.json',
    data_dir: str = 'user_data/data'
) -> bool:
    """
    Download historical OHLCV data using Freqtrade.
    
    Args:
        pairs: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
        timeframes: List of timeframes (e.g., ['5m', '1h', '1d'])
        exchange: Exchange name
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        config_path: Path to config file
        data_dir: Output directory for data
        
    Returns:
        True if download successful, False otherwise
    """
    # Build timerange
    if start_date and end_date:
        timerange = f"{start_date}-{end_date}"
    elif start_date:
        timerange = f"{start_date}-"
    else:
        # Default: last 90 days
        end = datetime.now()
        start = end - timedelta(days=90)
        timerange = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    
    # Build command
    cmd = [
        'freqtrade', 'download-data',
        '--config', config_path,
        '--pairs', *pairs,
        '--timeframes', *timeframes,
        '--timerange', timerange,
        '--exchange', exchange,
        '--datadir', data_dir
    ]
    
    logger.info(f"Downloading data: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout to prevent infinite hangs
        )
        logger.info(f"Download completed successfully")
        logger.debug(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Download timed out after 300 seconds")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Freqtrade not found. Install it or use Docker.")
        return False


def download_data_docker(
    pairs: List[str],
    timeframes: List[str] = ['5m', '1h'],
    timerange: str = '20240101-20240301',
    exchange: str = 'binance'
) -> bool:
    """
    Download data using Docker Compose.
    
    This is the recommended method as it doesn't require local Freqtrade installation.
    """
    pairs_str = ' '.join(pairs)
    timeframes_str = ' '.join(timeframes)
    
    cmd = [
        'docker-compose',
        '-f', 'docker-compose.backtest.yml',
        'run', '--rm',
        '-e', f'PAIRS={pairs_str}',
        '-e', f'TIMERANGE={timerange}',
        'data-downloader'
    ]
    
    logger.info(f"Downloading data via Docker: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10 minute timeout for Docker (slower startup)
        )
        logger.info("Download completed successfully")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Docker download timed out after 600 seconds")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Docker Compose not found. Install Docker.")
        return False
