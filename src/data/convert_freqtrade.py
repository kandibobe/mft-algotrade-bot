import os
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("data_converter")

def convert_freqtrade_data(data_dir: str = "user_data/data/binance"):
    """
    Convert Freqtrade .json and .json.gz data files to .feather format.
    
    Args:
        data_dir: Directory containing Freqtrade .json/.json.gz data files.
    """
    path = Path(data_dir)
    # Print Absolute Path
    logger.info(f"Looking for data in (absolute path): {os.path.abspath(data_dir)}")

    if not path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    logger.info(f"Scanning for files in {data_dir} (recursive)...")
    
    files = []
    # Recursive scan
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            file_path = Path(root) / filename
            logger.info(f"Found file: {file_path}") # Log every found file
            
            # Relaxed filter
            if filename.endswith(".json") or filename.endswith(".json.gz"):
                 files.append(file_path)

    if not files:
        logger.warning(f"No .json or .json.gz files found in {data_dir}")
        return

    success_count = 0
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    for file_path in files:
        try:
            logger.info(f"Converting {file_path.name}...")
            
            # Load data
            if file_path.name.endswith('.gz'):
                df = pd.read_json(file_path, compression='gzip')
            else:
                df = pd.read_json(file_path)
            
            # Check if dataframe is empty
            if df.empty:
                 logger.warning(f"Skipping {file_path.name}: Empty dataframe")
                 continue

            # Assign columns if it's a list of lists (which results in int columns 0-5)
            if len(df.columns) == 6 and list(df.columns) == [0, 1, 2, 3, 4, 5]:
                df.columns = columns
            elif not all(col in df.columns for col in columns):
                # If columns are not correct and it's not the standard list of lists
                logger.warning(f"Skipping {file_path.name}: Unexpected column structure. Columns found: {list(df.columns)}")
                continue

            # Ensure proper types
            df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Determine output path
            if file_path.name.endswith('.json.gz'):
                feather_path = file_path.with_name(file_path.name.replace('.json.gz', '.feather'))
            else:
                feather_path = file_path.with_suffix('.feather')
            
            # Save as feather
            df.to_feather(feather_path)
            
            logger.info(f"Saved to {feather_path.name}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to convert {file_path.name}: {e}")

    logger.info(f"Conversion complete. {success_count}/{len(files)} files converted.")

if __name__ == "__main__":
    convert_freqtrade_data()
