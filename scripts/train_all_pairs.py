#!/usr/bin/env python3
"""
Train All Pairs Script
======================

Reads pair whitelist from config and trains ML models for ALL pairs.
Auto-downloads missing data.

Usage:
    python scripts/train_all_pairs.py --config user_data/config/config.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing tools
from scripts.train_models import MLTrainingPipeline
from scripts.download_data import download_data

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_whitelist(config_path: str) -> list:
    """Load pair whitelist from config file."""
    path = Path(config_path)
    if not path.exists():
        # Fallback to root config.json if user_data path fails
        root_path = Path("config.json")
        if root_path.exists():
            print(f"⚠️  Config not found at {config_path}, using {root_path}")
            path = root_path
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    
    pairs = config.get('exchange', {}).get('pair_whitelist', [])
    if not pairs:
        raise ValueError("No pairs found in config (exchange.pair_whitelist)")
    
    return pairs

def check_data_exists(pair: str, timeframe: str, data_dir: Path) -> bool:
    """Check if data exists for pair."""
    pair_filename = pair.replace('/', '_')
    feather_path = data_dir / f"{pair_filename}-{timeframe}.feather"
    json_path = data_dir / f"{pair_filename}-{timeframe}.json"
    
    return feather_path.exists() or json_path.exists()

def main():
    parser = argparse.ArgumentParser(description="Train ML models for all whitelisted pairs")
    parser.add_argument('--config', type=str, default='user_data/config/config.json', help='Path to config file')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe to train on')
    parser.add_argument('--days', type=int, default=30, help='Days of data to download if missing')
    parser.add_argument('--force-download', action='store_true', help='Force download even if data exists')
    parser.add_argument('--quick', action='store_true', help='Quick training mode')
    
    args = parser.parse_args()
    setup_logging()
    
    print("\n" + "="*70)
    print("🚀 TRAIN ALL PAIRS")
    print("="*70)
    
    # 1. Load Whitelist
    try:
        pairs = load_whitelist(args.config)
        print(f"📋 Loaded {len(pairs)} pairs from config:")
        print(f"   {', '.join(pairs)}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
        
    # 2. Check Data & Download
    data_dir = Path("user_data/data/binance")
    pairs_to_download = []
    
    for pair in pairs:
        exists = check_data_exists(pair, args.timeframe, data_dir)
        if not exists or args.force_download:
            pairs_to_download.append(pair)
            status = "Missing" if not exists else "Forced"
            print(f"   ⚠️  {pair}: Data {status} -> Queued for download")
        else:
            print(f"   ✅ {pair}: Data found")
            
    if pairs_to_download:
        print(f"\n📥 Downloading data for {len(pairs_to_download)} pairs...")
        success = download_data(
            pairs=pairs_to_download,
            timeframe=args.timeframe,
            days=args.days,
            use_docker=True # Defaulting to True as per safe default, user can change logic if needed
        )
        if not success:
            print("❌ Data download failed. Trying local freqtrade...")
            success = download_data(
                pairs=pairs_to_download,
                timeframe=args.timeframe,
                days=args.days,
                use_docker=False
            )
            
            if not success:
                print("❌ Failed to download data. Exiting.")
                sys.exit(1)
    
    # 3. Run Training
    print(f"\n🤖 Starting training for {len(pairs)} pairs...")
    
    pipeline = MLTrainingPipeline(
        data_dir=str(data_dir),
        models_dir="user_data/models",
        quick_mode=args.quick
    )
    
    pipeline.run(
        pairs=pairs,
        timeframe=args.timeframe,
        optimize=False # Default to no optimization for batch run to save time
    )

if __name__ == "__main__":
    main()
