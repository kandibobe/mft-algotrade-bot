#!/usr/bin/env python3
"""
FreqAI Data Purge Script
========================

Cleans up old FreqAI models and training data to prevent disk overflow.
Retains the N most recent models per identifier.

Usage:
    python3 scripts/purge_freqai_data.py [--dry-run]
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
# Ensure we can find paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREQAI_DIR = PROJECT_ROOT / "user_data" / "models"
FREQAI_MODELS_DIR = PROJECT_ROOT / "user_data" / "freqaimodels"

KEEP_LAST_N_MODELS = 2
MAX_AGE_DAYS = 7

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def purge_directory(directory: Path, keep_n: int, dry_run: bool):
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        print(f"Directory not found: {directory}")
        return

    msg = f"Scanning {directory}..."
    logger.info(msg)
    print(msg)
    
    # Group files/folders by identifier if possible, or just look for timestamps
    # FreqAI structure varies. We'll assume subdirectories are model runs or contain model files.
    
    # Strategy: Walk through, find files older than MAX_AGE_DAYS
    deleted_size = 0
    count = 0
    
    # 1. Delete very old files (older than MAX_AGE_DAYS)
    cutoff = datetime.now().timestamp() - (MAX_AGE_DAYS * 86400)
    
    for root, dirs, files in os.walk(directory):
        for f in files:
            fp = Path(root) / f
            try:
                stat = fp.stat()
                if stat.st_mtime < cutoff:
                    size = stat.st_size
                    if not dry_run:
                        fp.unlink()
                        if count % 100 == 0:
                            print(f"Deleted {fp.name}")
                    else:
                        if count % 100 == 0:
                            print(f"Would delete {fp.name}")
                            
                    deleted_size += size
                    count += 1
            except Exception as e:
                logger.error(f"Error checking {fp}: {e}")
                print(f"Error checking {fp}: {e}")

    msg = f"Purged {count} old files (> {MAX_AGE_DAYS} days). Freed {deleted_size / (1024**2):.2f} MB."
    logger.info(msg)
    print(msg)

def main():
    parser = argparse.ArgumentParser(description="Purge old FreqAI data")
    parser.add_argument("--dry-run", action="store_true", help="Simulate deletion")
    args = parser.parse_args()
    
    print("\n--- Starting FreqAI Purge ---")
    logger.info(f"Starting Purge (Dry Run: {args.dry_run})")
    
    # Purge freqaimodels (standard location)
    purge_directory(FREQAI_MODELS_DIR, KEEP_LAST_N_MODELS, args.dry_run)
    
    # Purge user_data/models (if used)
    purge_directory(FREQAI_DIR, KEEP_LAST_N_MODELS, args.dry_run)
    
    print("Purge Complete.\n")
    logger.info("Purge Complete.")

if __name__ == "__main__":
    main()
