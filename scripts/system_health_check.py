#!/usr/bin/env python3
"""
System Health Check Script for MFT/FreqAI Bot
=============================================

Performs the following checks:
1. FreqAI Model Freshness: Checks if models are being updated.
2. Disk Space: Monitors available disk space.
3. Exchange Connection: Verifies CCXT connection to the exchange.
4. Memory Usage: Checks RAM usage.

Usage:
    python3 scripts/system_health_check.py
"""

import os
import sys
import time
import shutil
import logging
import psutil
import ccxt
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
# Ensure we can find paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "user_data" / "models"
FREQAI_MODEL_DIR = PROJECT_ROOT / "user_data" / "freqaimodels"
LOG_DIR = PROJECT_ROOT / "user_data" / "logs"

MIN_DISK_SPACE_GB = 2.0
MAX_MODEL_AGE_HOURS = 24  # Warning if model older than 24h
EXCHANGE_ID = "binance"  # Change as needed

# Logging Setup
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), # Ensure output to stdout
        logging.FileHandler(LOG_DIR / "health_check.log")
    ]
)
logger = logging.getLogger(__name__)

def check_disk_space():
    """Check available disk space."""
    try:
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        free_gb = free / (1024**3)
        
        msg = f"Disk Space: {free_gb:.2f} GB free"
        logger.info(msg)
        print(msg)
        
        if free_gb < MIN_DISK_SPACE_GB:
            logger.error(f"CRITICAL: Low Disk Space! Only {free_gb:.2f} GB remaining.")
            print(f"CRITICAL: Low Disk Space! Only {free_gb:.2f} GB remaining.")
            return False
        return True
    except Exception as e:
        logger.error(f"Disk check failed: {e}")
        print(f"Disk check failed: {e}")
        return False

def check_model_freshness():
    """Check if FreqAI models are being updated."""
    try:
        # Check both directories
        dirs_to_check = [FREQAI_MODEL_DIR]
        if MODEL_DIR.exists():
            dirs_to_check.append(MODEL_DIR)
            
        found_models = False
        latest_mtime = 0
        
        for d in dirs_to_check:
            if not d.exists():
                print(f"Directory not found: {d}")
                continue
                
            for p in d.rglob("*"):
                if p.is_file():
                    found_models = True
                    mtime = p.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        
        if not found_models:
            msg = "No FreqAI models found. Is the bot training?"
            logger.warning(msg)
            print(f"WARNING: {msg}")
            # Don't fail the check if just starting, but warn
            return True 
            
        last_update = datetime.fromtimestamp(latest_mtime)
        age = datetime.now() - last_update
        
        msg = f"Latest Model Update: {last_update} ({age.total_seconds()/3600:.1f} hours ago)"
        logger.info(msg)
        print(msg)
        
        if age.total_seconds() > MAX_MODEL_AGE_HOURS * 3600:
            logger.warning(f"FreqAI Models are STALE! Last update {age.total_seconds()/3600:.1f} hours ago.")
            print("WARNING: Models are STALE!")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Model freshness check failed: {e}")
        print(f"Model freshness check failed: {e}")
        return False

def check_exchange_connection():
    """Check connection to exchange via CCXT."""
    try:
        exchange_class = getattr(ccxt, EXCHANGE_ID)
        exchange = exchange_class({'timeout': 10000})
        
        # Fetch ticker (lightweight)
        exchange.fetch_ticker('BTC/USDT')
        msg = f"Exchange Connection ({EXCHANGE_ID}): OK"
        logger.info(msg)
        print(msg)
        return True
    except Exception as e:
        logger.error(f"Exchange Connection FAILED: {e}")
        print(f"Exchange Connection FAILED: {e}")
        return False

def check_memory():
    """Check system memory usage."""
    try:
        mem = psutil.virtual_memory()
        msg = f"Memory Usage: {mem.percent}% (Available: {mem.available / (1024**3):.2f} GB)"
        logger.info(msg)
        print(msg)
        
        if mem.percent > 90:
            logger.warning("High Memory Usage detected!")
            print("WARNING: High Memory Usage!")
            return False
        return True
    except Exception as e:
        logger.error(f"Memory check failed: {e}")
        print(f"Memory check failed: {e}")
        return False

def main():
    print("\n--- Starting System Health Check ---")
    logger.info("--- Starting System Health Check ---")
    
    status = {
        "disk": check_disk_space(),
        "memory": check_memory(),
        "models": check_model_freshness(),
        "exchange": check_exchange_connection()
    }
    
    failed = [k for k, v in status.items() if not v]
    
    if failed:
        logger.error(f"Health Check FAILED for: {', '.join(failed)}")
        print(f"\n❌ Health Check FAILED for: {', '.join(failed)}")
        sys.exit(1)
    else:
        logger.info("System Health: GREEN")
        print("\n✅ System Health: GREEN")
        sys.exit(0)

if __name__ == "__main__":
    main()
