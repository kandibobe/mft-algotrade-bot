"""
System Watchdog for Stoic Citadel
=================================

Monitors critical system components and health checks.
Restarts the service if a deadlock or crash is detected.
"""

import logging
import os
import sys
import time
import requests
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("user_data/logs/watchdog.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Watchdog")

# Configuration
API_URL = "http://localhost:8080/api/v1/ping"
CHECK_INTERVAL = 60  # Check every 60 seconds
MAX_RETRIES = 3      # Retry 3 times before restarting
DOCKER_CONTAINER_NAME = "stoic_citadel_freqtrade"

def check_api_health():
    """Ping the Freqtrade API."""
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            return True
        logger.warning(f"API Check Failed: Status Code {response.status_code}")
        return False
    except Exception as e:
        logger.warning(f"API Check Failed: {e}")
        return False

def check_log_freshness(log_file="user_data/logs/freqtrade.log", max_age_seconds=300):
    """Check if the main log file has been updated recently."""
    if not os.path.exists(log_file):
        logger.warning(f"Log file not found: {log_file}")
        return False
    
    try:
        last_modified = os.path.getmtime(log_file)
        age = time.time() - last_modified
        if age > max_age_seconds:
            logger.warning(f"Log Stale: Last update was {age:.0f}s ago (Limit: {max_age_seconds}s)")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking log freshness: {e}")
        return True # Assume healthy if we can't read log to prevent loop

def restart_service():
    """Restart the Docker container."""
    logger.critical("🚨 RESTARTING SERVICE DUE TO HEALTH CHECK FAILURE 🚨")
    try:
        # Use docker compose if possible, or docker restart
        cmd = "docker-compose restart freqtrade"
        subprocess.run(cmd, shell=True, check=True)
        logger.info("Service restart command issued successfully.")
        
        # Optional: Send Telegram Alert (using curl/requests if token is available)
        # We rely on the bot coming back online to report "I'm alive"
    except subprocess.CalledProcessError as e:
        logger.critical(f"Failed to restart service: {e}")

def main():
    logger.info("🛡️ Watchdog started guarding Stoic Citadel")
    failures = 0
    
    while True:
        try:
            api_ok = check_api_health()
            logs_ok = check_log_freshness()
            
            if api_ok and logs_ok:
                failures = 0 # Reset counter
            else:
                failures += 1
                logger.warning(f"Health Check Failed ({failures}/{MAX_RETRIES}). API: {api_ok}, Logs: {logs_ok}")
            
            if failures >= MAX_RETRIES:
                restart_service()
                failures = 0 # Reset after restart attempt
                time.sleep(120) # Give it time to boot
                
        except KeyboardInterrupt:
            logger.info("Watchdog stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in watchdog loop: {e}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()