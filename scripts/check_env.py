#!/usr/bin/env python3
"""
Environment Validation Script
=============================

Validates that all required environment variables are set and correctly formatted.
This script should be run before starting the main application.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.utils.logger import setup_logger

logger = setup_logger("env_check")

REQUIRED_VARS = [
    "STOIC_MASTER_KEY",  # Required for secret decryption
]

OPTIONAL_VARS = [
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "SENTRY_DSN",
]


def check_env():
    """Check environment variables."""
    # Load .env file
    env_path = Path(".env")
    if not env_path.exists():
        logger.error(f".env file not found at {env_path.absolute()}")
        return False

    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path.absolute()}")

    missing_required = []
    for var in REQUIRED_VARS:
        if not os.getenv(var):
            missing_required.append(var)

    if missing_required:
        logger.error(f"Missing required environment variables: {', '.join(missing_required)}")
        return False

    # Check optional vars
    missing_optional = []
    for var in OPTIONAL_VARS:
        if not os.getenv(var):
            missing_optional.append(var)

    if missing_optional:
        logger.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")

    # Specific Validation Logic
    
    # Check Master Key Length (should be reasonable)
    master_key = os.getenv("STOIC_MASTER_KEY")
    if master_key and len(master_key) < 8:
        logger.warning("STOIC_MASTER_KEY is very short. Ensure it is secure.")

    logger.info("Environment validation passed.")
    return True


if __name__ == "__main__":
    if not check_env():
        sys.exit(1)
    sys.exit(0)
