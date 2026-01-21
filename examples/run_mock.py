#!/usr/bin/env python3
"""
MOCK ENTRY POINT / PLACEHOLDER
==============================

⚠️ WARNING: This is NOT the production entry point! ⚠️

The actual trading logic is executed by Freqtrade.
Entry Point: `freqtrade trade --config config.json --strategy StoicEnsembleStrategyV4`

This file (`src/main.py`) serves as a placeholder for a future standalone execution engine
that bypasses Freqtrade, but it is currently non-functional/mock code.

DO NOT RUN THIS FILE FOR LIVE TRADING.
"""

import sys
import time

from src.utils.logger import log, setup_structured_logging


def main():
    """
    Mock entry point.
    """
    setup_structured_logging(level="INFO", json_output=True, enable_console=True, enable_file=False)

    log.warning("deprecated_entry_point_warning")
    print("\n" + "=" * 60)
    print("🛑 THIS IS A MOCK ENTRY POINT")
    print("=" * 60)
    print("You are trying to run `src/main.py`.")
    print("This file contains mock logic and is NOT connected to the exchange.")
    print("\nTo run the bot, use Freqtrade:")
    print("  freqtrade trade --config config.json --strategy StoicEnsembleStrategyV4")
    print("\nTo train models:")
    print("  python scripts/train_meta_model.py")
    print("=" * 60 + "\n")

    # Simulate startup for testing purposes only
    time.sleep(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
