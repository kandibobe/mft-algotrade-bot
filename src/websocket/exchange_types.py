#!/usr/bin/env python3
"""
Exchange types and enums for WebSocket data streaming.
"""

from enum import Enum


class Exchange(Enum):
    """Supported exchanges."""

    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    KRAKEN = "kraken"
