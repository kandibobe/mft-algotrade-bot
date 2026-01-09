#!/usr/bin/env python3
"""
Data types and interfaces for WebSocket data streaming.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IWebSocketClient(Protocol):
    """Interface for WebSocket client dependency."""

    async def connect(
        self,
        uri: str,
        ping_interval: float | None = None,
        ping_timeout: float | None = None,
        close_timeout: float | None = None,
    ):
        """Connect to WebSocket server."""
        ...

    async def send(self, message: str) -> None:
        """Send message to WebSocket."""
        ...

    async def recv(self) -> str:
        """Receive message from WebSocket."""
        ...

    async def close(self) -> None:
        """Close WebSocket connection."""
        ...

    async def __aenter__(self):
        """Async context manager entry."""
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        ...


@dataclass
class TickerData:
    """Normalized ticker data."""

    exchange: str
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    change_24h: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "volume_24h": self.volume_24h,
            "change_24h": self.change_24h,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


@dataclass
class TradeData:
    """Normalized trade data."""

    exchange: str
    symbol: str
    trade_id: str
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "timestamp": self.timestamp,
        }


@dataclass
class OrderbookData:
    """Normalized L2 orderbook data."""

    exchange: str
    symbol: str
    bids: list[list[float]]  # [[price, volume], ...]
    asks: list[list[float]]  # [[price, volume], ...]
    timestamp: float
    imbalance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "bids": self.bids,
            "asks": self.asks,
            "imbalance": self.imbalance,
            "timestamp": self.timestamp,
        }
