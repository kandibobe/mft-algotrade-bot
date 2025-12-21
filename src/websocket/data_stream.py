#!/usr/bin/env python3
"""
WebSocket Data Streaming Service
=================================

Real-time market data streaming via WebSocket connections.
Provides low-latency price updates for multiple symbols simultaneously.

Features:
- Multi-exchange WebSocket connections
- Automatic reconnection with exponential backoff
- Rate limiting and message buffering
- Data normalization across exchanges
- Health monitoring and alerting

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable

import websockets
from websockets.exceptions import ConnectionClosed

from .exchange_handlers import ExchangeHandler, create_exchange_handler

logger = logging.getLogger(__name__)


# =============================================================================
# Dependency Injection Interfaces
# =============================================================================


@runtime_checkable
class IWebSocketClient(Protocol):
    """Interface for WebSocket client dependency."""

    async def connect(
        self,
        uri: str,
        ping_interval: Optional[float] = None,
        ping_timeout: Optional[float] = None,
        close_timeout: Optional[float] = None,
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


logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Supported exchanges."""

    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    KRAKEN = "kraken"


@dataclass
class StreamConfig:
    """Configuration for WebSocket stream."""

    exchange: Exchange
    symbols: List[str]
    channels: List[str] = field(default_factory=lambda: ["ticker", "trade"])
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    max_queue_size: int = 10000

    @property
    def ws_url(self) -> str:
        """Get WebSocket URL for exchange."""
        urls = {
            Exchange.BINANCE: "wss://stream.binance.com:9443/ws",
            Exchange.BYBIT: "wss://stream.bybit.com/v5/public/spot",
            Exchange.OKX: "wss://ws.okx.com:8443/ws/v5/public",
            Exchange.KRAKEN: "wss://ws.kraken.com",
        }
        return urls.get(self.exchange, "")


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

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "timestamp": self.timestamp,
        }


class WebSocketDataStream:
    """
    Real-time market data streaming via WebSocket.

    Usage:
        stream = WebSocketDataStream(config)

        @stream.on_ticker
        async def handle_ticker(ticker: TickerData):
            print(f"{ticker.symbol}: {ticker.last}")

        @stream.on_trade
        async def handle_trade(trade: TradeData):
            print(f"Trade: {trade.price} x {trade.quantity}")

        await stream.start()
    """

    def __init__(self, config: StreamConfig, websocket_client: Optional[IWebSocketClient] = None):
        """
        Initialize WebSocket data stream with dependency injection.

        Args:
            config: Stream configuration
            websocket_client: WebSocket client instance (implements IWebSocketClient)
        """
        self.config = config
        self._ws: Optional[IWebSocketClient] = None
        self._websocket_client = websocket_client
        self._running = False
        self._reconnect_delay = config.reconnect_delay

        # Exchange handler (Strategy pattern)
        self._exchange_handler: ExchangeHandler = create_exchange_handler(config.exchange)

        # Callback handlers
        self._ticker_handlers: List[Callable[[TickerData], Any]] = []
        self._trade_handlers: List[Callable[[TradeData], Any]] = []
        self._error_handlers: List[Callable[[Exception], Any]] = []

        # Message queue for buffering
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)

        # Statistics
        self._stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "reconnects": 0,
            "errors": 0,
            "last_message_time": 0.0,
            "uptime_start": 0.0,
        }

        # Subscribed symbols tracking
        self._subscribed: Set[str] = set()

        # Validate websocket_client implements interface
        if self._websocket_client and not isinstance(self._websocket_client, IWebSocketClient):
            logger.warning("websocket_client does not implement IWebSocketClient interface")

    # =========================================================================
    # Decorator Methods for Event Handlers
    # =========================================================================

    def on_ticker(self, handler: Callable):
        """Decorator to register ticker handler."""
        self._ticker_handlers.append(handler)
        return handler

    def on_trade(self, handler: Callable):
        """Decorator to register trade handler."""
        self._trade_handlers.append(handler)
        return handler

    def on_error(self, handler: Callable):
        """Decorator to register error handler."""
        self._error_handlers.append(handler)
        return handler

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def start(self):
        """Start the WebSocket stream."""
        self._running = True
        self._stats["uptime_start"] = time.time()

        # Start message processor
        asyncio.create_task(self._process_messages())

        # Connect and maintain connection
        while self._running:
            try:
                await self._connect()
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self._stats["errors"] += 1
                await self._handle_reconnect()

    async def stop(self):
        """Stop the WebSocket stream."""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("WebSocket stream stopped")

    async def _connect(self):
        """Establish WebSocket connection."""
        url = self.config.ws_url
        logger.info(f"Connecting to {url}")

        # Use provided websocket client or create default
        if self._websocket_client:
            # If client implements connect method, use it
            if hasattr(self._websocket_client, "connect"):
                self._ws = await self._websocket_client.connect(
                    uri=url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                    close_timeout=10,
                )
            else:
                # Assume client is already a connected WebSocket
                self._ws = self._websocket_client
            logger.info(f"Using provided WebSocket client for {self.config.exchange.value}")
        else:
            # Create default websockets client
            self._ws = await websockets.connect(
                url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=10,
            )
            logger.info(f"Created default WebSocket connection to {self.config.exchange.value}")

        self._reconnect_delay = self.config.reconnect_delay
        logger.info(f"Connected to {self.config.exchange.value}")

        # Subscribe to channels
        await self._subscribe()

        # Listen for messages
        await self._listen()

    async def _subscribe(self):
        """Subscribe to configured channels and symbols using exchange handler."""
        await self._exchange_handler.subscribe(self._ws, self.config.symbols, self.config.channels)
        self._subscribed = set(self.config.symbols)

    async def _listen(self):
        """Listen for incoming messages."""
        try:
            # Use async iteration if available, otherwise use recv()
            if hasattr(self._ws, "__aiter__"):
                async for message in self._ws:
                    await self._handle_incoming_message(message)
            else:
                while self._running:
                    try:
                        message = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
                        await self._handle_incoming_message(message)
                    except asyncio.TimeoutError:
                        continue

        except ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            raise
        except Exception as e:
            logger.error(f"Listen error: {e}")
            raise

    async def _handle_reconnect(self):
        """Handle reconnection with exponential backoff."""
        self._stats["reconnects"] += 1
        logger.info(f"Reconnecting in {self._reconnect_delay}s...")
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self.config.max_reconnect_delay)

    # =========================================================================
    # Message Processing
    # =========================================================================

    async def _handle_incoming_message(self, message: str):
        """Handle incoming WebSocket message."""
        self._stats["messages_received"] += 1
        self._stats["last_message_time"] = time.time()

        try:
            await self._message_queue.put(message)
        except asyncio.QueueFull:
            logger.warning("Message queue full, dropping message")

    async def _process_messages(self):
        """Process messages from queue."""
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._handle_message(message)
                self._stats["messages_processed"] += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")

    async def _handle_message(self, raw_message: str):
        """Parse and route message to handlers using exchange handler."""
        try:
            data = json.loads(raw_message)
            await self._exchange_handler.handle_message(
                data, self._ticker_handlers, self._trade_handlers
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            for handler in self._error_handlers:
                await handler(e)

    # =========================================================================
    # Dynamic Subscription Management
    # =========================================================================

    async def subscribe_symbol(self, symbol: str):
        """Add symbol to subscription using exchange handler."""
        if symbol in self._subscribed:
            return

        if self._ws is None:
            logger.error("Cannot subscribe symbol: WebSocket not connected")
            return

        await self._exchange_handler.subscribe_symbol(self._ws, symbol, self.config.channels)
        self._subscribed.add(symbol)
        logger.info(f"Subscribed to {symbol}")

    async def unsubscribe_symbol(self, symbol: str):
        """Remove symbol from subscription using exchange handler."""
        if symbol not in self._subscribed:
            return

        if self._ws is None:
            logger.error("Cannot unsubscribe symbol: WebSocket not connected")
            return

        await self._exchange_handler.unsubscribe_symbol(self._ws, symbol, self.config.channels)
        self._subscribed.discard(symbol)
        logger.info(f"Unsubscribed from {symbol}")

    # =========================================================================
    # Health & Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        uptime = time.time() - self._stats["uptime_start"] if self._stats["uptime_start"] else 0
        return {
            **self._stats,
            "uptime_seconds": uptime,
            "connected": self._ws is not None and self._running,
            "subscribed_symbols": list(self._subscribed),
            "queue_size": self._message_queue.qsize(),
            "messages_per_second": (self._stats["messages_received"] / max(1, uptime)),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        last_msg_age = time.time() - self._stats["last_message_time"]

        return {
            "service": "websocket_stream",
            "status": "healthy" if last_msg_age < 30 else "degraded",
            "exchange": self.config.exchange.value,
            "connected": self._ws is not None,
            "last_message_age_seconds": last_msg_age,
            "stats": self.get_stats(),
        }
