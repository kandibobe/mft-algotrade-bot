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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from .data_types import IWebSocketClient, OrderbookData, TickerData, TradeData
from .exchange_handlers import ExchangeHandler, create_exchange_handler
from .exchange_types import Exchange

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for WebSocket stream."""

    exchange: Exchange
    symbols: list[str]
    channels: list[str] = field(default_factory=lambda: ["ticker", "trade"])
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


class WebSocketDataStream:
    """
    Real-time market data streaming via WebSocket.
    """

    def __init__(self, config: StreamConfig, websocket_client: IWebSocketClient | None = None):
        """
        Initialize WebSocket data stream.
        """
        self.config = config
        self._ws: IWebSocketClient | None = None
        self._websocket_client = websocket_client
        self._running = False
        self._reconnect_delay = config.reconnect_delay

        # Exchange handler
        self._exchange_handler: ExchangeHandler = create_exchange_handler(config.exchange)

        # Callback handlers
        self._ticker_handlers: list[Callable[[TickerData], Any]] = []
        self._trade_handlers: list[Callable[[TradeData], Any]] = []
        self._orderbook_handlers: list[Callable[[OrderbookData], Any]] = []
        self._error_handlers: list[Callable[[Exception], Any]] = []

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
        self._subscribed: set[str] = set()

        # Background tasks
        self._background_tasks = set()

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

    def on_orderbook(self, handler: Callable):
        """Decorator to register orderbook handler."""
        self._orderbook_handlers.append(handler)
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
        task = asyncio.create_task(self._process_messages())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        # Connect and maintain connection
        while self._running:
            try:
                await self._connect()
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self._stats["errors"] += 1
                await self._handle_reconnect()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False

    async def stop(self):
        """Stop the WebSocket stream and clean up resources."""
        self._running = False

        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._ws = None

        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
                self._message_queue.task_done()
            except asyncio.QueueEmpty:
                break

        self._ticker_handlers.clear()
        self._trade_handlers.clear()
        self._orderbook_handlers.clear()
        self._error_handlers.clear()

        logger.info("WebSocket stream stopped and cleaned up")

    async def _connect(self):
        """Establish WebSocket connection."""
        url = self.config.ws_url
        logger.info(f"Connecting to {url}")

        if self._websocket_client:
            if hasattr(self._websocket_client, "connect"):
                self._ws = await self._websocket_client.connect(
                    uri=url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                    close_timeout=10,
                )
            else:
                self._ws = self._websocket_client
        else:
            self._ws = await websockets.connect(
                url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=10,
            )

        self._reconnect_delay = self.config.reconnect_delay
        await self._subscribe()
        await self._listen()

    async def _subscribe(self):
        """Subscribe to configured channels and symbols."""
        await self._exchange_handler.subscribe(self._ws, self.config.symbols, self.config.channels)
        self._subscribed = set(self.config.symbols)

    async def _listen(self):
        """Listen for incoming messages."""
        try:
            # Check if we should use watchdog based on last_message_time
            # For some markets/timeframes, 5s might be too short during low liquidity
            from src.config.unified_config import load_config

            u_cfg = load_config()
            watchdog_timeout = u_cfg.system.ws_watchdog_timeout

            if hasattr(self._ws, "__aiter__"):
                while self._running:
                    try:
                        # Add watchdog for each message
                        message = await asyncio.wait_for(self._ws.recv(), timeout=watchdog_timeout)
                        await self._handle_incoming_message(message)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Websocket Watchdog: No data for {watchdog_timeout} seconds. Reconnecting..."
                        )
                        break  # Exit _listen to trigger reconnect in start() loop
            else:
                while self._running:
                    try:
                        message = await asyncio.wait_for(self._ws.recv(), timeout=watchdog_timeout)
                        await self._handle_incoming_message(message)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Websocket Watchdog: No data for {watchdog_timeout} seconds. Reconnecting..."
                        )
                        break

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
        """Parse and route message to handlers."""
        try:
            data = json.loads(raw_message)
            await self._exchange_handler.handle_message(
                data, self._ticker_handlers, self._trade_handlers, self._orderbook_handlers
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
        if symbol in self._subscribed:
            return
        if self._ws is None:
            return
        await self._exchange_handler.subscribe_symbol(self._ws, symbol, self.config.channels)
        self._subscribed.add(symbol)

    async def unsubscribe_symbol(self, symbol: str):
        if symbol not in self._subscribed:
            return
        if self._ws is None:
            return
        await self._exchange_handler.unsubscribe_symbol(self._ws, symbol, self.config.channels)
        self._subscribed.discard(symbol)

    # =========================================================================
    # Health & Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        uptime = time.time() - self._stats["uptime_start"] if self._stats["uptime_start"] else 0
        return {
            **self._stats,
            "uptime_seconds": uptime,
            "connected": self._ws is not None and self._running,
            "subscribed_symbols": list(self._subscribed),
            "queue_size": self._message_queue.qsize(),
            "messages_per_second": (self._stats["messages_received"] / max(1, uptime)),
        }

    async def health_check(self) -> dict[str, Any]:
        last_msg_age = time.time() - self._stats["last_message_time"]
        return {
            "service": "websocket_stream",
            "status": "healthy" if last_msg_age < 30 else "degraded",
            "exchange": self.config.exchange.value,
            "connected": self._ws is not None,
            "last_message_age_seconds": last_msg_age,
            "stats": self.get_stats(),
        }
