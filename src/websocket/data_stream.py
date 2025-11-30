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
from typing import Any, Callable, Dict, List, Optional, Set
import websockets
from websockets.exceptions import ConnectionClosed

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
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
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
            "timestamp": self.timestamp
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
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._ws = None
        self._running = False
        self._reconnect_delay = config.reconnect_delay
        
        # Callback handlers
        self._ticker_handlers: List[Callable] = []
        self._trade_handlers: List[Callable] = []
        self._error_handlers: List[Callable] = []
        
        # Message queue for buffering
        self._message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.max_queue_size
        )
        
        # Statistics
        self._stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "reconnects": 0,
            "errors": 0,
            "last_message_time": 0.0,
            "uptime_start": 0.0
        }
        
        # Subscribed symbols tracking
        self._subscribed: Set[str] = set()
    
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
        
        async with websockets.connect(
            url,
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
            close_timeout=10
        ) as ws:
            self._ws = ws
            self._reconnect_delay = self.config.reconnect_delay
            logger.info(f"Connected to {self.config.exchange.value}")
            
            # Subscribe to channels
            await self._subscribe()
            
            # Listen for messages
            await self._listen()
    
    async def _subscribe(self):
        """Subscribe to configured channels and symbols."""
        if self.config.exchange == Exchange.BINANCE:
            await self._subscribe_binance()
        elif self.config.exchange == Exchange.BYBIT:
            await self._subscribe_bybit()
        # Add more exchanges as needed
    
    async def _subscribe_binance(self):
        """Binance-specific subscription."""
        streams = []
        for symbol in self.config.symbols:
            symbol_lower = symbol.replace("/", "").lower()
            if "ticker" in self.config.channels:
                streams.append(f"{symbol_lower}@ticker")
            if "trade" in self.config.channels:
                streams.append(f"{symbol_lower}@trade")
            if "kline" in self.config.channels:
                streams.append(f"{symbol_lower}@kline_1m")
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
        await self._ws.send(json.dumps(subscribe_msg))
        self._subscribed = set(self.config.symbols)
        logger.info(f"Subscribed to {len(streams)} Binance streams")
    
    async def _subscribe_bybit(self):
        """Bybit-specific subscription."""
        args = []
        for symbol in self.config.symbols:
            symbol_upper = symbol.replace("/", "").upper()
            if "ticker" in self.config.channels:
                args.append(f"tickers.{symbol_upper}")
            if "trade" in self.config.channels:
                args.append(f"publicTrade.{symbol_upper}")
        
        subscribe_msg = {
            "op": "subscribe",
            "args": args
        }
        await self._ws.send(json.dumps(subscribe_msg))
        self._subscribed = set(self.config.symbols)
        logger.info(f"Subscribed to {len(args)} Bybit streams")
    
    async def _listen(self):
        """Listen for incoming messages."""
        try:
            async for message in self._ws:
                self._stats["messages_received"] += 1
                self._stats["last_message_time"] = time.time()
                
                try:
                    await self._message_queue.put(message)
                except asyncio.QueueFull:
                    logger.warning("Message queue full, dropping message")
                    
        except ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            raise
    
    async def _handle_reconnect(self):
        """Handle reconnection with exponential backoff."""
        self._stats["reconnects"] += 1
        logger.info(f"Reconnecting in {self._reconnect_delay}s...")
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self.config.max_reconnect_delay
        )
    
    # =========================================================================
    # Message Processing
    # =========================================================================
    
    async def _process_messages(self):
        """Process messages from queue."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
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
            
            if self.config.exchange == Exchange.BINANCE:
                await self._handle_binance_message(data)
            elif self.config.exchange == Exchange.BYBIT:
                await self._handle_bybit_message(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            for handler in self._error_handlers:
                await handler(e)
    
    async def _handle_binance_message(self, data: Dict):
        """Handle Binance-specific message format."""
        event_type = data.get("e")
        
        if event_type == "24hrTicker":
            ticker = TickerData(
                exchange="binance",
                symbol=data["s"],
                bid=float(data["b"]),
                ask=float(data["a"]),
                last=float(data["c"]),
                volume_24h=float(data["v"]),
                change_24h=float(data["P"]),
                timestamp=data["E"] / 1000
            )
            for handler in self._ticker_handlers:
                await handler(ticker)
                
        elif event_type == "trade":
            trade = TradeData(
                exchange="binance",
                symbol=data["s"],
                trade_id=str(data["t"]),
                price=float(data["p"]),
                quantity=float(data["q"]),
                side="buy" if data["m"] else "sell",
                timestamp=data["T"] / 1000
            )
            for handler in self._trade_handlers:
                await handler(trade)
    
    async def _handle_bybit_message(self, data: Dict):
        """Handle Bybit-specific message format."""
        topic = data.get("topic", "")
        
        if "tickers" in topic:
            ticker_data = data.get("data", {})
            ticker = TickerData(
                exchange="bybit",
                symbol=ticker_data.get("symbol", ""),
                bid=float(ticker_data.get("bid1Price", 0)),
                ask=float(ticker_data.get("ask1Price", 0)),
                last=float(ticker_data.get("lastPrice", 0)),
                volume_24h=float(ticker_data.get("volume24h", 0)),
                change_24h=float(ticker_data.get("price24hPcnt", 0)) * 100,
                timestamp=data.get("ts", time.time() * 1000) / 1000
            )
            for handler in self._ticker_handlers:
                await handler(ticker)
                
        elif "publicTrade" in topic:
            for trade_data in data.get("data", []):
                trade = TradeData(
                    exchange="bybit",
                    symbol=trade_data.get("s", ""),
                    trade_id=str(trade_data.get("i", "")),
                    price=float(trade_data.get("p", 0)),
                    quantity=float(trade_data.get("v", 0)),
                    side=trade_data.get("S", "").lower(),
                    timestamp=trade_data.get("T", time.time() * 1000) / 1000
                )
                for handler in self._trade_handlers:
                    await handler(trade)
    
    # =========================================================================
    # Dynamic Subscription Management
    # =========================================================================
    
    async def subscribe_symbol(self, symbol: str):
        """Add symbol to subscription."""
        if symbol in self._subscribed:
            return
        
        if self.config.exchange == Exchange.BINANCE:
            symbol_lower = symbol.replace("/", "").lower()
            streams = [f"{symbol_lower}@ticker", f"{symbol_lower}@trade"]
            msg = {"method": "SUBSCRIBE", "params": streams, "id": int(time.time())}
            await self._ws.send(json.dumps(msg))
        
        self._subscribed.add(symbol)
        logger.info(f"Subscribed to {symbol}")
    
    async def unsubscribe_symbol(self, symbol: str):
        """Remove symbol from subscription."""
        if symbol not in self._subscribed:
            return
        
        if self.config.exchange == Exchange.BINANCE:
            symbol_lower = symbol.replace("/", "").lower()
            streams = [f"{symbol_lower}@ticker", f"{symbol_lower}@trade"]
            msg = {"method": "UNSUBSCRIBE", "params": streams, "id": int(time.time())}
            await self._ws.send(json.dumps(msg))
        
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
            "messages_per_second": (
                self._stats["messages_received"] / max(1, uptime)
            )
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
            "stats": self.get_stats()
        }
