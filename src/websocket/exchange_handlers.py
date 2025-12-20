#!/usr/bin/env python3
"""
Exchange Handlers for WebSocket Data Streaming
==============================================

Abstract base class and concrete implementations for exchange-specific
WebSocket message handling. Uses Strategy pattern to eliminate code duplication.

Author: Stoic Citadel Team
License: MIT
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass

from .data_stream import TickerData, TradeData, Exchange, IWebSocketClient

logger = logging.getLogger(__name__)


class ExchangeHandler(ABC):
    """Abstract base class for exchange-specific WebSocket handlers."""
    
    def __init__(self, exchange: Exchange):
        self.exchange = exchange
    
    @abstractmethod
    async def subscribe(self, websocket: IWebSocketClient, symbols: List[str], channels: List[str]) -> None:
        """
        Subscribe to channels and symbols on the exchange.
        
        Args:
            websocket: WebSocket connection
            symbols: List of trading symbols
            channels: List of channels to subscribe to
        """
        pass
    
    @abstractmethod
    async def handle_message(
        self, 
        data: Dict[str, Any], 
        ticker_handlers: List[Callable[[TickerData], Awaitable[None]]], 
        trade_handlers: List[Callable[[TradeData], Awaitable[None]]]
    ) -> None:
        """
        Parse exchange-specific message and call appropriate handlers.
        
        Args:
            data: Parsed JSON message from WebSocket
            ticker_handlers: List of ticker callback functions
            trade_handlers: List of trade callback functions
        """
        pass
    
    @abstractmethod
    async def subscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """
        Subscribe to a single symbol dynamically.
        
        Args:
            websocket: WebSocket connection
            symbol: Trading symbol to subscribe to
            channels: List of channels to subscribe to
        """
        pass
    
    @abstractmethod
    async def unsubscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """
        Unsubscribe from a single symbol dynamically.
        
        Args:
            websocket: WebSocket connection
            symbol: Trading symbol to unsubscribe from
            channels: List of channels to unsubscribe from
        """
        pass
    
    def get_exchange_name(self) -> str:
        """Get exchange name as string."""
        return self.exchange.value


class BinanceHandler(ExchangeHandler):
    """Handler for Binance WebSocket API."""
    
    def __init__(self):
        super().__init__(Exchange.BINANCE)
    
    async def subscribe(self, websocket: IWebSocketClient, symbols: List[str], channels: List[str]) -> None:
        """Binance-specific subscription."""
        streams = []
        for symbol in symbols:
            symbol_lower = symbol.replace("/", "").lower()
            if "ticker" in channels:
                streams.append(f"{symbol_lower}@ticker")
            if "trade" in channels:
                streams.append(f"{symbol_lower}@trade")
            if "kline" in channels:
                streams.append(f"{symbol_lower}@kline_1m")
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(streams)} Binance streams")
    
    async def handle_message(
        self, 
        data: Dict[str, Any], 
        ticker_handlers: List[Callable[[TickerData], Awaitable[None]]], 
        trade_handlers: List[Callable[[TradeData], Awaitable[None]]]
    ) -> None:
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
            for handler in ticker_handlers:
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
            for handler in trade_handlers:  # type: ignore
                await handler(trade)  # type: ignore
    
    async def subscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """Add symbol to Binance subscription."""
        symbol_lower = symbol.replace("/", "").lower()
        streams = []
        if "ticker" in channels:
            streams.append(f"{symbol_lower}@ticker")
        if "trade" in channels:
            streams.append(f"{symbol_lower}@trade")
        
        msg = {
            "method": "SUBSCRIBE", 
            "params": streams, 
            "id": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(msg))
        logger.info(f"Subscribed to Binance symbol {symbol}")
    
    async def unsubscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """Remove symbol from Binance subscription."""
        symbol_lower = symbol.replace("/", "").lower()
        streams = []
        if "ticker" in channels:
            streams.append(f"{symbol_lower}@ticker")
        if "trade" in channels:
            streams.append(f"{symbol_lower}@trade")
        
        msg = {
            "method": "UNSUBSCRIBE", 
            "params": streams, 
            "id": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(msg))
        logger.info(f"Unsubscribed from Binance symbol {symbol}")


class BybitHandler(ExchangeHandler):
    """Handler for Bybit WebSocket API."""
    
    def __init__(self):
        super().__init__(Exchange.BYBIT)
    
    async def subscribe(self, websocket: IWebSocketClient, symbols: List[str], channels: List[str]) -> None:
        """Bybit-specific subscription."""
        args = []
        for symbol in symbols:
            symbol_upper = symbol.replace("/", "").upper()
            if "ticker" in channels:
                args.append(f"tickers.{symbol_upper}")
            if "trade" in channels:
                args.append(f"publicTrade.{symbol_upper}")
        
        subscribe_msg = {
            "op": "subscribe",
            "args": args
        }
        await websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(args)} Bybit streams")
    
    async def handle_message(
        self, 
        data: Dict[str, Any], 
        ticker_handlers: List[Callable[[TickerData], Awaitable[None]]], 
        trade_handlers: List[Callable[[TradeData], Awaitable[None]]]
    ) -> None:
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
            for handler in ticker_handlers:
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
                for handler in trade_handlers:  # type: ignore
                    await handler(trade)  # type: ignore
    
    async def subscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """Add symbol to Bybit subscription."""
        symbol_upper = symbol.replace("/", "").upper()
        args = []
        if "ticker" in channels:
            args.append(f"tickers.{symbol_upper}")
        if "trade" in channels:
            args.append(f"publicTrade.{symbol_upper}")
        
        msg = {"op": "subscribe", "args": args}
        await websocket.send(json.dumps(msg))
        logger.info(f"Subscribed to Bybit symbol {symbol}")
    
    async def unsubscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """Remove symbol from Bybit subscription."""
        symbol_upper = symbol.replace("/", "").upper()
        args = []
        if "ticker" in channels:
            args.append(f"tickers.{symbol_upper}")
        if "trade" in channels:
            args.append(f"publicTrade.{symbol_upper}")
        
        msg = {"op": "unsubscribe", "args": args}
        await websocket.send(json.dumps(msg))
        logger.info(f"Unsubscribed from Bybit symbol {symbol}")


class OkxHandler(ExchangeHandler):
    """Handler for OKX WebSocket API."""
    
    def __init__(self):
        super().__init__(Exchange.OKX)
    
    async def subscribe(self, websocket: IWebSocketClient, symbols: List[str], channels: List[str]) -> None:
        """OKX-specific subscription."""
        # TODO: Implement OKX subscription logic
        logger.warning("OKX handler not fully implemented")
        pass
    
    async def handle_message(
        self, 
        data: Dict[str, Any], 
        ticker_handlers: List[Callable[[TickerData], Awaitable[None]]], 
        trade_handlers: List[Callable[[TradeData], Awaitable[None]]]
    ) -> None:
        """Handle OKX-specific message format."""
        # TODO: Implement OKX message handling
        logger.warning("OKX handler not fully implemented")
        pass
    
    async def subscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """Add symbol to OKX subscription."""
        # TODO: Implement OKX dynamic subscription
        logger.warning("OKX handler not fully implemented")
        pass
    
    async def unsubscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """Remove symbol from OKX subscription."""
        # TODO: Implement OKX dynamic unsubscription
        logger.warning("OKX handler not fully implemented")
        pass


class KrakenHandler(ExchangeHandler):
    """Handler for Kraken WebSocket API."""
    
    def __init__(self):
        super().__init__(Exchange.KRAKEN)
    
    async def subscribe(self, websocket: IWebSocketClient, symbols: List[str], channels: List[str]) -> None:
        """Kraken-specific subscription."""
        # TODO: Implement Kraken subscription logic
        logger.warning("Kraken handler not fully implemented")
        pass
    
    async def handle_message(
        self, 
        data: Dict[str, Any], 
        ticker_handlers: List[Callable[[TickerData], Awaitable[None]]], 
        trade_handlers: List[Callable[[TradeData], Awaitable[None]]]
    ) -> None:
        """Handle Kraken-specific message format."""
        # TODO: Implement Kraken message handling
        logger.warning("Kraken handler not fully implemented")
        pass
    
    async def subscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """Add symbol to Kraken subscription."""
        # TODO: Implement Kraken dynamic subscription
        logger.warning("Kraken handler not fully implemented")
        pass
    
    async def unsubscribe_symbol(self, websocket: IWebSocketClient, symbol: str, channels: List[str]) -> None:
        """Remove symbol from Kraken subscription."""
        # TODO: Implement Kraken dynamic unsubscription
        logger.warning("Kraken handler not fully implemented")
        pass


# Factory function to create appropriate handler
def create_exchange_handler(exchange: Exchange) -> ExchangeHandler:
    """
    Factory function to create exchange handler based on exchange type.
    
    Args:
        exchange: Exchange enum value
        
    Returns:
        ExchangeHandler instance for the specified exchange
    """
    handlers = {
        Exchange.BINANCE: BinanceHandler,
        Exchange.BYBIT: BybitHandler,
        Exchange.OKX: OkxHandler,
        Exchange.KRAKEN: KrakenHandler,
    }
    
    handler_class = handlers.get(exchange)
    if not handler_class:
        raise ValueError(f"No handler available for exchange: {exchange}")
    
    return handler_class()  # type: ignore
