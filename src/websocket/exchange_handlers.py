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
from collections.abc import Awaitable, Callable
from typing import Any

from .data_types import IWebSocketClient, OrderbookData, TickerData, TradeData
from .exchange_types import Exchange

logger = logging.getLogger(__name__)


class ExchangeHandler(ABC):
    """Abstract base class for exchange-specific WebSocket handlers."""

    def __init__(self, exchange: Exchange):
        self.exchange = exchange

    @abstractmethod
    async def subscribe(
        self, websocket: IWebSocketClient, symbols: list[str], channels: list[str]
    ) -> None:
        """
        Subscribe to channels and symbols on the exchange.
        """
        pass

    @abstractmethod
    async def handle_message(
        self,
        data: dict[str, Any],
        ticker_handlers: list[Callable[[TickerData], Awaitable[None]]],
        trade_handlers: list[Callable[[TradeData], Awaitable[None]]],
        orderbook_handlers: list[Callable[[OrderbookData], Awaitable[None]]],
    ) -> None:
        """
        Parse exchange-specific message and call appropriate handlers.
        """
        pass

    @abstractmethod
    async def subscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
        """
        Subscribe to a single symbol dynamically.
        """
        pass

    @abstractmethod
    async def unsubscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
        """
        Unsubscribe from a single symbol dynamically.
        """
        pass

    def get_exchange_name(self) -> str:
        """Get exchange name as string."""
        return self.exchange.value


class BinanceHandler(ExchangeHandler):
    """Handler for Binance WebSocket API."""

    def __init__(self):
        super().__init__(Exchange.BINANCE)

    async def subscribe(
        self, websocket: IWebSocketClient, symbols: list[str], channels: list[str]
    ) -> None:
        """Binance-specific subscription."""
        streams = []
        for symbol in symbols:
            symbol_lower = symbol.replace("/", "").lower()
            if "ticker" in channels:
                streams.append(f"{symbol_lower}@ticker")
            if "trade" in channels:
                streams.append(f"{symbol_lower}@trade")
            if "orderbook" in channels:
                streams.append(f"{symbol_lower}@depth5@100ms") # Top 5 levels, 100ms update
            if "kline" in channels:
                streams.append(f"{symbol_lower}@kline_1m")

        subscribe_msg = {"method": "SUBSCRIBE", "params": streams, "id": int(time.time() * 1000)}
        await websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(streams)} Binance streams")

    async def handle_message(
        self,
        data: dict[str, Any],
        ticker_handlers: list[Callable[[TickerData], Awaitable[None]]],
        trade_handlers: list[Callable[[TradeData], Awaitable[None]]],
        orderbook_handlers: list[Callable[[OrderbookData], Awaitable[None]]],
    ) -> None:
        """Handle Binance-specific message format."""
        # 1. Handle Depth/Orderbook (partial book depth)
        # Note: top-N depth doesn't have 'e' field, but has 'lastUpdateId'
        if "bids" in data and "asks" in data and "lastUpdateId" in data:
            # For Binance @depthN, symbol is NOT in the payload usually.
            symbol = data.get("s", "UNKNOWN/USDT")

            bids = [[float(b[0]), float(b[1])] for b in data["bids"]]
            asks = [[float(a[0]), float(a[1])] for a in data["asks"]]

            bid_vol = sum(b[1] for b in bids)
            ask_vol = sum(a[1] for a in asks)
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

            ob = OrderbookData(
                exchange="binance",
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=time.time(),
                imbalance=imbalance
            )
            for handler in orderbook_handlers:
                await handler(ob)
            return

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
                timestamp=data["E"] / 1000,
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
                timestamp=data["T"] / 1000,
            )
            for handler in trade_handlers:
                await handler(trade)

    async def subscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
        """Add symbol to Binance subscription."""
        symbol_lower = symbol.replace("/", "").lower()
        streams = []
        if "ticker" in channels:
            streams.append(f"{symbol_lower}@ticker")
        if "trade" in channels:
            streams.append(f"{symbol_lower}@trade")
        if "orderbook" in channels:
            streams.append(f"{symbol_lower}@depth5@100ms")

        msg = {"method": "SUBSCRIBE", "params": streams, "id": int(time.time() * 1000)}
        await websocket.send(json.dumps(msg))
        logger.info(f"Subscribed to Binance symbol {symbol}")

    async def unsubscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
        """Remove symbol from Binance subscription."""
        symbol_lower = symbol.replace("/", "").lower()
        streams = []
        if "ticker" in channels:
            streams.append(f"{symbol_lower}@ticker")
        if "trade" in channels:
            streams.append(f"{symbol_lower}@trade")

        msg = {"method": "UNSUBSCRIBE", "params": streams, "id": int(time.time() * 1000)}
        await websocket.send(json.dumps(msg))
        logger.info(f"Unsubscribed from Binance symbol {symbol}")


class BybitHandler(ExchangeHandler):
    """Handler for Bybit WebSocket API."""

    def __init__(self):
        super().__init__(Exchange.BYBIT)

    async def subscribe(
        self, websocket: IWebSocketClient, symbols: list[str], channels: list[str]
    ) -> None:
        """Bybit-specific subscription."""
        args = []
        for symbol in symbols:
            symbol_upper = symbol.replace("/", "").upper()
            if "ticker" in channels:
                args.append(f"tickers.{symbol_upper}")
            if "trade" in channels:
                args.append(f"publicTrade.{symbol_upper}")

        subscribe_msg = {"op": "subscribe", "args": args}
        await websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(args)} Bybit streams")

    async def handle_message(
        self,
        data: dict[str, Any],
        ticker_handlers: list[Callable[[TickerData], Awaitable[None]]],
        trade_handlers: list[Callable[[TradeData], Awaitable[None]]],
        orderbook_handlers: list[Callable[[OrderbookData], Awaitable[None]]],
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
                timestamp=data.get("ts", time.time() * 1000) / 1000,
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
                    timestamp=trade_data.get("T", time.time() * 1000) / 1000,
                )
                for handler in trade_handlers:
                    await handler(trade)

    async def subscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
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

    async def unsubscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
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

    async def subscribe(
        self, websocket: IWebSocketClient, symbols: list[str], channels: list[str]
    ) -> None:
        """OKX-specific subscription."""
        args = []
        for symbol in symbols:
            symbol_upper = symbol.replace("/", "-").upper()
            if "ticker" in channels:
                args.append({"channel": "tickers", "instId": symbol_upper})
            if "trade" in channels:
                args.append({"channel": "trades", "instId": symbol_upper})

        subscribe_msg = {"op": "subscribe", "args": args}
        await websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(args)} OKX streams")

    async def handle_message(
        self,
        data: dict[str, Any],
        ticker_handlers: list[Callable[[TickerData], Awaitable[None]]],
        trade_handlers: list[Callable[[TradeData], Awaitable[None]]],
        orderbook_handlers: list[Callable[[OrderbookData], Awaitable[None]]],
    ) -> None:
        """Handle OKX-specific message format."""
        event = data.get("event")
        if event == "subscribe" or event == "error":
            logger.debug(f"OKX subscription event: {data}")
            return

        arg = data.get("arg", {})
        channel = arg.get("channel", "")
        data_list = data.get("data", [])

        if not data_list:
            return

        if channel == "tickers":
            for ticker_data in data_list:
                ticker = TickerData(
                    exchange="okx",
                    symbol=ticker_data.get("instId", "").replace("-", "/"),
                    bid=float(ticker_data.get("bidPx", 0)),
                    ask=float(ticker_data.get("askPx", 0)),
                    last=float(ticker_data.get("last", 0)),
                    volume_24h=float(ticker_data.get("vol24h", 0)),
                    change_24h=float(ticker_data.get("lastPx", 0))
                    / float(ticker_data.get("open24h", 1))
                    - 1
                    if ticker_data.get("open24h")
                    else 0,
                    timestamp=int(ticker_data.get("ts", time.time() * 1000)) / 1000,
                )
                for handler in ticker_handlers:
                    await handler(ticker)

        elif channel == "trades":
            for trade_data in data_list:
                trade = TradeData(
                    exchange="okx",
                    symbol=arg.get("instId", "").replace("-", "/"),
                    trade_id=str(trade_data.get("tradeId", "")),
                    price=float(trade_data.get("px", 0)),
                    quantity=float(trade_data.get("sz", 0)),
                    side="buy" if trade_data.get("side") == "buy" else "sell",
                    timestamp=int(trade_data.get("ts", time.time() * 1000)) / 1000,
                )
                for handler in trade_handlers:
                    await handler(trade)

    async def subscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
        """Add symbol to OKX subscription."""
        symbol_upper = symbol.replace("/", "-").upper()
        args = []
        if "ticker" in channels:
            args.append({"channel": "tickers", "instId": symbol_upper})
        if "trade" in channels:
            args.append({"channel": "trades", "instId": symbol_upper})

        msg = {"op": "subscribe", "args": args}
        await websocket.send(json.dumps(msg))
        logger.info(f"Subscribed to OKX symbol {symbol}")

    async def unsubscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
        """Remove symbol from OKX subscription."""
        symbol_upper = symbol.replace("/", "-").upper()
        args = []
        if "ticker" in channels:
            args.append({"channel": "tickers", "instId": symbol_upper})
        if "trade" in channels:
            args.append({"channel": "trades", "instId": symbol_upper})

        msg = {"op": "unsubscribe", "args": args}
        await websocket.send(json.dumps(msg))
        logger.info(f"Unsubscribed from OKX symbol {symbol}")


class KrakenHandler(ExchangeHandler):
    """Handler for Kraken WebSocket API."""

    def __init__(self):
        super().__init__(Exchange.KRAKEN)

    async def subscribe(
        self, websocket: IWebSocketClient, symbols: list[str], channels: list[str]
    ) -> None:
        """Kraken-specific subscription."""
        subscribe_msg = {
            "event": "subscribe",
            "pair": [self._format_symbol(symbol) for symbol in symbols],
            "subscription": {},
        }

        if "ticker" in channels:
            subscribe_msg["subscription"]["ticker"] = {}
        if "trade" in channels:
            subscribe_msg["subscription"]["trade"] = {}

        await websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to {len(symbols)} Kraken symbols")

    async def handle_message(
        self,
        data: dict[str, Any],
        ticker_handlers: list[Callable[[TickerData], Awaitable[None]]],
        trade_handlers: list[Callable[[TradeData], Awaitable[None]]],
        orderbook_handlers: list[Callable[[OrderbookData], Awaitable[None]]],
    ) -> None:
        """Handle Kraken-specific message format."""
        event = data.get("event")
        if event == "subscriptionStatus" or event == "heartbeat":
            logger.debug(f"Kraken event: {event}")
            return

        # Check if it's ticker data
        if isinstance(data, list) and len(data) >= 4 and isinstance(data[1], dict) and "a" in data[1]:
            ticker_data = data[1]
            pair_name = data[3]
            symbol = self._parse_symbol(pair_name)

            ticker = TickerData(
                exchange="kraken",
                symbol=symbol,
                bid=float(ticker_data.get("b", [0])[0]),
                ask=float(ticker_data.get("a", [0])[0]),
                last=float(ticker_data.get("c", [0])[0]),
                volume_24h=float(ticker_data.get("v", [0, 0])[1]),
                change_24h=float(ticker_data.get("p", [0, 0])[1]),
                timestamp=time.time(),
            )
            for handler in ticker_handlers:
                await handler(ticker)

    async def subscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
        """Add symbol to Kraken subscription."""
        subscribe_msg = {
            "event": "subscribe",
            "pair": [self._format_symbol(symbol)],
            "subscription": {},
        }

        if "ticker" in channels:
            subscribe_msg["subscription"]["ticker"] = {}
        if "trade" in channels:
            subscribe_msg["subscription"]["trade"] = {}

        await websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to Kraken symbol {symbol}")

    async def unsubscribe_symbol(
        self, websocket: IWebSocketClient, symbol: str, channels: list[str]
    ) -> None:
        """Remove symbol from Kraken subscription."""
        unsubscribe_msg = {
            "event": "unsubscribe",
            "pair": [self._format_symbol(symbol)],
            "subscription": {},
        }

        if "ticker" in channels:
            unsubscribe_msg["subscription"]["ticker"] = {}
        if "trade" in channels:
            unsubscribe_msg["subscription"]["trade"] = {}

        await websocket.send(json.dumps(unsubscribe_msg))
        logger.info(f"Unsubscribed from Kraken symbol {symbol}")

    def _format_symbol(self, symbol: str) -> str:
        formatted = symbol.replace("/", "")
        if formatted.startswith("BTC"):
            formatted = "XBT" + formatted[3:]
        return formatted

    def _parse_symbol(self, kraken_symbol: str) -> str:
        if kraken_symbol.startswith("XBT"):
            kraken_symbol = "BTC" + kraken_symbol[3:]
        if len(kraken_symbol) >= 6:
            base = kraken_symbol[:3]
            quote = kraken_symbol[3:]
            return f"{base}/{quote}"
        return kraken_symbol


def create_exchange_handler(exchange: Exchange) -> ExchangeHandler:
    """Factory function to create exchange handler."""
    handlers = {
        Exchange.BINANCE: BinanceHandler,
        Exchange.BYBIT: BybitHandler,
        Exchange.OKX: OkxHandler,
        Exchange.KRAKEN: KrakenHandler,
    }

    handler_class = handlers.get(exchange)
    if not handler_class:
        raise ValueError(f"No handler available for exchange: {exchange}")

    return handler_class()
    handler_class = handlers.get(exchange)
