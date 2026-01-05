#!/usr/bin/env python3
"""
Data Aggregator for Multiple Streams
=====================================

Aggregates data from multiple WebSocket streams into unified view.

Features:
- Multi-exchange aggregation
- Best bid/ask calculation
- Volume-weighted average price (VWAP)
- Spread monitoring
- Arbitrage detection

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .data_stream import Exchange, StreamConfig, TickerData, TradeData, WebSocketDataStream

logger = logging.getLogger(__name__)


@dataclass
class AggregatedTicker:
    """Aggregated ticker from multiple exchanges."""

    symbol: str
    best_bid: float
    best_bid_exchange: str
    best_ask: float
    best_ask_exchange: str
    spread: float
    spread_pct: float
    exchanges: dict[str, TickerData]
    vwap: float
    total_volume_24h: float
    timestamp: float
    is_reliable: bool = True
    reliability_reason: str | None = None

    @property
    def arbitrage_opportunity(self) -> bool:
        """Check if arbitrage is possible (best bid > best ask on different exchanges)."""
        return self.best_bid > self.best_ask and self.best_bid_exchange != self.best_ask_exchange

    @property
    def arbitrage_profit_pct(self) -> float:
        """Calculate potential arbitrage profit percentage."""
        if not self.arbitrage_opportunity:
            return 0.0
        return (self.best_bid - self.best_ask) / self.best_ask * 100


@dataclass
class TradeVolume:
    """Accumulated trade volume statistics."""

    symbol: str
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    total_value: float = 0.0
    vwap: float = 0.0
    window_start: float = field(default_factory=time.time)

    @property
    def net_volume(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def buy_ratio(self) -> float:
        total = self.buy_volume + self.sell_volume
        return self.buy_volume / total if total > 0 else 0.5


class DataAggregator:
    """
    Aggregates data from multiple exchange streams.

    Usage:
        aggregator = DataAggregator()

        # Add exchanges
        aggregator.add_exchange(Exchange.BINANCE, ["BTC/USDT", "ETH/USDT"])
        aggregator.add_exchange(Exchange.BYBIT, ["BTC/USDT", "ETH/USDT"])

        # Register callbacks
        @aggregator.on_aggregated_ticker
        async def handle_ticker(ticker: AggregatedTicker):
            if ticker.arbitrage_opportunity:
                logger.info(f"Arbitrage! {ticker.arbitrage_profit_pct:.2f}% profit")

        # Start
        await aggregator.start()
    """

    def __init__(self, aggregation_interval: float = 0.1):
        self.aggregation_interval = aggregation_interval
        self._streams: dict[Exchange, WebSocketDataStream] = {}
        self._running = False

        # Data storage
        self._tickers: dict[str, dict[str, TickerData]] = defaultdict(
            dict
        )  # symbol -> exchange -> ticker
        self._trade_volumes: dict[str, TradeVolume] = {}  # symbol -> volume
        self._recent_trades: dict[str, list[TradeData]] = defaultdict(list)  # symbol -> trades

        # Callbacks
        self._aggregated_ticker_handlers: list[Callable] = []
        self._arbitrage_handlers: list[Callable] = []
        self._volume_alert_handlers: list[Callable] = []

        # Configuration
        self._volume_window_seconds = 60  # 1 minute window
        self._max_recent_trades = 1000

    # =========================================================================
    # Configuration
    # =========================================================================

    def add_exchange(
        self, exchange: Exchange, symbols: list[str], channels: list[str] | None = None
    ):
        """Add exchange stream to aggregator."""
        config = StreamConfig(
            exchange=exchange, symbols=symbols, channels=channels or ["ticker", "trade"]
        )
        stream = WebSocketDataStream(config)

        # Register internal handlers
        @stream.on_ticker
        async def handle_ticker(ticker: TickerData):
            await self._process_ticker(ticker)

        @stream.on_trade
        async def handle_trade(trade: TradeData):
            await self._process_trade(trade)

        self._streams[exchange] = stream
        logger.info(f"Added {exchange.value} with {len(symbols)} symbols")

    # =========================================================================
    # Decorators for Event Handlers
    # =========================================================================

    def on_aggregated_ticker(self, handler: Callable):
        """Register handler for aggregated ticker updates."""
        self._aggregated_ticker_handlers.append(handler)
        return handler

    def on_arbitrage(self, handler: Callable):
        """Register handler for arbitrage opportunities."""
        self._arbitrage_handlers.append(handler)
        return handler

    def on_volume_alert(self, handler: Callable):
        """Register handler for volume alerts."""
        self._volume_alert_handlers.append(handler)
        return handler

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start all streams and aggregation."""
        self._running = True

        # Start aggregation task
        asyncio.create_task(self._aggregation_loop())
        asyncio.create_task(self._volume_cleanup_loop())

        # Start all exchange streams
        tasks = [asyncio.create_task(stream.start()) for stream in self._streams.values()]

        logger.info(f"Started aggregator with {len(self._streams)} exchanges")

        # Wait for all streams (they run forever)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """Stop all streams."""
        self._running = False
        for stream in self._streams.values():
            await stream.stop()
        logger.info("Aggregator stopped")

    # =========================================================================
    # Data Processing
    # =========================================================================

    async def _process_ticker(self, ticker: TickerData):
        """Process incoming ticker data with quality checks."""
        symbol = self._normalize_symbol(ticker.symbol)
        
        # 📊 Data Quality Guard
        if not self._is_data_valid(ticker):
            return

        self._tickers[symbol][ticker.exchange] = ticker

    def _is_data_valid(self, ticker: TickerData) -> bool:
        """Validate incoming ticker for outliers or bad data."""
        symbol = self._normalize_symbol(ticker.symbol)
        
        # 1. Zero/Negative price check
        if ticker.last <= 0 or ticker.bid <= 0 or ticker.ask <= 0:
            logger.warning(f"🚩 Data Quality Alert: Non-positive price for {symbol} on {ticker.exchange}")
            return False
            
        # 2. Outlier detection (compared to last known price)
        if symbol in self._tickers and ticker.exchange in self._tickers[symbol]:
            last_ticker = self._tickers[symbol][ticker.exchange]
            price_change_pct = abs(ticker.last - last_ticker.last) / last_ticker.last * 100
            
            # If price jumps > 10% in a sub-second interval, it's likely a bad tick or extreme event
            if price_change_pct > 10.0:
                logger.error(f"🚩 Data Quality Alert: Extreme price jump ({price_change_pct:.2f}%) for {symbol} on {ticker.exchange}")
                return False
                
        return True

    async def _process_trade(self, trade: TradeData):
        """Process incoming trade data."""
        symbol = self._normalize_symbol(trade.symbol)

        # Add to recent trades
        self._recent_trades[symbol].append(trade)
        if len(self._recent_trades[symbol]) > self._max_recent_trades:
            self._recent_trades[symbol] = self._recent_trades[symbol][-self._max_recent_trades :]

        # Update volume statistics
        if symbol not in self._trade_volumes:
            self._trade_volumes[symbol] = TradeVolume(symbol=symbol)

        vol = self._trade_volumes[symbol]
        value = trade.price * trade.quantity

        if trade.side == "buy":
            vol.buy_volume += trade.quantity
            vol.buy_count += 1
        else:
            vol.sell_volume += trade.quantity
            vol.sell_count += 1

        vol.total_value += value

        # Update VWAP
        total_volume = vol.buy_volume + vol.sell_volume
        if total_volume > 0:
            vol.vwap = vol.total_value / total_volume

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format."""
        return symbol.upper().replace("-", "/").replace("_", "/")

    # =========================================================================
    # Aggregation
    # =========================================================================

    async def _aggregation_loop(self):
        """Main aggregation loop."""
        while self._running:
            try:
                await self._aggregate_and_emit()
                await asyncio.sleep(self.aggregation_interval)
            except Exception as e:
                logger.error(f"Aggregation error: {e}")

    async def _aggregate_and_emit(self):
        """Aggregate data and emit to handlers."""
        for symbol, exchange_tickers in self._tickers.items():
            if not exchange_tickers:
                continue

            aggregated = self._aggregate_ticker(symbol, exchange_tickers)

            # Emit to handlers
            for handler in self._aggregated_ticker_handlers:
                await handler(aggregated)

    def _aggregate_ticker(
        self, symbol: str, exchange_tickers: dict[str, TickerData]
    ) -> AggregatedTicker:
        """Aggregate tickers from multiple exchanges."""
        # Find best bid and ask
        best_bid = 0.0
        best_bid_exchange = ""
        best_ask = float("inf")
        best_ask_exchange = ""
        total_volume = 0.0
        weighted_price = 0.0

        for exchange, ticker in exchange_tickers.items():
            if ticker.bid > best_bid:
                best_bid = ticker.bid
                best_bid_exchange = exchange

            if ticker.ask < best_ask:
                best_ask = ticker.ask
                best_ask_exchange = exchange

            total_volume += ticker.volume_24h
            weighted_price += ticker.last * ticker.volume_24h

        vwap = weighted_price / total_volume if total_volume > 0 else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0

        # Check reliability (Stale data check)
        is_reliable = True
        reason = None
        now = time.time()
        
        # If the latest update in ANY exchange is older than 5 seconds, mark as unreliable
        latest_update = max(t.timestamp for t in exchange_tickers.values()) if exchange_tickers else 0
        if now - latest_update > 5.0:
            is_reliable = False
            reason = f"Stale data: last update {now - latest_update:.1f}s ago"

        aggregated_ticker = AggregatedTicker(
            symbol=symbol,
            best_bid=best_bid,
            best_bid_exchange=best_bid_exchange,
            best_ask=best_ask,
            best_ask_exchange=best_ask_exchange,
            spread=spread,
            spread_pct=spread_pct,
            exchanges=exchange_tickers.copy(),
            vwap=vwap,
            total_volume_24h=total_volume,
            timestamp=now,
            is_reliable=is_reliable,
            reliability_reason=reason
        )

        if aggregated_ticker.arbitrage_opportunity:
            for handler in self._arbitrage_handlers:
                asyncio.create_task(handler(aggregated_ticker))
        
        return aggregated_ticker


    async def _volume_cleanup_loop(self):
        """Periodically reset volume windows."""
        while self._running:
            await asyncio.sleep(self._volume_window_seconds)

            # Reset volume statistics
            for symbol in self._trade_volumes:
                self._trade_volumes[symbol] = TradeVolume(symbol=symbol)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_aggregated_ticker(self, symbol: str) -> AggregatedTicker | None:
        """Get current aggregated ticker for symbol."""
        symbol = self._normalize_symbol(symbol)
        if symbol not in self._tickers:
            return None
        return self._aggregate_ticker(symbol, self._tickers[symbol])

    def get_trade_volume(self, symbol: str) -> TradeVolume | None:
        """Get current trade volume statistics."""
        symbol = self._normalize_symbol(symbol)
        return self._trade_volumes.get(symbol)

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[TradeData]:
        """Get recent trades for symbol."""
        symbol = self._normalize_symbol(symbol)
        return self._recent_trades.get(symbol, [])[-limit:]

    def get_all_symbols(self) -> list[str]:
        """Get list of all tracked symbols."""
        return list(self._tickers.keys())

    # =========================================================================
    # Health & Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get aggregator statistics."""
        stream_stats = {
            exchange.value: stream.get_stats() for exchange, stream in self._streams.items()
        }

        return {
            "exchanges": len(self._streams),
            "symbols": len(self._tickers),
            "stream_stats": stream_stats,
            "volume_windows": len(self._trade_volumes),
        }

    async def health_check(self) -> dict[str, Any]:
        """Check aggregator health."""
        stream_health = {}
        all_healthy = True

        for exchange, stream in self._streams.items():
            health = await stream.health_check()
            stream_health[exchange.value] = health
            if health["status"] != "healthy":
                all_healthy = False

        return {
            "service": "data_aggregator",
            "status": "healthy" if all_healthy else "degraded",
            "streams": stream_health,
            "stats": self.get_stats(),
        }
