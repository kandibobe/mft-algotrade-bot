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
- Orderbook imbalance calculation (L2)

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from .data_stream import Exchange, StreamConfig, TickerData, TradeData, WebSocketDataStream
from .data_types import OrderbookData

logger = logging.getLogger(__name__)


@dataclass
class AggregatedTicker:
    """Aggregated ticker from multiple exchanges with L2 metrics."""

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
    imbalance: float = 0.0  # L2 Imbalance metric
    trade_flow_imbalance: float = 0.0  # Trade flow (buy vol - sell vol)
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
    """

    def __init__(self, aggregation_interval: float = 0.1):
        self.aggregation_interval = aggregation_interval
        self._streams: dict[Exchange, WebSocketDataStream] = {}
        self._running = False

        # Data storage
        self._tickers: dict[str, dict[str, TickerData]] = defaultdict(dict)
        self._orderbooks: dict[str, dict[str, OrderbookData]] = defaultdict(dict)
        self._trade_volumes: dict[str, TradeVolume] = {}
        self._recent_trades: dict[str, list[TradeData]] = defaultdict(list)

        # Callbacks
        self._aggregated_ticker_handlers: list[Callable] = []
        self._arbitrage_handlers: list[Callable] = []
        self._volume_alert_handlers: list[Callable] = []

        # Configuration
        self._volume_window_seconds = 60
        self._max_recent_trades = 1000

        # Background tasks
        self._background_tasks = set()

    def add_exchange(
        self, exchange: Exchange, symbols: list[str], channels: list[str] | None = None
    ):
        """Add exchange stream to aggregator."""
        config = StreamConfig(
            exchange=exchange,
            symbols=symbols,
            channels=channels or ["ticker", "trade", "orderbook"],
        )
        stream = WebSocketDataStream(config)

        # Register internal handlers
        @stream.on_ticker
        async def handle_ticker(ticker: TickerData):
            await self._process_ticker(ticker)

        @stream.on_trade
        async def handle_trade(trade: TradeData):
            await self._process_trade(trade)

        @stream.on_orderbook
        async def handle_orderbook(orderbook: OrderbookData):
            await self._process_orderbook(orderbook)

        self._streams[exchange] = stream
        logger.info(f"Added {exchange.value} with {len(symbols)} symbols")

    def on_aggregated_ticker(self, handler: Callable):
        """Register handler for aggregated ticker updates."""
        self._aggregated_ticker_handlers.append(handler)
        return handler

    async def start(self):
        """Start all streams and aggregation."""
        self._running = True

        task = asyncio.create_task(self._aggregation_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        task2 = asyncio.create_task(self._volume_cleanup_loop())
        self._background_tasks.add(task2)
        task2.add_done_callback(self._background_tasks.discard)

        tasks = [asyncio.create_task(stream.start()) for stream in self._streams.values()]
        logger.info(f"Started aggregator with {len(self._streams)} exchanges")
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """Stop all streams."""
        self._running = False
        for stream in self._streams.values():
            await stream.stop()
        logger.info("Aggregator stopped")

    async def _process_ticker(self, ticker: TickerData):
        """Process incoming ticker data."""
        symbol = self._normalize_symbol(ticker.symbol)
        if self._is_data_valid(ticker):
            # Check for extreme outliers against current state
            if exchange_data := self._tickers.get(symbol, {}).get(ticker.exchange):
                if abs(ticker.last - exchange_data.last) / exchange_data.last > 0.4:  # 40% jump
                    logger.warning(
                        f"Rejected outlier ticker for {symbol} on {ticker.exchange}: {ticker.last} vs {exchange_data.last}"
                    )
                    return

            self._tickers[symbol][ticker.exchange] = ticker

    async def _process_trade(self, trade: TradeData):
        """Process incoming trade data."""
        symbol = self._normalize_symbol(trade.symbol)
        self._recent_trades[symbol].append(trade)
        if len(self._recent_trades[symbol]) > self._max_recent_trades:
            self._recent_trades[symbol] = self._recent_trades[symbol][-self._max_recent_trades :]

        if symbol not in self._trade_volumes:
            self._trade_volumes[symbol] = TradeVolume(symbol=symbol)
        vol = self._trade_volumes[symbol]
        vol.total_value += trade.price * trade.quantity
        if trade.side == "buy":
            vol.buy_volume += trade.quantity
        else:
            vol.sell_volume += trade.quantity
        total_volume = vol.buy_volume + vol.sell_volume
        if total_volume > 0:
            vol.vwap = vol.total_value / total_volume

    async def _process_orderbook(self, orderbook: OrderbookData):
        """Process incoming orderbook data."""
        symbol = self._normalize_symbol(orderbook.symbol)
        self._orderbooks[symbol][orderbook.exchange] = orderbook

    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.upper().replace("-", "/").replace("_", "/")

    async def _aggregation_loop(self):
        while self._running:
            try:
                await self._aggregate_and_emit()
                await asyncio.sleep(self.aggregation_interval)
            except Exception as e:
                logger.error(f"Aggregation error: {e}")

    async def _aggregate_and_emit(self):
        for symbol, exchange_tickers in self._tickers.items():
            if not exchange_tickers:
                continue
            aggregated = self._aggregate_ticker(symbol, exchange_tickers)
            for handler in self._aggregated_ticker_handlers:
                await handler(aggregated)

    def _aggregate_ticker(
        self, symbol: str, exchange_tickers: dict[str, TickerData]
    ) -> AggregatedTicker:
        best_bid = 0.0
        best_bid_exchange = ""
        best_ask = float("inf")
        best_ask_exchange = ""
        total_volume = 0.0
        weighted_price = 0.0

        # Calculate imbalance from orderbooks if available
        avg_imbalance = 0.0
        if symbol in self._orderbooks:
            obs = self._orderbooks[symbol]
            if obs:
                avg_imbalance = sum(ob.imbalance for ob in obs.values()) / len(obs)

        # Outlier Detection (Median-based)
        import statistics

        reliability_issues = []

        valid_bids = []
        valid_asks = []

        all_bids = [t.bid for t in exchange_tickers.values() if t.bid > 0]
        all_asks = [t.ask for t in exchange_tickers.values() if t.ask > 0]

        if len(all_bids) >= 3:
            median_bid = statistics.median(all_bids)
            valid_bids = [b for b in all_bids if abs(b - median_bid) / median_bid < 0.05]
            if len(valid_bids) < len(all_bids):
                reliability_issues.append("Outlier bids detected")
        else:
            valid_bids = all_bids

        if len(all_asks) >= 3:
            median_ask = statistics.median(all_asks)
            valid_asks = [a for a in all_asks if abs(a - median_ask) / median_ask < 0.05]
            if len(valid_asks) < len(all_asks):
                reliability_issues.append("Outlier asks detected")
        else:
            valid_asks = all_asks

        # Aggregation loop
        for exchange, ticker in exchange_tickers.items():
            # Skip if filtered out
            if ticker.bid not in valid_bids and len(all_bids) >= 3:
                continue
            if ticker.ask not in valid_asks and len(all_asks) >= 3:
                continue

            if ticker.bid > best_bid:
                best_bid = ticker.bid
                best_bid_exchange = exchange
            if ticker.ask < best_ask:
                best_ask = ticker.ask
                best_ask_exchange = exchange
            total_volume += ticker.volume_24h
            weighted_price += ticker.last * ticker.volume_24h

        vwap = weighted_price / total_volume if total_volume > 0 else 0

        if best_bid == 0 or best_ask == float("inf"):
            # Fallback if everything was filtered or empty
            spread = 0.0
            spread_pct = 0.0
            is_reliable = False
            reliability_issues.append("Insufficient valid data")
        else:
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0

        now = time.time()

        latest_update = (
            max(t.timestamp for t in exchange_tickers.values()) if exchange_tickers else 0
        )
        if now - latest_update >= 5.0:
            reliability_issues.append("Stale data")

        is_reliable = len(reliability_issues) == 0
        reliability_reason = "; ".join(reliability_issues) if reliability_issues else ""

        # Calculate trade flow imbalance if available
        trade_flow = 0.0
        if symbol in self._trade_volumes:
            vol = self._trade_volumes[symbol]
            total = vol.buy_volume + vol.sell_volume
            if total > 0:
                trade_flow = (vol.buy_volume - vol.sell_volume) / total

        return AggregatedTicker(
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
            imbalance=avg_imbalance,
            trade_flow_imbalance=trade_flow,
            is_reliable=is_reliable,
            reliability_reason=reliability_reason,
        )

    def _is_data_valid(self, ticker: TickerData) -> bool:
        return ticker.last > 0 and ticker.bid > 0 and ticker.ask > 0

    async def _volume_cleanup_loop(self):
        while self._running:
            await asyncio.sleep(self._volume_window_seconds)
            for symbol in self._trade_volumes:
                self._trade_volumes[symbol] = TradeVolume(symbol=symbol)

    def get_aggregated_ticker(self, symbol: str) -> AggregatedTicker | None:
        symbol = self._normalize_symbol(symbol)
        if symbol not in self._tickers:
            return None
        return self._aggregate_ticker(symbol, self._tickers[symbol])
