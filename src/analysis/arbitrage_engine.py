"""
Arbitrage Engine
================

Real-time detection of cross-exchange and triangular arbitrage opportunities.
Integrated with the Data Aggregator for low-latency signals.
"""

import logging
import time
from dataclasses import dataclass

from src.utils.logger import log
from src.websocket.aggregator import AggregatedTicker

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Details of a detected arbitrage opportunity."""

    type: str  # 'cross_exchange' or 'triangular'
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    expected_profit_abs: float
    timestamp: float
    is_actionable: bool = True


class ArbitrageEngine:
    """
    Engine to analyze market data for arbitrage opportunities.
    """

    def __init__(self, min_profit_threshold: float = 0.002):  # 0.2% threshold
        self.min_profit_threshold = min_profit_threshold
        self._last_opportunities: dict[str, ArbitrageOpportunity] = {}

    def analyze_ticker(self, ticker: AggregatedTicker) -> ArbitrageOpportunity | None:
        """
        Analyze a single aggregated ticker for cross-exchange arbitrage.
        """
        if not ticker.arbitrage_opportunity:
            return None

        profit_pct = ticker.arbitrage_profit_pct / 100.0

        if profit_pct >= self.min_profit_threshold:
            opportunity = ArbitrageOpportunity(
                type="cross_exchange",
                symbol=ticker.symbol,
                buy_exchange=ticker.best_ask_exchange,
                sell_exchange=ticker.best_bid_exchange,
                buy_price=ticker.best_ask,
                sell_price=ticker.best_bid,
                profit_pct=profit_pct,
                expected_profit_abs=0.0,  # Needs position size info
                timestamp=time.time(),
            )

            # Actionability check (e.g., volume check)
            # In a real system, we'd check if top of book volume is sufficient

            log.info(
                "arbitrage_detected",
                symbol=ticker.symbol,
                profit_pct=f"{profit_pct:.4%}",
                buy=ticker.best_ask_exchange,
                sell=ticker.best_bid_exchange,
            )

            return opportunity

        return None

    def analyze_triangular(
        self, tickers: dict[str, AggregatedTicker]
    ) -> list[ArbitrageOpportunity]:
        """
        Detect triangular arbitrage (e.g., BTC/USDT -> ETH/BTC -> ETH/USDT).
        Placeholder for Q2 2026 implementation.
        """
        opportunities = []
        # Logic: find closed loops of symbols and calculate product of exchange rates
        return opportunities
