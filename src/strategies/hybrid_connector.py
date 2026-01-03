"""
Hybrid Connector
================

Bridges Freqtrade Strategy (Synchronous/Threaded) with Websocket Aggregator (AsyncIO).
Enables "MFT" capabilities by injecting real-time orderbook metrics into the decision loop.
"""

import asyncio
import logging
import threading
from typing import Dict, Optional, Any
from datetime import datetime

from src.websocket.aggregator import DataAggregator, AggregatedTicker
from src.websocket.data_stream import Exchange
from src.order_manager.smart_order_executor import SmartOrderExecutor

logger = logging.getLogger(__name__)

class HybridConnectorMixin:
    """
    Mixin for Freqtrade Strategies to enable Real-Time data access.
    """
    
    _aggregator: Optional[DataAggregator] = None
    _executor: Optional[SmartOrderExecutor] = None
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _thread: Optional[threading.Thread] = None
    _metrics_cache: Dict[str, AggregatedTicker] = {}
    _cache_lock = threading.Lock()
    
    def initialize_hybrid_connector(self, pairs: list[str], exchange_name: str = "binance"):
        """
        Start the Websocket Aggregator in a background thread.
        Should be called from bot_start().
        """
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            logger.info("Hybrid Connector disabled in Backtest/Hyperopt mode.")
            return

        if self._aggregator is not None:
            logger.warning("Hybrid Connector already initialized.")
            return

        logger.info("🚀 Initializing Hybrid Connector (Websocket Bridge)...")
        
        # Create a new event loop for the background thread
        self._loop = asyncio.new_event_loop()
        self._aggregator = DataAggregator()
        
        # Initialize Executor with config
        from src.config.manager import ConfigurationManager
        dry_run = True # Default safe
        try:
            config = ConfigurationManager.get_config()
            exchange_config = {
                'name': config.exchange.name,
                'key': config.exchange.api_key,
                'secret': config.exchange.api_secret,
            }
            dry_run = config.dry_run
        except Exception:
            logger.warning("Could not load exchange config for SmartExecutor, execution disabled.")
            exchange_config = None

        self._executor = SmartOrderExecutor(
            aggregator=self._aggregator, 
            exchange_config=exchange_config,
            dry_run=dry_run
        )
        
        # Add exchange
        # Map string name to Enum
        try:
            exch_enum = Exchange(exchange_name.lower())
            self._aggregator.add_exchange(exch_enum, pairs)
        except ValueError:
            logger.error(f"Exchange {exchange_name} not supported by Aggregator.")
            return

        # Register callback to update local cache
        @self._aggregator.on_aggregated_ticker
        async def on_ticker(ticker: AggregatedTicker):
            with self._cache_lock:
                self._metrics_cache[ticker.symbol] = ticker

        # Start the loop in a separate thread
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        logger.info("Hybrid Connector started successfully.")

    def _run_async_loop(self):
        """Internal method to run the asyncio loop."""
        asyncio.set_event_loop(self._loop)
        
        tasks = [self._aggregator.start()]
        if self._executor:
            tasks.append(self._executor.start())
            
        self._loop.run_until_complete(asyncio.gather(*tasks))

    def get_realtime_metrics(self, pair: str) -> Optional[AggregatedTicker]:
        """
        Non-blocking access to the latest websocket data.
        """
        # Normalize pair (BTC/USDT -> BTC/USDT) - Aggregator handles normalization internally
        # but let's be safe
        normalized_pair = pair.upper().replace('_', '/')
        with self._cache_lock:
            return self._metrics_cache.get(normalized_pair)

    def check_market_safety(self, pair: str, side: str) -> bool:
        """
        MFT Safety Check.
        Returns True if safe to trade, False otherwise.
        """
        # In backtest, we assume safety (or rely on backtest data)
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            return True
            
        ticker = self.get_realtime_metrics(pair)
        if not ticker:
            # If no data yet, maybe allow? Or be safe and deny?
            # Let's log warning and allow for now, to avoid blocking startup trades
            logger.warning(f"No real-time data for {pair} yet.")
            return True
            
        # 1. Spread Check
        if ticker.spread_pct > 0.5: # 0.5% spread is huge for MFT
            logger.info(f"🚫 trade rejected: Spread too high ({ticker.spread_pct:.2f}%)")
            return False
            
        # 2. Arbitrage/Cross-Exchange Integrity (Optional)
        if ticker.arbitrage_opportunity:
            logger.info(f"⚡ Arbitrage opportunity detected for {pair}! Profit: {ticker.arbitrage_profit_pct:.2f}%")
            
        return True
