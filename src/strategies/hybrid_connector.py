"""
Hybrid Connector
================

Bridges Freqtrade Strategy (Synchronous/Threaded) with Websocket Aggregator (AsyncIO).
Enables "MFT" capabilities by injecting real-time orderbook metrics into the decision loop.
"""

import asyncio
import logging
import threading

from src.order_manager.smart_order_executor import SmartOrderExecutor
from src.websocket.aggregator import AggregatedTicker, DataAggregator
from src.websocket.data_stream import Exchange

logger = logging.getLogger(__name__)


class HybridConnectorMixin:
    """
    Mixin for Freqtrade Strategies to enable Real-Time data access.
    """

    _aggregator: DataAggregator | None = None
    _executor: SmartOrderExecutor | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None
    _metrics_cache: dict[str, AggregatedTicker] = {}
    _cache_lock = threading.Lock()

    def __getstate__(self):
        """Custom pickling to avoid unpicklable objects."""
        state = self.__dict__.copy()
        for key in ('_loop', '_thread', '_cache_lock', '_aggregator', '_executor'):
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()

    def initialize_hybrid_connector(
        self,
        pairs: list[str],
        exchange_name: str | None = None,
        shadow_mode: bool = False,
        risk_manager: "RiskManager | None" = None
    ) -> None:
        """
        Initialize the Hybrid Connector and start the Websocket Aggregator in a background thread.
        
        This method acts as the entry point for bridging the synchronous strategy loop
        with the asynchronous execution layer. It sets up the event loop, aggregator,
        and order executor.

        Args:
            pairs: List of trading pairs to monitor.
            exchange_name: Optional override for exchange name. Defaults to config.
            shadow_mode: If True, executes in shadow mode (no real orders).
            risk_manager: Optional RiskManager instance for pre-trade checks.
        
        Raises:
            Exception: If configuration loading fails.
        """
        # 1. Skip if running in Backtest or Hyperopt modes (Simulated only)
        if self.dp.runmode.value in ("backtest", "hyperopt"):
            logger.info("Hybrid Connector disabled in Backtest/Hyperopt mode.")
            return

        # 2. Prevent double initialization
        if self._aggregator is not None:
            logger.warning("Hybrid Connector already initialized.")
            return

        logger.info("🚀 Initializing Hybrid Connector (Websocket Bridge)...")

        # 3. Create a dedicated asyncio loop for the background thread
        self._loop = asyncio.new_event_loop()
        self._aggregator = DataAggregator()

        # 4. Load Configuration safely
        from src.config.manager import ConfigurationManager

        try:
            config = ConfigurationManager.get_config()
            exchange_config = {
                "name": config.exchange.name,
                "key": config.exchange.api_key,
                "secret": config.exchange.api_secret,
            }
            dry_run = config.dry_run
            target_exchange = exchange_name or config.exchange.name
        except Exception as e:
            logger.critical(f"Failed to load configuration for Hybrid Connector: {e}", exc_info=True)
            # Re-raising is appropriate here as the system cannot function without config
            raise

        # 5. Initialize Smart Order Executor
        self._executor = SmartOrderExecutor(
            aggregator=self._aggregator,
            exchange_config=exchange_config,
            dry_run=dry_run,
            shadow_mode=shadow_mode,
            risk_manager=risk_manager
        )

        # 6. Add Exchange and Pairs to Aggregator
        try:
            exch_enum = Exchange(target_exchange.lower())
            self._aggregator.add_exchange(exch_enum, pairs)
        except ValueError:
            logger.error(f"Exchange '{target_exchange}' is not supported by the Aggregator. "
                         f"Supported exchanges: {[e.value for e in Exchange]}")
            return

        # 7. Register callback to update local cache
        @self._aggregator.on_aggregated_ticker
        async def on_ticker(ticker: AggregatedTicker):
            with self._cache_lock:
                self._metrics_cache[ticker.symbol] = ticker

        # 8. Start the AsyncIO loop in a separate daemon thread
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True, name="HybridConnectorThread")
        self._thread.start()
        logger.info("Hybrid Connector started successfully.")

    def _run_async_loop(self):
        """Internal method to run the asyncio loop."""
        asyncio.set_event_loop(self._loop)

        tasks = [self._aggregator.start()]
        if self._executor:
            tasks.append(self._executor.start())

        self._loop.run_until_complete(asyncio.gather(*tasks))

    def get_realtime_metrics(self, pair: str) -> AggregatedTicker | None:
        """Non-blocking access to the latest websocket data."""
        normalized_pair = pair.upper().replace("_", "/")
        with self._cache_lock:
            return self._metrics_cache.get(normalized_pair)

    def check_market_safety(self, pair: str, side: str) -> bool:
        """MFT Safety Check."""
        if self.dp.runmode.value in ("backtest", "hyperopt"):
            return True

        ticker = self.get_realtime_metrics(pair)
        if not ticker:
            return True

        if not ticker.is_reliable:
            logger.warning(f"🚫 Trade rejected: Data unreliable for {pair}.")
            return False

        if ticker.spread_pct > 0.5:
            return False

        return True

    def get_orderbook_imbalance(self, pair: str) -> float:
        """Calculate simple orderbook imbalance from aggregated ticker."""
        ticker = self.get_realtime_metrics(pair)
        if not ticker:
            return 0.0
        return getattr(ticker, 'imbalance', 0.0)
            # Let's log warning and allow for now, to avoid blocking startup trades
