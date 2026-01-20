"""
Async Data Fetcher
==================

Asynchronous data fetching using ccxt.async_support with retry logic.

Key Features:
1. Non-blocking async/await for concurrent data fetching
2. Automatic retry with exponential backoff (tenacity)
3. Rate limiting to avoid exchange bans
4. Concurrent multi-pair fetching

Usage:
    async with AsyncDataFetcher(exchange='binance') as fetcher:
        # Fetch single pair
        df = await fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=1000)

        # Fetch multiple pairs concurrently
        data = await fetcher.fetch_multiple(['BTC/USDT', 'ETH/USDT'], '1h')
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

try:
    import ccxt.async_support as ccxt_async

    CCXT_ASYNC_AVAILABLE = True
except ImportError:
    CCXT_ASYNC_AVAILABLE = False
    ccxt_async = None

try:
        from tenacity import (
            before_sleep_log,
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        __all_tenacity__ = [
            "before_sleep_log",
            "retry",
            "retry_if_exception_type",
            "stop_after_attempt",
            "wait_exponential",
        ]

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FetcherConfig:
    """Configuration for async data fetcher."""

    # Exchange settings
    exchange: str = "binance"
    sandbox: bool = False  # Use testnet

    # API credentials (optional for public data)
    api_key: str | None = None
    api_secret: str | None = None

    # Rate limiting
    rate_limit: bool = True
    requests_per_second: float = 10.0

    # Retry settings
    max_retries: int = 5
    retry_min_wait: float = 1.0  # seconds
    retry_max_wait: float = 60.0  # seconds

    # Timeout settings
    timeout: int = 30000  # milliseconds

    # Concurrency
    max_concurrent_requests: int = 5


class AsyncDataFetcher:
    """
    Asynchronous data fetcher with retry logic.

    Problem: Synchronous API calls block the entire bot while waiting for response.
    Solution: Use async/await for non-blocking I/O.

    Usage:
        async with AsyncDataFetcher() as fetcher:
            df = await fetcher.fetch_ohlcv('BTC/USDT', '1h')

        # Or without context manager:
        fetcher = AsyncDataFetcher()
        await fetcher.connect()
        df = await fetcher.fetch_ohlcv('BTC/USDT', '1h')
        await fetcher.close()
    """

    def __init__(self, config: FetcherConfig | None = None):
        """Initialize async fetcher."""
        if not CCXT_ASYNC_AVAILABLE:
            raise ImportError("ccxt.async_support not available. Install with: pip install ccxt")

        self.config = config or FetcherConfig()
        self.exchange: Any | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._connected = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize exchange connection."""
        if self._connected:
            return

        exchange_class = getattr(ccxt_async, self.config.exchange, None)
        if not exchange_class:
            raise ValueError(f"Exchange '{self.config.exchange}' not supported")

        exchange_config = {
            "enableRateLimit": self.config.rate_limit,
            "timeout": self.config.timeout,
        }

        if self.config.api_key:
            exchange_config["apiKey"] = self.config.api_key
        if self.config.api_secret:
            exchange_config["secret"] = self.config.api_secret

        self.exchange = exchange_class(exchange_config)

        if self.config.sandbox:
            self.exchange.set_sandbox_mode(True)

        # Initialize semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        self._connected = True
        logger.info(f"Connected to {self.config.exchange} (async mode)")

    async def close(self) -> None:
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
            self._connected = False
            logger.info("Exchange connection closed")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with automatic retry.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h', '1d')
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch (max usually 1000)

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        if not self._connected:
            await self.connect()

        async with self._semaphore:
            ohlcv = await self._fetch_with_retry(symbol, timeframe, since, limit)

        if not ohlcv:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df

    async def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since: int | None,
        limit: int,
    ) -> list:
        """
        Fetch OHLCV with tenacity retry logic.

        Retries on:
        - Network errors (ConnectionError, TimeoutError)
        - Exchange temporary errors (RateLimitExceeded, ExchangeNotAvailable)
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                return await self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                )

            except (
                asyncio.TimeoutError,
                ConnectionError,
            ) as e:
                last_error = e
                wait_time = min(
                    self.config.retry_min_wait * (2**attempt), self.config.retry_max_wait
                )
                logger.warning(
                    f"Fetch attempt {attempt + 1}/{self.config.max_retries} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)

            except Exception as e:
                # Check for ccxt-specific errors
                error_str = str(e).lower()
                if any(x in error_str for x in ["rate limit", "too many requests", "ddos"]):
                    last_error = e
                    wait_time = min(
                        self.config.retry_min_wait * (2**attempt), self.config.retry_max_wait
                    )
                    logger.warning(
                        f"Rate limited on attempt {attempt + 1}. Waiting {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                elif any(x in error_str for x in ["not available", "maintenance"]):
                    last_error = e
                    wait_time = self.config.retry_max_wait
                    logger.warning(f"Exchange unavailable. Waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # Non-retryable error
                    logger.error(f"Non-retryable error: {e}")
                    raise

        # All retries failed
        logger.error(f"All {self.config.max_retries} retries failed for {symbol}")
        raise last_error or Exception("Unknown fetch error")

    async def fetch_multiple(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        limit: int = 1000,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols concurrently.

        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            limit: Number of candles per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if not self._connected:
            await self.connect()

        tasks = [self.fetch_ohlcv(symbol, timeframe, limit=limit) for symbol in symbols]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {symbol}: {result}")
                data[symbol] = pd.DataFrame()
            else:
                data[symbol] = result

        return data

    async def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Fetch current order book.

        Args:
            symbol: Trading pair
            limit: Depth of order book

        Returns:
            Dictionary with 'bids', 'asks', 'timestamp'
        """
        if not self._connected:
            await self.connect()

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    return await self.exchange.fetch_order_book(symbol, limit)
                except Exception as e:
                    error_str = str(e).lower()
                    if any(x in error_str for x in ["rate limit", "timeout", "connection"]):
                        wait_time = self.config.retry_min_wait * (2**attempt)
                        logger.warning(f"Orderbook fetch failed: {e}. Retrying...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise

        raise Exception(f"Failed to fetch orderbook for {symbol}")

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Fetch current ticker (price, volume, etc.).

        Args:
            symbol: Trading pair

        Returns:
            Ticker dictionary with bid, ask, last, volume, etc.
        """
        if not self._connected:
            await self.connect()

        async with self._semaphore:
            return await self.exchange.fetch_ticker(symbol)

    async def fetch_balance(self) -> dict[str, Any]:
        """
        Fetch account balance (requires API credentials).

        Returns:
            Balance dictionary
        """
        if not self._connected:
            await self.connect()

        if not self.config.api_key:
            raise ValueError("API credentials required for balance fetch")

        async with self._semaphore:
            return await self.exchange.fetch_balance()

    async def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data across multiple API calls.

        Automatically handles pagination for large date ranges.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start datetime
            end_date: End datetime (defaults to now)

        Returns:
            Combined DataFrame with all historical data
        """
        if not self._connected:
            await self.connect()

        end_date = end_date or datetime.now()
        all_data = []

        # Convert timeframe to milliseconds
        tf_ms = self._timeframe_to_ms(timeframe)
        max_candles = 1000  # Most exchanges limit to 1000

        current_start = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        while current_start < end_ts:
            df = await self.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_start,
                limit=max_candles,
            )

            if df.empty:
                break

            all_data.append(df)

            # Move start to after last candle
            last_ts = int(df.index[-1].timestamp() * 1000)
            current_start = last_ts + tf_ms

            # Small delay to respect rate limits
            await asyncio.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, axis=0)
        # Deduplicate and sort
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Ensure timezone-naive comparison if needed
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        if combined.index.tz is not None:
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize(combined.index.tz)
            if end_ts.tz is None:
                end_ts = end_ts.tz_localize(combined.index.tz)

        # Filter to exact date range
        combined = combined[(combined.index >= start_ts) & (combined.index <= end_ts)]

        logger.info(
            f"Fetched {len(combined)} historical candles for {symbol} "
            f"from {start_date} to {end_date}"
        )

        return combined

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        multipliers = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000,
        }

        unit = timeframe[-1]
        value = int(timeframe[:-1])

        return value * multipliers.get(unit, 60 * 1000)


class AsyncOrderExecutor:
    """
    Asynchronous order executor with state machine and retry logic.

    Integrates with SmartLimitExecutor for fee-optimized execution.

    Usage:
        async with AsyncOrderExecutor(exchange='binance') as executor:
            result = await executor.execute_smart_limit(
                symbol='BTC/USDT',
                side='buy',
                quantity=0.1,
            )
    """

    def __init__(
        self,
        config: FetcherConfig | None = None,
        smart_limit_config: Any | None = None,
    ):
        """Initialize async order executor."""
        self.config = config or FetcherConfig()
        self.smart_limit_config = smart_limit_config
        self.exchange: Any | None = None
        self._connected = False
        self._semaphore: asyncio.Semaphore | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize exchange connection with credentials."""
        if self._connected:
            return

        if not self.config.api_key or not self.config.api_secret:
            raise ValueError("API credentials required for order execution")

        exchange_class = getattr(ccxt_async, self.config.exchange, None)
        if not exchange_class:
            raise ValueError(f"Exchange '{self.config.exchange}' not supported")

        self.exchange = exchange_class(
            {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": self.config.rate_limit,
                "timeout": self.config.timeout,
            }
        )

        if self.config.sandbox:
            self.exchange.set_sandbox_mode(True)

        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._connected = True

        logger.info(f"Order executor connected to {self.config.exchange}")

    async def close(self) -> None:
        """Close connection."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
            self._connected = False

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """
        Create limit order with retry logic.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order quantity
            price: Limit price
            params: Additional exchange-specific parameters

        Returns:
            Order result dictionary
        """
        if not self._connected:
            await self.connect()

        async with self._semaphore:
            return await self._execute_with_retry(
                self.exchange.create_limit_order, symbol, side, amount, price, params or {}
            )

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """
        Create market order with retry logic.

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order quantity
            params: Additional parameters

        Returns:
            Order result dictionary
        """
        if not self._connected:
            await self.connect()

        async with self._semaphore:
            return await self._execute_with_retry(
                self.exchange.create_market_order, symbol, side, amount, params or {}
            )

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Cancel order."""
        if not self._connected:
            await self.connect()

        async with self._semaphore:
            return await self._execute_with_retry(self.exchange.cancel_order, order_id, symbol)

    async def fetch_order(
        self,
        order_id: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Fetch order status."""
        if not self._connected:
            await self.connect()

        async with self._semaphore:
            return await self._execute_with_retry(self.exchange.fetch_order, order_id, symbol)

    async def _execute_with_retry(self, func, *args, **kwargs) -> dict[str, Any]:
        """Execute exchange function with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if retryable
                retryable = any(
                    x in error_str
                    for x in [
                        "rate limit",
                        "timeout",
                        "connection",
                        "network",
                        "temporary",
                        "maintenance",
                    ]
                )

                if retryable and attempt < self.config.max_retries - 1:
                    wait_time = min(
                        self.config.retry_min_wait * (2**attempt), self.config.retry_max_wait
                    )
                    logger.warning(
                        f"Order attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise last_error or Exception("Unknown execution error")


# Convenience function for quick data fetching
async def fetch_ohlcv_async(
    symbol: str,
    timeframe: str = "1h",
    exchange: str = "binance",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Quick async OHLCV fetch without manual connection management.

    Usage:
        df = await fetch_ohlcv_async('BTC/USDT', '1h')
    """
    config = FetcherConfig(exchange=exchange)
    async with AsyncDataFetcher(config) as fetcher:
        return await fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)