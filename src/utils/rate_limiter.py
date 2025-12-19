"""
Rate Limiter for Exchange API Calls
====================================

Token bucket algorithm implementation to prevent API bans.

Usage:
    from src.utils.rate_limiter import TokenBucketLimiter

    # Binance spot trading limits: 1200 requests/minute
    @TokenBucketLimiter(max_calls=1200, period=60.0)
    def fetch_ticker(symbol):
        return exchange.fetch_ticker(symbol)

    # Or use as context manager:
    limiter = TokenBucketLimiter(max_calls=100, period=60.0)
    with limiter:
        data = exchange.fetch_orderbook('BTC/USDT')

Copyright (c) 2024-2025 Stoic Citadel
PROPRIETARY - All Rights Reserved
"""

import time
import logging
import asyncio
from collections import deque
from functools import wraps
from typing import Callable, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class TokenBucketLimiter:
    """
    Token bucket rate limiter to prevent Exchange API bans.

    Implements a sliding window approach with exponential backoff on limit hits.

    Attributes:
        max_calls: Maximum number of calls allowed in the period
        period: Time period in seconds (e.g., 60.0 for per-minute)
        calls: Deque of timestamps of recent API calls
        lock: Thread lock for concurrent access

    Example:
        # Binance limits:
        # - Spot trading: 1200 requests/minute
        # - Futures: 2400 requests/minute
        # - Order placement: 100 orders/10 seconds

        @TokenBucketLimiter(max_calls=1200, period=60.0)
        def safe_api_call():
            return exchange.fetch_ticker('BTC/USDT')
    """

    def __init__(
        self,
        max_calls: int,
        period: float,
        burst_limit: Optional[int] = None,
        enable_backoff: bool = True
    ):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in period
            period: Time window in seconds
            burst_limit: Optional burst allowance (default: max_calls * 0.2)
            enable_backoff: Whether to use exponential backoff
        """
        self.max_calls = max_calls
        self.period = period
        self.burst_limit = burst_limit or int(max_calls * 0.2)
        self.enable_backoff = enable_backoff

        self.calls = deque()
        self.lock = Lock()

        # Backoff state
        self.consecutive_limits = 0
        self.last_limit_hit = 0.0

        logger.info(
            f"Rate limiter initialized: {max_calls} calls per {period}s "
            f"(burst: {self.burst_limit})"
        )

    def _cleanup_old_calls(self, now: float) -> None:
        """Remove calls outside the current time window."""
        cutoff = now - self.period
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()

    def _calculate_backoff(self) -> float:
        """
        Calculate exponential backoff delay.

        Returns:
            Sleep time in seconds
        """
        if not self.enable_backoff:
            return 0.0

        # Reset if enough time passed since last limit
        if time.time() - self.last_limit_hit > self.period * 2:
            self.consecutive_limits = 0
            return 0.0

        # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s (max)
        backoff = min(0.5 * (2 ** self.consecutive_limits), 8.0)
        return backoff

    def acquire(self) -> None:
        """
        Acquire permission to make an API call.

        Blocks if rate limit would be exceeded.
        """
        with self.lock:
            now = time.time()
            self._cleanup_old_calls(now)

            current_calls = len(self.calls)

            # Check if we're hitting the limit
            if current_calls >= self.max_calls:
                # Calculate required wait time
                oldest_call = self.calls[0]
                sleep_time = self.period - (now - oldest_call)

                # Apply backoff if enabled
                backoff = self._calculate_backoff()
                sleep_time += backoff

                self.consecutive_limits += 1
                self.last_limit_hit = now

                logger.warning(
                    f"Rate limit reached ({current_calls}/{self.max_calls}). "
                    f"Sleeping {sleep_time:.2f}s (backoff: {backoff:.2f}s, "
                    f"consecutive hits: {self.consecutive_limits})"
                )

                time.sleep(sleep_time)

                # Re-cleanup after sleep
                now = time.time()
                self._cleanup_old_calls(now)

            # Reset consecutive limits on success
            if current_calls < self.max_calls:
                self.consecutive_limits = 0

            # Record this call
            self.calls.append(now)

    async def acquire_async(self) -> None:
        """Async version of acquire for async code."""
        # Simple spinlock for async (could be improved with asyncio.Lock)
        while True:
            with self.lock:
                now = time.time()
                self._cleanup_old_calls(now)

                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    self.consecutive_limits = 0
                    break

                # Calculate wait time
                oldest_call = self.calls[0]
                sleep_time = self.period - (now - oldest_call)
                backoff = self._calculate_backoff()
                sleep_time += backoff

                self.consecutive_limits += 1
                self.last_limit_hit = now

                logger.warning(
                    f"Rate limit reached (async). Sleeping {sleep_time:.2f}s"
                )

            await asyncio.sleep(sleep_time)

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dict with current call count, limit, and utilization percentage
        """
        with self.lock:
            now = time.time()
            self._cleanup_old_calls(now)

            current_calls = len(self.calls)
            utilization = (current_calls / self.max_calls) * 100

            return {
                'current_calls': current_calls,
                'max_calls': self.max_calls,
                'period': self.period,
                'utilization_pct': utilization,
                'consecutive_limits': self.consecutive_limits,
                'time_until_reset': self.period if self.calls else 0
            }

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to rate limit a function.

        Usage:
            @TokenBucketLimiter(max_calls=100, period=60.0)
            def my_api_call():
                return exchange.fetch_ticker('BTC/USDT')
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            await self.acquire_async()
            return await func(*args, **kwargs)

        # Return async wrapper if function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False


class ExchangeRateLimiter:
    """
    Pre-configured rate limiters for popular exchanges.

    Usage:
        from src.utils.rate_limiter import ExchangeRateLimiter

        limiter = ExchangeRateLimiter.binance_spot()

        @limiter
        def fetch_data():
            return exchange.fetch_ticker('BTC/USDT')
    """

    @staticmethod
    def binance_spot() -> TokenBucketLimiter:
        """Binance Spot trading limits: 1200 requests/minute."""
        return TokenBucketLimiter(max_calls=1200, period=60.0)

    @staticmethod
    def binance_futures() -> TokenBucketLimiter:
        """Binance Futures limits: 2400 requests/minute."""
        return TokenBucketLimiter(max_calls=2400, period=60.0)

    @staticmethod
    def binance_orders() -> TokenBucketLimiter:
        """Binance order placement: 100 orders/10 seconds."""
        return TokenBucketLimiter(max_calls=100, period=10.0)

    @staticmethod
    def bybit() -> TokenBucketLimiter:
        """Bybit limits: 120 requests/minute."""
        return TokenBucketLimiter(max_calls=120, period=60.0)

    @staticmethod
    def coinbase() -> TokenBucketLimiter:
        """Coinbase Pro limits: 10 requests/second."""
        return TokenBucketLimiter(max_calls=10, period=1.0)

    @staticmethod
    def kraken() -> TokenBucketLimiter:
        """Kraken limits: 15 requests/second."""
        return TokenBucketLimiter(max_calls=15, period=1.0)


# Convenience function for quick usage
def rate_limit(max_calls: int, period: float = 60.0):
    """
    Quick decorator for rate limiting.

    Args:
        max_calls: Maximum calls in period
        period: Time period in seconds (default: 60)

    Example:
        @rate_limit(max_calls=100, period=60.0)
        def my_function():
            pass
    """
    return TokenBucketLimiter(max_calls=max_calls, period=period)
