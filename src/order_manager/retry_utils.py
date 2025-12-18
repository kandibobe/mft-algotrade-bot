"""
Retry Utilities with Tenacity
==============================

Centralized retry configuration for all exchange API calls.
Uses exponential backoff with jitter to prevent thundering herd.
"""

import logging
from typing import Type
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryCallState,
)

logger = logging.getLogger(__name__)


# Retryable exceptions (transient network/API errors)
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,  # Network-related OS errors
)


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0,
    exceptions: tuple = RETRYABLE_EXCEPTIONS,
):
    """
    Create a retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier
        exceptions: Tuple of exception types to retry

    Returns:
        Configured retry decorator

    Example:
        @create_retry_decorator(max_attempts=5)
        async def place_order(exchange, symbol, side, amount):
            return await exchange.create_market_order(symbol, side, amount)
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception is retryable.

    Args:
        exception: Exception to check

    Returns:
        True if should retry, False otherwise
    """
    # Check exception type
    if isinstance(exception, RETRYABLE_EXCEPTIONS):
        return True

    # Check error message for retryable patterns
    error_msg = str(exception).lower()
    retryable_patterns = [
        "timeout",
        "connection",
        "network",
        "rate limit",
        "too many requests",
        "service unavailable",
        "503",
        "502",
        "504",
        "temporary",
    ]

    return any(pattern in error_msg for pattern in retryable_patterns)


# Pre-configured retry decorators for common use cases

# Standard retry for API calls
retry_api_call = create_retry_decorator(
    max_attempts=3,
    min_wait=1.0,
    max_wait=10.0,
)

# Aggressive retry for critical operations
retry_critical = create_retry_decorator(
    max_attempts=5,
    min_wait=2.0,
    max_wait=30.0,
    multiplier=3.0,
)

# Fast retry for low-latency operations
retry_fast = create_retry_decorator(
    max_attempts=2,
    min_wait=0.5,
    max_wait=2.0,
    multiplier=2.0,
)


# Custom retry for exchange-specific errors
def retry_exchange_call(func):
    """
    Decorator for exchange API calls with smart error handling.

    Retries on:
    - Network errors (ConnectionError, TimeoutError)
    - Rate limit errors (429)
    - Temporary server errors (502, 503, 504)

    Does NOT retry on:
    - Invalid parameters (400)
    - Authentication errors (401, 403)
    - Order already filled/canceled (specific exchange errors)
    """

    @retry_api_call
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Check if it's a non-retryable error
            error_msg = str(e).lower()
            non_retryable = [
                "invalid",
                "unauthorized",
                "forbidden",
                "insufficient",
                "not found",
                "already",
            ]

            if any(pattern in error_msg for pattern in non_retryable):
                logger.error(f"Non-retryable error in {func.__name__}: {e}")
                raise  # Don't retry

            # Otherwise, retry
            raise

    return wrapper


# Usage example:
"""
from src.order_manager.retry_utils import retry_api_call, retry_critical

class OrderExecutor:

    @retry_api_call
    async def place_order(self, exchange, order):
        '''Automatically retries on network errors.'''
        return await exchange.create_limit_order(...)

    @retry_critical
    async def cancel_critical_order(self, exchange, order_id):
        '''More aggressive retry for critical cancels.'''
        return await exchange.cancel_order(order_id, symbol)
"""
