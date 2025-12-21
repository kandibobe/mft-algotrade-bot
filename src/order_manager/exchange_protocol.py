"""
Exchange API Protocol
====================

Type-safe protocol for exchange API interactions.
Ensures all exchange implementations follow the same interface.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ExchangeAPI(Protocol):
    """
    Protocol for exchange API implementations.

    All exchange adapters (CCXT, custom REST, etc.) must implement these methods.
    This ensures type safety and makes testing easier with mocks.

    Usage:
        def execute_order(exchange: ExchangeAPI, symbol: str):
            result = exchange.create_limit_order(symbol, "buy", 1.0, 50000.0)
            # mypy will verify exchange has this method!
    """

    async def create_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> Dict[str, Any]:
        """
        Create a limit order.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            amount: Quantity to trade
            price: Limit price

        Returns:
            Order response with at least:
            {
                "id": str,           # Order ID
                "status": str,       # "open", "closed", "canceled"
                "filled": float,     # Filled quantity
                "average": float,    # Average fill price
            }
        """
        ...

    async def create_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """
        Create a market order.

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            amount: Quantity to trade

        Returns:
            Order response (same format as create_limit_order)
        """
        ...

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair

        Returns:
            Cancellation response

        Raises:
            OrderNotFound: If order doesn't exist
        """
        ...

    async def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Fetch order status.

        Args:
            order_id: Order ID
            symbol: Trading pair

        Returns:
            Order info with:
            {
                "id": str,
                "status": str,       # "open", "closed", "canceled", "expired"
                "filled": float,     # Filled quantity
                "remaining": float,  # Remaining quantity
                "average": float,    # Average fill price
                "fee": {
                    "cost": float,
                    "currency": str
                }
            }
        """
        ...

    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch current order book.

        Args:
            symbol: Trading pair
            limit: Number of levels to fetch

        Returns:
            Order book with:
            {
                "bids": [[price, size], ...],  # Sorted high to low
                "asks": [[price, size], ...],  # Sorted low to high
                "timestamp": int,
                "datetime": str
            }
        """
        ...


class SyncExchangeAPI(Protocol):
    """
    Synchronous version of ExchangeAPI for backward compatibility.

    Note: Prefer async version for production use!
    """

    def create_limit_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> Dict[str, Any]: ...

    def create_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]: ...

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]: ...

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]: ...

    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, Any]: ...


# Type aliases for order responses
OrderResponse = Dict[str, Any]
OrderBookResponse = Dict[str, Any]
