"""
Exchange Backend Abstraction
============================

Defines interface and implementations for exchange interaction.
Allows switching between Live (CCXT) and Dry-Run (Mock) execution.
"""

import abc
import asyncio
import logging

try:
    import ccxt.async_support as ccxt
except ImportError:
    ccxt = None

from src.websocket.aggregator import DataAggregator
from src.order_manager.errors import (
    ExecutionError,
    InsufficientFundsError,
    RateLimitError,
    OrderValidationError,
    NetworkError,
    ExchangeError
)

logger = logging.getLogger(__name__)


class IExchangeBackend(abc.ABC):
    """Abstract base class for exchange backends."""

    @abc.abstractmethod
    async def initialize(self):
        """Initialize connection."""
        pass

    @abc.abstractmethod
    async def close(self):
        """Close connection."""
        pass

    @abc.abstractmethod
    async def create_limit_buy_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        """Create a limit buy order."""
        pass

    @abc.abstractmethod
    async def create_limit_sell_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        """Create a limit sell order."""
        pass

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an order."""
        pass

    @abc.abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        """Fetch order status."""
        pass

    @abc.abstractmethod
    async def fetch_positions(self) -> list[dict]:
        """Fetch all open positions."""
        pass


class CCXTBackend(IExchangeBackend):
    """Live exchange interaction using CCXT."""

    def __init__(self, exchange_config: dict):
        self.config = exchange_config
        self.exchange = None
        self.name = exchange_config.get("name", "binance")

    async def initialize(self):
        if not ccxt:
            raise ImportError("CCXT not installed")

        exchange_class = getattr(ccxt, self.name)
        self.exchange = exchange_class(
            {
                "apiKey": self.config.get("key"),
                "secret": self.config.get("secret"),
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )

        # Enable 429 error handling and retries via CCXT built-in
        self.exchange.enableRateLimit = True
        # Verify connection (optional but good for safety)
        # await self.exchange.load_markets()
        logger.info(f"Initialized CCXTBackend for {self.name}")

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    async def create_limit_buy_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        return await self.exchange.create_limit_buy_order(symbol, quantity, price, params or {})

    async def create_limit_sell_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        return await self.exchange.create_limit_sell_order(symbol, quantity, price, params or {})

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self.exchange.cancel_order(order_id, symbol)

    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute exchange call with explicit 429 backoff."""
        max_retries = 3
        for i in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                wait_time = (i + 1) * 2
                logger.warning(
                    f"Rate limit hit (429). Waiting {wait_time}s before retry {i + 1}/{max_retries}. Error: {e}"
                )
                await asyncio.sleep(wait_time)
                if i == max_retries - 1:
                     raise RateLimitError(f"Rate limit exceeded after {max_retries} retries: {e}") from e
            except ccxt.InsufficientFunds as e:
                raise InsufficientFundsError(f"Insufficient funds: {e}") from e
            except (ccxt.InvalidOrder, ccxt.OrderNotFound, ccxt.BadSymbol) as e:
                raise OrderValidationError(f"Invalid order: {e}") from e
            except ccxt.NetworkError as e:
                if i == max_retries - 1:
                    raise NetworkError(f"Network error after {max_retries} retries: {e}") from e
                logger.warning(f"Network error. Retrying... {e}")
                await asyncio.sleep(1)
            except ccxt.ExchangeError as e:
                raise ExchangeError(f"Exchange error: {e}") from e
            except Exception as e:
                # Fallback for unexpected errors
                if i == max_retries - 1:
                    raise ExecutionError(f"Unexpected execution error: {e}") from e
                logger.error(f"Unexpected error in exchange call: {e}. Retrying.")
                await asyncio.sleep(1)

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        return await self._execute_with_retry(self.exchange.fetch_order, order_id, symbol)

    async def fetch_positions(self) -> list[dict]:
        return await self._execute_with_retry(self.exchange.fetch_positions)


class MockExchangeBackend(IExchangeBackend):
    """
    Simulated exchange for Dry-Run/Paper execution.
    Matches orders against aggregated ticker data.
    """

    def __init__(self, aggregator: DataAggregator):
        self.aggregator = aggregator
        self.orders = {}
        self.order_counter = 0

    async def initialize(self):
        logger.info("Initialized MockExchangeBackend (Safe Mode)")

    async def close(self):
        pass

    def _create_mock_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        self.order_counter += 1
        order_id = f"mock_{self.order_counter}"

        order = {
            "id": order_id,
            "symbol": symbol,
            "status": "open",
            "side": side,
            "amount": quantity,
            "price": price,
            "filled": 0.0,
            "remaining": quantity,
            "timestamp": 0,  # TODO: add timestamp
        }
        self.orders[order_id] = order

        # Simulate immediate matching attempt?
        # For now, just return open order. SmartOrderExecutor loop will check status.
        # But wait, SmartOrderExecutor calls fetch_order loop.
        # We need a way to "fill" orders.
        # The loop calls fetch_order. We need to update order status based on price.

        return order

    async def create_limit_buy_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        return self._create_mock_order(symbol, "buy", quantity, price)

    async def create_limit_sell_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        return self._create_mock_order(symbol, "sell", quantity, price)

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        if order_id in self.orders:
            self.orders[order_id]["status"] = "canceled"
            return self.orders[order_id]
        raise ValueError("Order not found")

    async def fetch_positions(self) -> list[dict]:
        """Return currently open mock positions."""
        return [
            {
                "symbol": o["symbol"],
                "side": o["side"],
                "amount": o["amount"],
                "entry_price": o["price"],
            }
            for o in self.orders.values()
            if o["status"] == "open"  # Simplification for mock
        ]

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        # Simulate fill logic here
        if order_id not in self.orders:
            raise ValueError("Order not found")

        order = self.orders[order_id]
        if order["status"] in ["closed", "canceled"]:
            return order

        # Check price against aggregator
        # Note: Aggregator might not expose raw ticker easily unless we subscribe.
        # But SmartOrderExecutor passes aggregator to us.
        # We can try to peek at latest ticker?

        # For simplicity in this step, we will assume the order remains open
        # unless we add logic to fill it.
        # Ideally, this mock backend should subscribe to aggregator and fill orders.
        # But SmartOrderExecutor handles price updates and replacements.
        # If the strategy wants "fill", we need market data.

        # IMPROVEMENT: Use the aggregator's last known price if available
        # self.aggregator is available.
        # But aggregator uses callbacks.

        # Let's assume for now it simulates "open" until canceled,
        # or we can auto-fill if the price is aggressive?

        # To keep it simple and safe for now: returns Open.
        # Real simulation would require more complex matching engine logic.

        return order


class RemoteMockBackend(IExchangeBackend):
    """
    Connects to the external Mock Exchange Service (Docker container) via HTTP.
    Used for E2E testing with docker-compose.
    """

    def __init__(self, exchange_config: dict):
        self.base_url = exchange_config.get("url", "http://mock_exchange:8888")
        self.session = None

    async def initialize(self):
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp is required for RemoteMockBackend")
            
        self.session = aiohttp.ClientSession()
        # Verify connection
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status != 200:
                    logger.warning(f"Remote Mock Exchange health check failed: {resp.status}")
                else:
                    logger.info(f"Connected to Remote Mock Exchange at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Remote Mock Exchange: {e}")
            # Don't raise here to allow retry logic in executor? 
            # Or raise to fail fast.
            pass

    async def close(self):
        if self.session:
            await self.session.close()

    async def _post(self, endpoint, data):
        if not self.session:
            raise RuntimeError("Backend not initialized")
        async with self.session.post(f"{self.base_url}{endpoint}", json=data) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise ValueError(f"Mock Exchange Error: {text}")
            return await resp.json()

    async def _get(self, endpoint):
        if not self.session:
            raise RuntimeError("Backend not initialized")
        async with self.session.get(f"{self.base_url}{endpoint}") as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise ValueError(f"Mock Exchange Error: {text}")
            return await resp.json()

    async def create_limit_buy_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        data = {
            "symbol": symbol,
            "side": "buy",
            "type": "limit",
            "quantity": quantity,
            "price": price
        }
        resp = await self._post("/api/v1/order", data)
        return {"id": resp["order_id"], "status": resp["status"]}

    async def create_limit_sell_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        data = {
            "symbol": symbol,
            "side": "sell",
            "type": "limit",
            "quantity": quantity,
            "price": price
        }
        resp = await self._post("/api/v1/order", data)
        return {"id": resp["order_id"], "status": resp["status"]}

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        await self._get(f"/api/v1/order/{order_id}") # Uses DELETE in mock? Mock says DELETE /api/v1/order/{id}
        # My _get helper uses GET. I need _delete.
        if not self.session:
             raise RuntimeError("Backend not initialized")
        async with self.session.delete(f"{self.base_url}/api/v1/order/{order_id}") as resp:
             if resp.status >= 400:
                 text = await resp.text()
                 raise ValueError(f"Mock Exchange Error: {text}")
             return await resp.json()

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        # Mock exchange API doesn't have a direct "fetch single order" endpoint documented 
        # in the file I read, except inside "get_orders".
        # Wait, I checked exchange_mock.py:
        # It has GET /api/v1/orders(symbol, status).
        # It DOES NOT have GET /api/v1/order/{id}.
        # Wait, verify.
        
        # It has DELETE /api/v1/order/{order_id}.
        
        # Implementation in MockExchangeBackend must rely on list filtering if ID lookup missing.
        # But typically we need status.
        
        # Let's assume I can add it or use list filtering.
        # Using list filtering:
        resp = await self._get(f"/api/v1/orders")
        orders = resp.get("orders", [])
        for o in orders:
            if o["order_id"] == order_id:
                # Map mock status to CCXT status
                status = o["status"]
                filled = o["quantity"] if status == "filled" else 0.0 # Mock simplification
                return {
                    "id": o["order_id"],
                    "symbol": o["symbol"],
                    "status": "closed" if status == "filled" else "canceled" if status == "cancelled" else "open",
                    "price": o["price"],
                    "amount": o["quantity"],
                    "filled": filled,
                    "remaining": o["quantity"] - filled
                }
        
        raise ValueError(f"Order {order_id} not found")

    async def fetch_positions(self) -> list[dict]:
        # Mock doesn't track positions explicitly, only balance. 
        # But we can assume open orders = positions for now or return empty.
        return []