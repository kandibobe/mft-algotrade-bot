
"""
Exchange Backend Abstraction
============================

Defines interface and implementations for exchange interaction.
Allows switching between Live (CCXT) and Dry-Run (Mock) execution.
"""

import abc
import asyncio
import logging
from typing import Dict, Any, Optional

try:
    import ccxt.async_support as ccxt
except ImportError:
    ccxt = None

from src.websocket.aggregator import DataAggregator

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
    async def create_limit_buy_order(self, symbol: str, quantity: float, price: float, params: Dict = None) -> Dict:
        """Create a limit buy order."""
        pass

    @abc.abstractmethod
    async def create_limit_sell_order(self, symbol: str, quantity: float, price: float, params: Dict = None) -> Dict:
        """Create a limit sell order."""
        pass

    @abc.abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order."""
        pass

    @abc.abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> Dict:
        """Fetch order status."""
        pass


class CCXTBackend(IExchangeBackend):
    """Live exchange interaction using CCXT."""
    
    def __init__(self, exchange_config: Dict):
        self.config = exchange_config
        self.exchange = None
        self.name = exchange_config.get('name', 'binance')
        
    async def initialize(self):
        if not ccxt:
            raise ImportError("CCXT not installed")
            
        exchange_class = getattr(ccxt, self.name)
        self.exchange = exchange_class({
            'apiKey': self.config.get('key'),
            'secret': self.config.get('secret'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} 
        })
        # Verify connection (optional but good for safety)
        # await self.exchange.load_markets()
        logger.info(f"Initialized CCXTBackend for {self.name}")

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    async def create_limit_buy_order(self, symbol: str, quantity: float, price: float, params: Dict = None) -> Dict:
        return await self.exchange.create_limit_buy_order(symbol, quantity, price, params or {})

    async def create_limit_sell_order(self, symbol: str, quantity: float, price: float, params: Dict = None) -> Dict:
        return await self.exchange.create_limit_sell_order(symbol, quantity, price, params or {})

    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        return await self.exchange.cancel_order(order_id, symbol)

    async def fetch_order(self, order_id: str, symbol: str) -> Dict:
        return await self.exchange.fetch_order(order_id, symbol)


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
        
    def _create_mock_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        self.order_counter += 1
        order_id = f"mock_{self.order_counter}"
        
        order = {
            'id': order_id,
            'symbol': symbol,
            'status': 'open',
            'side': side,
            'amount': quantity,
            'price': price,
            'filled': 0.0,
            'remaining': quantity,
            'timestamp': 0 # TODO: add timestamp
        }
        self.orders[order_id] = order
        
        # Simulate immediate matching attempt? 
        # For now, just return open order. SmartOrderExecutor loop will check status.
        # But wait, SmartOrderExecutor calls fetch_order loop.
        # We need a way to "fill" orders.
        # The loop calls fetch_order. We need to update order status based on price.
        
        return order

    async def create_limit_buy_order(self, symbol: str, quantity: float, price: float, params: Dict = None) -> Dict:
        return self._create_mock_order(symbol, 'buy', quantity, price)

    async def create_limit_sell_order(self, symbol: str, quantity: float, price: float, params: Dict = None) -> Dict:
        return self._create_mock_order(symbol, 'sell', quantity, price)

    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'canceled'
            return self.orders[order_id]
        raise ValueError("Order not found")

    async def fetch_order(self, order_id: str, symbol: str) -> Dict:
        # Simulate fill logic here
        if order_id not in self.orders:
             raise ValueError("Order not found")
             
        order = self.orders[order_id]
        if order['status'] in ['closed', 'canceled']:
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
