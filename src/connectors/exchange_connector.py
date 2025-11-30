#!/usr/bin/env python3
"""
Unified Exchange Connector
===========================

Provides unified interface for multiple cryptocurrency exchanges.
Supports both REST API and WebSocket connections.

Supported exchanges:
- Binance (Spot, Futures)
- Bybit (Spot, Derivatives)
- OKX
- Kraken

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    BINANCE_SPOT = "binance_spot"
    BINANCE_FUTURES = "binance_futures"
    BYBIT_SPOT = "bybit_spot"
    BYBIT_DERIVATIVES = "bybit_derivatives"
    OKX = "okx"
    KRAKEN = "kraken"


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    exchange_type: ExchangeType
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""  # For OKX
    testnet: bool = False
    rate_limit: int = 10  # requests per second
    timeout: int = 30
    
    @property
    def base_url(self) -> str:
        """Get REST API base URL."""
        urls = {
            ExchangeType.BINANCE_SPOT: (
                "https://testnet.binance.vision" if self.testnet 
                else "https://api.binance.com"
            ),
            ExchangeType.BINANCE_FUTURES: (
                "https://testnet.binancefuture.com" if self.testnet
                else "https://fapi.binance.com"
            ),
            ExchangeType.BYBIT_SPOT: (
                "https://api-testnet.bybit.com" if self.testnet
                else "https://api.bybit.com"
            ),
            ExchangeType.BYBIT_DERIVATIVES: (
                "https://api-testnet.bybit.com" if self.testnet
                else "https://api.bybit.com"
            ),
            ExchangeType.OKX: (
                "https://www.okx.com"  # OKX uses header for testnet
            ),
            ExchangeType.KRAKEN: "https://api.kraken.com",
        }
        return urls.get(self.exchange_type, "")
    
    @property
    def ws_url(self) -> str:
        """Get WebSocket URL."""
        urls = {
            ExchangeType.BINANCE_SPOT: (
                "wss://testnet.binance.vision/ws" if self.testnet
                else "wss://stream.binance.com:9443/ws"
            ),
            ExchangeType.BINANCE_FUTURES: (
                "wss://stream.binancefuture.com/ws" if self.testnet
                else "wss://fstream.binance.com/ws"
            ),
            ExchangeType.BYBIT_SPOT: (
                "wss://stream-testnet.bybit.com/v5/public/spot" if self.testnet
                else "wss://stream.bybit.com/v5/public/spot"
            ),
            ExchangeType.BYBIT_DERIVATIVES: (
                "wss://stream-testnet.bybit.com/v5/public/linear" if self.testnet
                else "wss://stream.bybit.com/v5/public/linear"
            ),
            ExchangeType.OKX: "wss://ws.okx.com:8443/ws/v5/public",
            ExchangeType.KRAKEN: "wss://ws.kraken.com",
        }
        return urls.get(self.exchange_type, "")


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    status: str
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""
    timestamp: float = field(default_factory=time.time)
    raw_response: Dict = field(default_factory=dict)


@dataclass
class Balance:
    """Account balance."""
    asset: str
    free: float
    locked: float
    
    @property
    def total(self) -> float:
        return self.free + self.locked


class ExchangeConnector:
    """
    Unified exchange connector with REST and WebSocket support.
    
    Usage:
        config = ExchangeConfig(
            exchange_type=ExchangeType.BINANCE_SPOT,
            api_key="your_key",
            api_secret="your_secret"
        )
        
        async with ExchangeConnector(config) as exchange:
            # Get ticker
            ticker = await exchange.get_ticker("BTC/USDT")
            
            # Place order
            order = await exchange.create_order(
                symbol="BTC/USDT",
                side="buy",
                order_type="limit",
                quantity=0.001,
                price=95000
            )
            
            # Get balances
            balances = await exchange.get_balances()
    """
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = asyncio.Semaphore(config.rate_limit)
        self._last_request_time = 0.0
        self._request_interval = 1.0 / config.rate_limit
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"Connected to {self.config.exchange_type.value}")
    
    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info(f"Disconnected from {self.config.exchange_type.value}")
    
    # =========================================================================
    # Market Data
    # =========================================================================
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker for symbol."""
        symbol_formatted = self._format_symbol(symbol)
        
        if self.config.exchange_type in [ExchangeType.BINANCE_SPOT, ExchangeType.BINANCE_FUTURES]:
            endpoint = "/api/v3/ticker/24hr" if "SPOT" in self.config.exchange_type.value.upper() else "/fapi/v1/ticker/24hr"
            data = await self._request("GET", endpoint, {"symbol": symbol_formatted})
            return {
                "symbol": symbol,
                "bid": float(data.get("bidPrice", 0)),
                "ask": float(data.get("askPrice", 0)),
                "last": float(data.get("lastPrice", 0)),
                "volume": float(data.get("volume", 0)),
                "change_pct": float(data.get("priceChangePercent", 0)),
            }
        
        elif self.config.exchange_type in [ExchangeType.BYBIT_SPOT, ExchangeType.BYBIT_DERIVATIVES]:
            category = "spot" if "SPOT" in self.config.exchange_type.value.upper() else "linear"
            data = await self._request("GET", "/v5/market/tickers", {
                "category": category,
                "symbol": symbol_formatted
            })
            ticker = data.get("result", {}).get("list", [{}])[0]
            return {
                "symbol": symbol,
                "bid": float(ticker.get("bid1Price", 0)),
                "ask": float(ticker.get("ask1Price", 0)),
                "last": float(ticker.get("lastPrice", 0)),
                "volume": float(ticker.get("volume24h", 0)),
                "change_pct": float(ticker.get("price24hPcnt", 0)) * 100,
            }
        
        raise NotImplementedError(f"Ticker not implemented for {self.config.exchange_type}")
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book for symbol."""
        symbol_formatted = self._format_symbol(symbol)
        
        if self.config.exchange_type in [ExchangeType.BINANCE_SPOT, ExchangeType.BINANCE_FUTURES]:
            endpoint = "/api/v3/depth" if "SPOT" in self.config.exchange_type.value.upper() else "/fapi/v1/depth"
            data = await self._request("GET", endpoint, {
                "symbol": symbol_formatted,
                "limit": limit
            })
            return {
                "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
            }
        
        raise NotImplementedError(f"Orderbook not implemented for {self.config.exchange_type}")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical klines/candlesticks."""
        symbol_formatted = self._format_symbol(symbol)
        
        if self.config.exchange_type in [ExchangeType.BINANCE_SPOT, ExchangeType.BINANCE_FUTURES]:
            endpoint = "/api/v3/klines" if "SPOT" in self.config.exchange_type.value.upper() else "/fapi/v1/klines"
            data = await self._request("GET", endpoint, {
                "symbol": symbol_formatted,
                "interval": interval,
                "limit": limit
            })
            return [{
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            } for k in data]
        
        raise NotImplementedError(f"Klines not implemented for {self.config.exchange_type}")
    
    # =========================================================================
    # Trading
    # =========================================================================
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> OrderResult:
        """Create a new order."""
        symbol_formatted = self._format_symbol(symbol)
        
        if self.config.exchange_type in [ExchangeType.BINANCE_SPOT, ExchangeType.BINANCE_FUTURES]:
            params = {
                "symbol": symbol_formatted,
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": quantity,
            }
            
            if order_type.upper() == "LIMIT":
                params["price"] = price
                params["timeInForce"] = time_in_force
            
            if stop_price:
                params["stopPrice"] = stop_price
            
            if client_order_id:
                params["newClientOrderId"] = client_order_id
            
            if "FUTURES" in self.config.exchange_type.value.upper() and reduce_only:
                params["reduceOnly"] = "true"
            
            endpoint = "/api/v3/order" if "SPOT" in self.config.exchange_type.value.upper() else "/fapi/v1/order"
            data = await self._request("POST", endpoint, params, signed=True)
            
            return OrderResult(
                order_id=str(data.get("orderId", "")),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price or 0,
                status=data.get("status", "UNKNOWN"),
                filled_quantity=float(data.get("executedQty", 0)),
                average_price=float(data.get("avgPrice", 0) or data.get("price", 0)),
                raw_response=data
            )
        
        raise NotImplementedError(f"Order creation not implemented for {self.config.exchange_type}")
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel an existing order."""
        symbol_formatted = self._format_symbol(symbol)
        
        if self.config.exchange_type in [ExchangeType.BINANCE_SPOT, ExchangeType.BINANCE_FUTURES]:
            params = {"symbol": symbol_formatted}
            if order_id:
                params["orderId"] = order_id
            if client_order_id:
                params["origClientOrderId"] = client_order_id
            
            endpoint = "/api/v3/order" if "SPOT" in self.config.exchange_type.value.upper() else "/fapi/v1/order"
            return await self._request("DELETE", endpoint, params, signed=True)
        
        raise NotImplementedError(f"Order cancellation not implemented for {self.config.exchange_type}")
    
    async def get_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get order status."""
        symbol_formatted = self._format_symbol(symbol)
        
        if self.config.exchange_type in [ExchangeType.BINANCE_SPOT, ExchangeType.BINANCE_FUTURES]:
            params = {"symbol": symbol_formatted}
            if order_id:
                params["orderId"] = order_id
            if client_order_id:
                params["origClientOrderId"] = client_order_id
            
            endpoint = "/api/v3/order" if "SPOT" in self.config.exchange_type.value.upper() else "/fapi/v1/order"
            return await self._request("GET", endpoint, params, signed=True)
        
        raise NotImplementedError(f"Get order not implemented for {self.config.exchange_type}")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = self._format_symbol(symbol)
        
        if self.config.exchange_type in [ExchangeType.BINANCE_SPOT, ExchangeType.BINANCE_FUTURES]:
            endpoint = "/api/v3/openOrders" if "SPOT" in self.config.exchange_type.value.upper() else "/fapi/v1/openOrders"
            return await self._request("GET", endpoint, params, signed=True)
        
        raise NotImplementedError(f"Get open orders not implemented for {self.config.exchange_type}")
    
    # =========================================================================
    # Account
    # =========================================================================
    
    async def get_balances(self) -> List[Balance]:
        """Get account balances."""
        if self.config.exchange_type == ExchangeType.BINANCE_SPOT:
            data = await self._request("GET", "/api/v3/account", {}, signed=True)
            return [
                Balance(
                    asset=b["asset"],
                    free=float(b["free"]),
                    locked=float(b["locked"])
                )
                for b in data.get("balances", [])
                if float(b["free"]) > 0 or float(b["locked"]) > 0
            ]
        
        elif self.config.exchange_type == ExchangeType.BINANCE_FUTURES:
            data = await self._request("GET", "/fapi/v2/balance", {}, signed=True)
            return [
                Balance(
                    asset=b["asset"],
                    free=float(b["availableBalance"]),
                    locked=float(b["balance"]) - float(b["availableBalance"])
                )
                for b in data
                if float(b["balance"]) > 0
            ]
        
        raise NotImplementedError(f"Get balances not implemented for {self.config.exchange_type}")
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions (for futures)."""
        if self.config.exchange_type == ExchangeType.BINANCE_FUTURES:
            data = await self._request("GET", "/fapi/v2/positionRisk", {}, signed=True)
            return [
                {
                    "symbol": p["symbol"],
                    "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
                    "quantity": abs(float(p["positionAmt"])),
                    "entry_price": float(p["entryPrice"]),
                    "mark_price": float(p["markPrice"]),
                    "unrealized_pnl": float(p["unRealizedProfit"]),
                    "leverage": int(p["leverage"]),
                }
                for p in data
                if float(p["positionAmt"]) != 0
            ]
        
        return []  # Spot doesn't have positions
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for exchange."""
        # Convert BTC/USDT to BTCUSDT
        return symbol.replace("/", "").replace("-", "").upper()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any],
        signed: bool = False
    ) -> Any:
        """Make HTTP request to exchange."""
        await self._rate_limit()
        
        if self._session is None:
            await self.connect()
        
        url = f"{self.config.base_url}{endpoint}"
        headers = {}
        
        if signed:
            params = self._sign_request(params)
            headers["X-MBX-APIKEY"] = self.config.api_key
        
        try:
            if method == "GET":
                async with self._session.get(url, params=params, headers=headers) as resp:
                    return await self._handle_response(resp)
            elif method == "POST":
                async with self._session.post(url, data=params, headers=headers) as resp:
                    return await self._handle_response(resp)
            elif method == "DELETE":
                async with self._session.delete(url, params=params, headers=headers) as resp:
                    return await self._handle_response(resp)
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            raise
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Any:
        """Handle API response."""
        text = await response.text()
        
        if response.status != 200:
            logger.error(f"API error {response.status}: {text}")
            raise Exception(f"API error {response.status}: {text}")
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    
    def _sign_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sign request for authenticated endpoints."""
        if self.config.exchange_type in [ExchangeType.BINANCE_SPOT, ExchangeType.BINANCE_FUTURES]:
            params["timestamp"] = int(time.time() * 1000)
            query_string = urlencode(params)
            signature = hmac.new(
                self.config.api_secret.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
        
        return params
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        async with self._rate_limiter:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._request_interval:
                await asyncio.sleep(self._request_interval - elapsed)
            self._last_request_time = time.time()
