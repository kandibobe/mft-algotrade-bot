#!/usr/bin/env python3
"""
Freqtrade API Connector
========================

Connects to Freqtrade's REST API for monitoring and control.

Features:
- Status monitoring
- Trade management
- Strategy switching
- Performance metrics

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class FreqtradeConfig:
    """Freqtrade API configuration."""
    host: str = "localhost"
    port: int = 8080
    username: str = "freqtrade"
    password: str = "freqtrade"
    use_ssl: bool = False
    
    @property
    def base_url(self) -> str:
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"


class FreqtradeConnector:
    """
    Connector for Freqtrade REST API.
    
    Usage:
        config = FreqtradeConfig(
            host="localhost",
            port=8080,
            username="admin",
            password="secret"
        )
        
        async with FreqtradeConnector(config) as ft:
            # Get status
            status = await ft.get_status()
            
            # Get open trades
            trades = await ft.get_trades()
            
            # Force sell
            await ft.force_exit("BTC/USDT")
    """
    
    def __init__(self, config: FreqtradeConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Connect and authenticate."""
        self._session = aiohttp.ClientSession()
        await self._authenticate()
        logger.info(f"Connected to Freqtrade at {self.config.base_url}")
    
    async def disconnect(self):
        """Close connection."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _authenticate(self):
        """Authenticate with Freqtrade API."""
        url = f"{self.config.base_url}/api/v1/token/login"
        auth = aiohttp.BasicAuth(self.config.username, self.config.password)
        
        async with self._session.post(url, auth=auth) as resp:
            if resp.status == 200:
                data = await resp.json()
                self._token = data.get("access_token")
            else:
                raise Exception(f"Authentication failed: {resp.status}")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Any:
        """Make authenticated request."""
        url = f"{self.config.base_url}/api/v1/{endpoint}"
        headers = {"Authorization": f"Bearer {self._token}"}
        
        try:
            if method == "GET":
                async with self._session.get(url, headers=headers) as resp:
                    return await resp.json()
            elif method == "POST":
                async with self._session.post(url, headers=headers, json=data) as resp:
                    return await resp.json()
            elif method == "DELETE":
                async with self._session.delete(url, headers=headers) as resp:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Freqtrade API error: {e}")
            raise
    
    # =========================================================================
    # Status & Info
    # =========================================================================
    
    async def ping(self) -> bool:
        """Check if Freqtrade is responsive."""
        try:
            result = await self._request("GET", "ping")
            return result.get("status") == "pong"
        except:
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return await self._request("GET", "status")
    
    async def get_show_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return await self._request("GET", "show_config")
    
    async def get_profit(self) -> Dict[str, Any]:
        """Get profit summary."""
        return await self._request("GET", "profit")
    
    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        return await self._request("GET", "balance")
    
    async def get_count(self) -> Dict[str, Any]:
        """Get trade count."""
        return await self._request("GET", "count")
    
    async def get_performance(self) -> List[Dict[str, Any]]:
        """Get performance per pair."""
        return await self._request("GET", "performance")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        return await self._request("GET", "stats")
    
    async def get_daily(self, days: int = 7) -> Dict[str, Any]:
        """Get daily profit."""
        return await self._request("GET", f"daily?timescale={days}")
    
    async def get_whitelist(self) -> Dict[str, Any]:
        """Get current whitelist."""
        return await self._request("GET", "whitelist")
    
    async def get_blacklist(self) -> Dict[str, Any]:
        """Get current blacklist."""
        return await self._request("GET", "blacklist")
    
    async def get_locks(self) -> Dict[str, Any]:
        """Get pair locks."""
        return await self._request("GET", "locks")
    
    # =========================================================================
    # Trade Management
    # =========================================================================
    
    async def get_trades(
        self,
        limit: int = 500,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get trades history."""
        return await self._request("GET", f"trades?limit={limit}&offset={offset}")
    
    async def get_trade(self, trade_id: int) -> Dict[str, Any]:
        """Get specific trade."""
        return await self._request("GET", f"trade/{trade_id}")
    
    async def force_entry(
        self,
        pair: str,
        side: str = "long",
        price: Optional[float] = None,
        stake_amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """Force entry into a trade."""
        data = {
            "pair": pair,
            "side": side
        }
        if price:
            data["price"] = price
        if stake_amount:
            data["stake_amount"] = stake_amount
        
        return await self._request("POST", "forceenter", data)
    
    async def force_exit(
        self,
        trade_id: int,
        ordertype: str = "market",
        amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """Force exit from a trade."""
        data = {
            "tradeid": trade_id,
            "ordertype": ordertype
        }
        if amount:
            data["amount"] = amount
        
        return await self._request("POST", "forceexit", data)
    
    async def delete_trade(self, trade_id: int) -> Dict[str, Any]:
        """Delete/cancel a trade."""
        return await self._request("DELETE", f"trades/{trade_id}")
    
    # =========================================================================
    # Bot Control
    # =========================================================================
    
    async def start(self) -> Dict[str, Any]:
        """Start the bot."""
        return await self._request("POST", "start")
    
    async def stop(self) -> Dict[str, Any]:
        """Stop the bot."""
        return await self._request("POST", "stop")
    
    async def stopbuy(self) -> Dict[str, Any]:
        """Stop buying (let existing trades finish)."""
        return await self._request("POST", "stopbuy")
    
    async def reload_config(self) -> Dict[str, Any]:
        """Reload configuration."""
        return await self._request("POST", "reload_config")
    
    async def add_blacklist(self, pairs: List[str]) -> Dict[str, Any]:
        """Add pairs to blacklist."""
        return await self._request("POST", "blacklist", {"blacklist": pairs})
    
    async def delete_blacklist(self, pairs: List[str]) -> Dict[str, Any]:
        """Remove pairs from blacklist."""
        return await self._request("DELETE", "blacklist", {"pairs_to_delete": pairs})
    
    # =========================================================================
    # Strategy
    # =========================================================================
    
    async def get_strategies(self) -> Dict[str, Any]:
        """Get available strategies."""
        return await self._request("GET", "strategies")
    
    async def get_strategy(self, strategy: str) -> Dict[str, Any]:
        """Get strategy details."""
        return await self._request("GET", f"strategy/{strategy}")
    
    # =========================================================================
    # Logs
    # =========================================================================
    
    async def get_logs(self, limit: int = 50) -> Dict[str, Any]:
        """Get recent logs."""
        return await self._request("GET", f"logs?limit={limit}")
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            ping_ok = await self.ping()
            status = await self.get_status() if ping_ok else {}
            profit = await self.get_profit() if ping_ok else {}
            
            return {
                "service": "freqtrade",
                "status": "healthy" if ping_ok else "unhealthy",
                "connected": ping_ok,
                "trading_status": status,
                "profit_summary": profit
            }
        except Exception as e:
            return {
                "service": "freqtrade",
                "status": "error",
                "error": str(e)
            }
