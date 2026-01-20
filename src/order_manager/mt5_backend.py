"""
MT5 Backend Implementation
==========================

Integration with MetaTrader 5 terminal using the MetaTrader5 Python package.
Supports:
- Connection management
- Order execution (Limit/Market)
- Position monitoring
- Balance tracking (Equity-based)

Note: MetaTrader5 library is blocking, so operations are offloaded to threads.
"""

import asyncio
import logging
from datetime import datetime

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

from src.order_manager.exchange_backend import IExchangeBackend

logger = logging.getLogger(__name__)


class MT5Backend(IExchangeBackend):
    """
    Exchange backend for MetaTrader 5.
    Wraps synchronous MT5 API calls in asyncio executors.
    """

    def __init__(self, exchange_config: dict):
        self.config = exchange_config
        self.login = int(exchange_config.get("uid", 0)) or int(exchange_config.get("key", 0))
        self.password = exchange_config.get("password", "") or exchange_config.get("secret", "")
        self.server = exchange_config.get("server", "")
        self.path = exchange_config.get("path", "")  # Path to terminal.exe if needed
        self.name = "mt5"
        self._check_dependency()

    def _check_dependency(self):
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 package is not installed.")

    async def initialize(self):
        """Initialize connection to MT5 terminal."""
        logger.info(f"Initializing MT5 Backend for account {self.login} on {self.server}...")
        
        def _init_sync():
            # Initialize with path if provided, else default
            if self.path:
                if not mt5.initialize(path=self.path):
                    return False, mt5.last_error()
            else:
                if not mt5.initialize():
                    return False, mt5.last_error()
            
            # Login
            if self.login and self.password and self.server:
                authorized = mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server
                )
                if not authorized:
                    return False, mt5.last_error()
            
            return True, None

        success, error = await asyncio.to_thread(_init_sync)
        if not success:
            logger.error(f"MT5 initialization failed: {error}")
            raise ConnectionError(f"MT5 init failed: {error}")
        
        logger.info("MT5 Backend initialized successfully.")

    async def close(self):
        """Shutdown MT5 connection."""
        await asyncio.to_thread(mt5.shutdown)
        logger.info("MT5 Backend closed.")

    async def _execute_order(self, symbol: str, action_type, quantity: float, price: float = 0.0, sl: float = 0.0, tp: float = 0.0) -> dict:
        """Internal helper to execute order."""
        
        def _send():
            # Check symbol
            info = mt5.symbol_info(symbol)
            if info is None:
                return {"error": f"Symbol {symbol} not found"}
            
            if not info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {"error": f"Symbol {symbol} not visible"}

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(quantity),
                "type": action_type,
                "price": float(price) if price > 0 else (mt5.symbol_info_tick(symbol).ask if action_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid),
                "sl": float(sl),
                "tp": float(tp),
                "deviation": 20,
                "magic": 123456,
                "comment": "StoicCitadel",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Adjust for pending orders if price is provided and it's a limit order logic
            # However, the interface asks for 'create_limit_buy_order'.
            # If price is set, we should use ORDER_TYPE_BUY_LIMIT or SELL_LIMIT and ACTION_PENDING
            
            is_limit = price > 0
            if is_limit:
                request["action"] = mt5.TRADE_ACTION_PENDING
                if action_type == mt5.ORDER_TYPE_BUY:
                    request["type"] = mt5.ORDER_TYPE_BUY_LIMIT
                elif action_type == mt5.ORDER_TYPE_SELL:
                    request["type"] = mt5.ORDER_TYPE_SELL_LIMIT
                request["price"] = price

            result = mt5.order_send(request)
            if result is None:
                return {"error": "order_send failed, result is None"}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"error": f"Order failed: {result.comment} ({result.retcode})"}

            return {
                "id": str(result.order),
                "symbol": symbol,
                "status": "open" if is_limit else "closed", # Market orders fill immediately or fail
                "price": result.price,
                "amount": result.volume,
                "retcode": result.retcode
            }

        return await asyncio.to_thread(_send)

    async def create_limit_buy_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        params = params or {}
        return await self._execute_order(symbol, mt5.ORDER_TYPE_BUY, quantity, price, params.get("stop_loss", 0), params.get("take_profit", 0))

    async def create_limit_sell_order(
        self, symbol: str, quantity: float, price: float, params: dict | None = None
    ) -> dict:
        params = params or {}
        return await self._execute_order(symbol, mt5.ORDER_TYPE_SELL, quantity, price, params.get("stop_loss", 0), params.get("take_profit", 0))

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        
        def _cancel():
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
                "magic": 123456,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {"error": f"Cancel failed: {result.comment}"}
            return {"id": order_id, "status": "canceled"}

        return await asyncio.to_thread(_cancel)

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        # Check open orders first, then history
        def _fetch():
            orders = mt5.orders_get(ticket=int(order_id))
            if orders:
                o = orders[0]
                return {
                    "id": str(o.ticket),
                    "symbol": o.symbol,
                    "status": "open",
                    "price": o.price_open,
                    "amount": o.volume_initial,
                    "filled": o.volume_initial - o.volume_current,
                    "remaining": o.volume_current
                }
            
            # Check history
            hist = mt5.history_orders_get(ticket=int(order_id))
            if hist:
                h = hist[0]
                status_map = {
                    mt5.ORDER_STATE_FILLED: "closed",
                    mt5.ORDER_STATE_CANCELED: "canceled",
                    mt5.ORDER_STATE_EXPIRED: "expired"
                }
                return {
                    "id": str(h.ticket),
                    "symbol": h.symbol,
                    "status": status_map.get(h.state, "unknown"),
                    "price": h.price_open,
                    "amount": h.volume_initial,
                    "filled": h.volume_initial - h.volume_current, # usually initial for filled
                }
            
            return {"error": "Order not found"}

        return await asyncio.to_thread(_fetch)

    async def fetch_positions(self) -> list[dict]:
        
        def _get_positions():
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            res = []
            for p in positions:
                res.append({
                    "symbol": p.symbol,
                    "side": "long" if p.type == mt5.ORDER_TYPE_BUY else "short",
                    "amount": p.volume,
                    "entry_price": p.price_open,
                    "unrealized_pnl": p.profit, # MT5 provides profit directly
                    "id": str(p.ticket)
                })
            return res

        return await asyncio.to_thread(_get_positions)

    async def fetch_balance(self) -> dict:
        """
        Fetch account balance and equity.
        Returns dict compatible with CCXT structure usually, but we need Equity.
        """
        def _get_account():
            info = mt5.account_info()
            if info is None:
                return {}
            
            return {
                "free": info.margin_free,
                "used": info.margin,
                "total": info.equity, # CCXT total usually refers to equity or balance depending on mode
                "equity": info.equity,
                "balance": info.balance,
                "margin_level": info.margin_level
            }
        
        return await asyncio.to_thread(_get_account)