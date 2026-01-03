"""
Smart Order Executor
====================

Manages and executes Smart Orders using asynchronous event loop.
Provides the bridge between high-level smart order logic and low-level exchange calls.
"""

import asyncio
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.order_manager.order_types import Order, OrderStatus, OrderType
from src.order_manager.smart_order import SmartOrder, ChaseLimitOrder, TWAPOrder, VWAPOrder
from src.websocket.aggregator import DataAggregator, AggregatedTicker
from src.order_manager.exchange_backend import IExchangeBackend, CCXTBackend, MockExchangeBackend
from src.utils.logger import log  # Use structured logger
from src.analysis.attribution import AttributionService
from src.notification.telegram import TelegramBot

logger = logging.getLogger(__name__)

class SmartOrderExecutor:
    """
    Asynchronous executor for Smart Orders.
    
    Features:
    - Non-blocking order management
    - Real-time price adjustments based on ticker updates
    - Automatic retry and error handling
    - Safe Execution Abstraction (Live/Dry-Run)
    """
    
    def __init__(self, aggregator: Optional[DataAggregator] = None, exchange_config: Optional[Dict] = None, dry_run: bool = True):
        self.aggregator = aggregator
        self._active_orders: Dict[str, SmartOrder] = {}
        self._order_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._lock = asyncio.Lock()
        
        self._exchange_config = exchange_config or {}
        self._dry_run = dry_run
        
        # Initialize Notification Bot
        self.telegram = TelegramBot()
        
        # Initialize Backend
        if self._dry_run:
            log.warning("STARTING IN DRY-RUN MODE (Mock Execution)")
            self.backend: IExchangeBackend = MockExchangeBackend(aggregator)
        else:
            log.warning("🚨 STARTING IN LIVE MODE (Real Execution)")
            self.backend = CCXTBackend(self._exchange_config)

    async def start(self):
        """Start the executor service."""
        self._running = True
        
        # Safety Guard for Live Trading
        if not self._dry_run:
            confirm = os.getenv("CONFIRM_LIVE_TRADING", "false").lower()
            if confirm != "true":
                msg = "LIVE TRADING BLOCKED: Environment variable CONFIRM_LIVE_TRADING=true is missing!"
                log.error(msg)
                raise RuntimeError(msg)
            
            log.warning("⚠️  LIVE TRADING ENABLED - REAL FUNDS AT RISK ⚠️")

        try:
            await self.backend.initialize()
            log.info("Smart Order Executor backend initialized")
        except Exception as e:
            log.error("Failed to initialize backend", error=str(e))
            # In live mode, this is critical. In dry run, maybe we survive?
            if not self._dry_run:
                raise e

        logger.info("Smart Order Executor started")
        
        # If aggregator is provided, subscribe to ticker updates
        if self.aggregator:
            @self.aggregator.on_aggregated_ticker
            async def handle_ticker(ticker: AggregatedTicker):
                await self._process_ticker_update(ticker)

    async def stop(self):
        """Stop the executor and cancel all active order tasks."""
        self._running = False
        async with self._lock:
            for task in self._order_tasks.values():
                task.cancel()
            self._order_tasks.clear()
            self._active_orders.clear()
            
        if self.backend:
            await self.backend.close()
            
        logger.info("Smart Order Executor stopped")

    async def submit_order(self, order: SmartOrder) -> str:
        """
        Submit a new smart order for execution.
        """
        async with self._lock:
            order.update_status(OrderStatus.SUBMITTED)
            self._active_orders[order.order_id] = order
            
            # Start a background task to monitor/manage this specific order
            task = asyncio.create_task(self._manage_order(order))
            self._order_tasks[order.order_id] = task
            
            from src.utils.logger import log_order
            log_order(
                order_id=order.order_id,
                symbol=order.symbol,
                order_type=order.order_type.value,
                side="buy" if order.is_buy else "sell",
                quantity=order.quantity,
                price=order.price,
                status="submitted",
                is_dry_run=self._dry_run
            )
            
            mode_str = "[DRY RUN] " if self._dry_run else ""
            self.telegram.send_message(
                f"<b>{mode_str}Order Submitted</b>\n"
                f"Symbol: {order.symbol}\n"
                f"Side: {'BUY' if order.is_buy else 'SELL'}\n"
                f"Quantity: {order.quantity}\n"
                f"Price: {order.price}"
            )
            
            return order.order_id

    async def cancel_order(self, order_id: str):
        """Cancel an active smart order."""
        async with self._lock:
            if order_id in self._active_orders:
                order = self._active_orders[order_id]
                order.update_status(OrderStatus.CANCELLED)
                
                if self.backend and order.exchange_order_id:
                    try:
                        await self.backend.cancel_order(order.exchange_order_id, order.symbol)
                    except Exception as e:
                        logger.warning(f"Failed to cancel exchange order: {e}")
                
                if order_id in self._order_tasks:
                    self._order_tasks[order_id].cancel()
                    del self._order_tasks[order_id]
                
                del self._active_orders[order_id]
                
                from src.utils.logger import log_order
                log_order(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    order_type=order.order_type.value,
                    side="buy" if order.is_buy else "sell",
                    quantity=order.quantity,
                    status="cancelled"
                )

    async def _manage_order(self, order: SmartOrder):
        """
        Internal loop to manage an individual order's lifecycle.
        """
        try:
            if isinstance(order, TWAPOrder):
                await self._execute_twap_order(order)
            elif isinstance(order, VWAPOrder):
                await self._execute_vwap_order(order)
            else:
                await self._execute_standard_order(order)
        except asyncio.CancelledError:
            logger.debug(f"Order management task for {order.order_id} cancelled")
        except Exception as e:
            logger.error(f"Error managing order {order.order_id}: {e}")
            order.update_status(OrderStatus.FAILED, error=str(e))
            
            mode_str = "[DRY RUN] " if self._dry_run else ""
            self.telegram.send_message(
                f"<b>❌ {mode_str}Order Failed</b>\n"
                f"Order ID: {order.order_id}\n"
                f"Symbol: {order.symbol}\n"
                f"Error: {str(e)}"
            )
        finally:
            async with self._lock:
                if order.order_id in self._active_orders:
                    del self._active_orders[order.order_id]
                if order.order_id in self._order_tasks:
                    del self._order_tasks[order.order_id]
                    
    async def _execute_standard_order(self, order: SmartOrder):
        """Logic for handling standard (e.g., ChaseLimit) orders."""
        if self.backend and not order.exchange_order_id:
            await self._place_initial_order(order)

        while order.is_active and self._running:
            if order.check_timeout():
                break
            
            if self.backend and order.exchange_order_id:
                try:
                    exch_order = await self.backend.fetch_order(order.exchange_order_id, order.symbol)
                    if exch_order['status'] == 'closed':
                        order.update_fill(exch_order['filled'] - order.filled_quantity, exch_order['price'])
                        order.update_status(OrderStatus.FILLED)
                        break
                    elif exch_order['status'] == 'canceled':
                        order.update_status(OrderStatus.CANCELLED)
                        break
                except Exception as e:
                    logger.warning(f"Error fetching order status: {e}")
            
            await asyncio.sleep(1.0)

    async def _execute_twap_order(self, order: TWAPOrder):
        """Logic for handling TWAP orders."""
        chunk_quantity = order.quantity / order.num_chunks
        interval_seconds = (order.duration_minutes * 60) / order.num_chunks

        for i in range(order.num_chunks):
            if not self._running or order.is_terminal:
                break
            
            logger.info(f"Executing TWAP chunk {i+1}/{order.num_chunks} for order {order.order_id}")
            
            chunk_order = ChaseLimitOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=chunk_quantity,
                price=order.price, 
                attribution_metadata=order.attribution_metadata
            )
            
            await self.submit_order(chunk_order)
            
            await asyncio.sleep(interval_seconds)
        
        order.update_status(OrderStatus.FILLED)

    async def _execute_vwap_order(self, order: VWAPOrder):
        """Logic for handling VWAP orders."""
        if not order.volume_profile or len(order.volume_profile) != order.num_chunks:
            raise ValueError("VWAP order requires a valid volume profile.")

        interval_seconds = (order.duration_minutes * 60) / order.num_chunks

        for i, volume_pct in enumerate(order.volume_profile):
            if not self._running or order.is_terminal:
                break

            chunk_quantity = order.quantity * volume_pct
            logger.info(f"Executing VWAP chunk {i+1}/{order.num_chunks} for order {order.order_id} ({chunk_quantity:.4f})")
            
            chunk_order = ChaseLimitOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=chunk_quantity,
                price=order.price,
                attribution_metadata=order.attribution_metadata
            )
            
            await self.submit_order(chunk_order)
            
            await asyncio.sleep(interval_seconds)
            
        order.update_status(OrderStatus.FILLED)

    async def _place_initial_order(self, order: SmartOrder):
        """Place the initial order on the exchange."""
        if not self.backend:
            return
            
        try:
            params = {}
            if order.is_buy:
                res = await self.backend.create_limit_buy_order(order.symbol, order.quantity, order.price, params)
            else:
                res = await self.backend.create_limit_sell_order(order.symbol, order.quantity, order.price, params)
            
            order.exchange_order_id = res['id']
            order.update_status(OrderStatus.OPEN)
            logger.info(f"Placed initial order {order.order_id} as {res['id']}")
            
        except Exception as e:
            logger.error(f"Failed to place initial order: {e}")
            order.update_status(OrderStatus.FAILED, str(e))
            raise e

    async def _process_ticker_update(self, ticker: AggregatedTicker):
        """
        Handle ticker updates and propagate them to relevant smart orders.
        """
        async with self._lock:
            orders_to_update = [
                order for order in self._active_orders.values() 
                if order.symbol == ticker.symbol and order.is_active
            ]
            
        for order in orders_to_update:
            try:
                ticker_dict = {
                    'best_bid': ticker.best_bid,
                    'best_ask': ticker.best_ask,
                    'spread_pct': ticker.spread_pct
                }
                
                old_price = getattr(order, 'price', None)
                order.on_ticker_update(ticker_dict)
                new_price = getattr(order, 'price', None)
                
                if old_price != new_price:
                    await self._replace_exchange_order(order)
                    
            except Exception as e:
                logger.error(f"Error processing ticker update for order {order.order_id}: {e}")

    async def _replace_exchange_order(self, order: SmartOrder):
        """
        Execute order modification on exchange (Cancel + Replace).
        """
        if not self.backend:
            logger.warning("No backend configured, cannot replace order.")
            return

        logger.info(f"Adjusting order {order.order_id} price to {order.price}")
        
        try:
            if order.exchange_order_id:
                try:
                    await self.backend.cancel_order(order.exchange_order_id, order.symbol)
                except Exception as e:
                    logger.warning(f"Cancel failed (order might be filled?): {e}")
                    return 

            params = {}
            
            if order.is_buy:
                new_order = await self.backend.create_limit_buy_order(
                    order.symbol, order.quantity, order.price, params
                )
            else:
                new_order = await self.backend.create_limit_sell_order(
                    order.symbol, order.quantity, order.price, params
                )
                
            order.exchange_order_id = new_order['id']
            
            logger.info(f"Replaced order. New ID: {new_order['id']}")
            
        except Exception as e:
            logger.error(f"Failed to replace order {order.order_id}: {e}")

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get the current status of an order."""
        if order_id in self._active_orders:
            return self._active_orders[order_id].status
        return None
