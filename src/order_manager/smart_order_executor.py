"""
Smart Order Executor
====================

Manages and executes Smart Orders using asynchronous event loop.
Provides the bridge between high-level smart order logic and low-level exchange calls.
"""

import asyncio
import logging
import os
import time
from datetime import datetime

from src.notification.telegram import TelegramBot
from src.order_manager.exchange_backend import CCXTBackend, IExchangeBackend, MockExchangeBackend
from src.order_manager.order_types import OrderStatus, IcebergOrder, PeggedOrder
from src.order_manager.smart_order import ChaseLimitOrder, SmartOrder, TWAPOrder, VWAPOrder
from src.risk.risk_manager import RiskManager
from src.utils.logger import log  # Use structured logger
from src.websocket.aggregator import AggregatedTicker, DataAggregator

logger = logging.getLogger(__name__)


class SmartOrderExecutor:
    """
    Asynchronous executor for Smart Orders.

    Features:
    - Non-blocking order management
    - Real-time price adjustments based on ticker updates
    - Automatic retry and error handling
    - Safe Execution Abstraction (Live/Dry-Run)
    - Integrated Risk Management Gate
    """

    def __init__(
        self,
        aggregator: DataAggregator | None = None,
        exchange_config: dict | None = None,
        dry_run: bool = True,
        shadow_mode: bool = False,
        risk_manager: RiskManager | None = None,
    ):
        self.aggregator = aggregator
        self._active_orders: dict[str, SmartOrder] = {}
        self._order_tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._lock = asyncio.Lock()

        self._exchange_config = exchange_config or {}
        self._dry_run = dry_run
        self._shadow_mode = shadow_mode

        # Initialize Risk Manager
        self.risk_manager = risk_manager or RiskManager()

        # Initialize Notification Bot
        self.telegram = TelegramBot()

        # Initialize Backend
        if self._dry_run or self._shadow_mode:
            log.warning(f"STARTING IN {'SHADOW' if self._shadow_mode else 'DRY-RUN'} MODE (Mock Execution)")
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

    async def submit_order(self, order: SmartOrder, exchange: str = None) -> str:
        """
        Submit a new smart order for execution.
        """
        exchange_name = exchange or self._exchange_config.get("name", "default")

        # 📊 Slippage Check (MFT Optimization)
        if self.aggregator and order.price:
            ticker = self.aggregator.get_aggregated_ticker(order.symbol)
            if ticker:
                # If we are buying, we care about distance from best ask
                # If we are selling, we care about distance from best bid
                market_price = ticker.best_ask if order.is_buy else ticker.best_bid
                if market_price > 0:
                    slippage = abs(order.price - market_price) / market_price * 100
                    max_slippage = 0.5  # 0.5% default, could be moved to config
                    if slippage > max_slippage:
                        reason = f"High slippage detected: {slippage:.2f}% (max {max_slippage}%)"
                        log.warning(f"Order {order.order_id} rejected: {reason}")
                        raise RuntimeError(f"Execution Gate Failed: {reason}")

        # 🛡️ Risk Gate: Check Circuit Breaker
        if not self.risk_manager.circuit_breaker.can_trade():
            reason = f"Circuit Breaker is {self.risk_manager.circuit_breaker.state.value}"
            logger.error(f"Order rejected: {reason}")
            raise RuntimeError(f"Risk Check Failed: {reason}")

        # 🛡️ Risk Gate: Evaluate Trade Risk (if prices available)
        if order.price and order.stop_price:
            res = self.risk_manager.evaluate_trade(
                symbol=order.symbol,
                entry_price=order.price,
                stop_loss_price=order.stop_price,
                side="long" if order.is_buy else "short",
                exchange=exchange_name,
            )
            if not res["allowed"]:
                reason = res["rejection_reason"]
                logger.error(f"Order rejected by Risk Manager: {reason}")
                raise RuntimeError(f"Risk Check Failed: {reason}")

        async with self._lock:
            order.update_status(OrderStatus.SUBMITTED)
            order.submission_timestamp = time.time()
            # Record signal timestamp if not set (Phase 3)
            if not order.signal_timestamp:
                order.signal_timestamp = time.time()
                
            self._active_orders[order.order_id] = order

            # Attach exchange info to order metadata
            if order.attribution_metadata is None:
                order.attribution_metadata = {}
            order.attribution_metadata["exchange"] = exchange_name

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
                is_dry_run=self._dry_run,
            )

            mode_str = ""
            if self._shadow_mode:
                mode_str = "[SHADOW] "
            elif self._dry_run:
                mode_str = "[DRY RUN] "

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
                    status="cancelled",
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
            elif isinstance(order, IcebergOrder):
                await self._execute_iceberg_order(order)
            elif isinstance(order, PeggedOrder):
                await self._execute_pegged_order(order)
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
                f"Error: {e!s}"
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
                    # In shadow mode, we can simulate realistic fill if aggregator is present
                    if self._shadow_mode and self.aggregator:
                        ticker = self.aggregator.get_aggregated_ticker(order.symbol)
                        if ticker:
                            # Simple matching: if buy price >= best ask, or sell price <= best bid
                            can_fill = False
                            if order.is_buy and order.price >= ticker.best_ask:
                                can_fill = True
                                fill_price = ticker.best_ask
                            elif not order.is_buy and order.price <= ticker.best_bid:
                                can_fill = True
                                fill_price = ticker.best_bid
                            
                            if can_fill:
                                order.fill_timestamp = time.time()
                                order.update_fill(order.quantity, fill_price)
                                order.update_status(OrderStatus.FILLED)
                                await self._log_shadow_trade(order)
                                break

                    exch_order = await self.backend.fetch_order(
                        order.exchange_order_id, order.symbol
                    )
                    if exch_order["status"] == "closed":
                        order.fill_timestamp = time.time()
                        order.update_fill(
                            exch_order["filled"] - order.filled_quantity, exch_order["price"]
                        )
                        order.update_status(OrderStatus.FILLED)
                        if self._shadow_mode:
                            await self._log_shadow_trade(order)
                        else:
                            # Log real execution metrics
                            await self._log_execution(order)
                        break
                    elif exch_order["status"] == "canceled":
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

            logger.info(
                f"Executing TWAP chunk {i + 1}/{order.num_chunks} for order {order.order_id}"
            )

            chunk_order = ChaseLimitOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=chunk_quantity,
                price=order.price,
                attribution_metadata=order.attribution_metadata,
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
            logger.info(
                f"Executing VWAP chunk {i + 1}/{order.num_chunks} for order {order.order_id} ({chunk_quantity:.4f})"
            )

            chunk_order = ChaseLimitOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=chunk_quantity,
                price=order.price,
                attribution_metadata=order.attribution_metadata,
            )

            await self.submit_order(chunk_order)

            await asyncio.sleep(interval_seconds)

        order.update_status(OrderStatus.FILLED)

    async def _execute_iceberg_order(self, order: IcebergOrder):
        """Logic for handling Iceberg orders."""
        remaining_quantity = order.quantity
        while remaining_quantity > 0 and self._running and not order.is_terminal:
            chunk_quantity = min(order.display_quantity, remaining_quantity)
            
            chunk_order = ChaseLimitOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=chunk_quantity,
                price=order.price,
                attribution_metadata=order.attribution_metadata,
            )
            
            chunk_order_id = await self.submit_order(chunk_order)
            
            # Wait for the chunk to be filled
            while self.get_order_status(chunk_order_id) != OrderStatus.FILLED:
                if not self._running or order.is_terminal or self.get_order_status(chunk_order_id) == OrderStatus.CANCELLED or self.get_order_status(chunk_order_id) == OrderStatus.REJECTED or self.get_order_status(chunk_order_id) == OrderStatus.FAILED:
                    break
                await asyncio.sleep(1)

            if self.get_order_status(chunk_order_id) == OrderStatus.FILLED:
                filled_order = self._active_orders[chunk_order_id]
                remaining_quantity -= filled_order.filled_quantity
                order.update_fill(filled_order.filled_quantity, filled_order.average_fill_price)
            else:
                # Chunk order failed or was cancelled, so we stop the iceberg order.
                order.update_status(OrderStatus.CANCELLED, "A chunk of the iceberg order failed or was cancelled.")
                break
        
        if remaining_quantity <= 0:
            order.update_status(OrderStatus.FILLED)

    async def _execute_pegged_order(self, order: PeggedOrder):
        """Logic for handling Pegged orders."""
        logger.warning("Pegged orders are not yet implemented.")
        order.update_status(OrderStatus.REJECTED, "Pegged orders are not yet implemented.")
    
    async def execute_arbitrage_trade(self, ticker: AggregatedTicker, amount: float):
        """Executes a cross-exchange arbitrage trade."""
        if not ticker.arbitrage_opportunity:
            return

        log.info(f"Executing arbitrage trade for {ticker.symbol}...")

        buy_exchange = ticker.best_ask_exchange
        sell_exchange = ticker.best_bid_exchange

        buy_price = ticker.best_ask
        sell_price = ticker.best_bid

        # Create two separate orders
        buy_order = ChaseLimitOrder(
            symbol=ticker.symbol,
            side="buy",
            quantity=amount,
            price=buy_price,
        )
        sell_order = ChaseLimitOrder(
            symbol=ticker.symbol,
            side="sell",
            quantity=amount,
            price=sell_price,
        )

        # We need a way to specify the exchange for each order.
        # This functionality is not yet implemented in the backend.
        # I will add a placeholder for now.
        log.warning("Arbitrage execution is not yet fully implemented.")
        log.warning("Need to add exchange-specific execution to the backend.")

        # await self.submit_order(buy_order, exchange=buy_exchange)
        # await self.submit_order(sell_order, exchange=sell_exchange)

    async def _place_initial_order(self, order: SmartOrder):
        """Place the initial order on the exchange."""
        if not self.backend:
            return

        try:
            params = {}
            if order.is_buy:
                res = await self.backend.create_limit_buy_order(
                    order.symbol, order.quantity, order.price, params
                )
            else:
                res = await self.backend.create_limit_sell_order(
                    order.symbol, order.quantity, order.price, params
                )

            order.exchange_order_id = res["id"]
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
                order
                for order in self._active_orders.values()
                if order.symbol == ticker.symbol and order.is_active
            ]

        for order in orders_to_update:
            try:
                ticker_dict = {
                    "best_bid": ticker.best_bid,
                    "best_ask": ticker.best_ask,
                    "spread_pct": ticker.spread_pct,
                }

                old_price = getattr(order, "price", None)
                order.on_ticker_update(ticker_dict)
                new_price = getattr(order, "price", None)

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

            order.exchange_order_id = new_order["id"]

            logger.info(f"Replaced order. New ID: {new_order['id']}")

        except Exception as e:
            logger.error(f"Failed to replace order {order.order_id}: {e}")

    def get_order_status(self, order_id: str) -> OrderStatus | None:
        """Get the current status of an order."""
        if order_id in self._active_orders:
            return self._active_orders[order_id].status
        return None

    async def emergency_liquidate_all(self):
        """
        🚨 EMERGENCY: Cancel all orders and close all positions immediately.
        Uses market orders for fastest possible liquidation.
        """
        log.critical("🚨 EMERGENCY LIQUIDATION INITIATED!")
        self.telegram.send_message("🚨 <b>EMERGENCY LIQUIDATION INITIATED!</b> 🚨")
        
        # 1. Activate Circuit Breaker
        self.risk_manager.circuit_breaker.trigger("Manual Emergency Stop")
        
        # 2. Cancel all pending smart orders
        async with self._lock:
            order_ids = list(self._active_orders.keys())
            for oid in order_ids:
                await self.cancel_order(oid)
        
        # 3. Fetch all open positions from backend and close them
        try:
            # Note: CCXTBackend needs a method to fetch positions if not already present
            # For now, we use RiskManager's tracked positions as a source
            positions = self.risk_manager.get_active_positions()
            for symbol, pos in positions.items():
                log.info(f"Closing position for {symbol} via Market Order")
                side = "sell" if pos['amount'] > 0 else "buy"
                await self.backend.create_market_order(symbol, side, abs(pos['amount']))
                
            log.info("✅ All positions liquidated.")
            self.telegram.send_message("✅ <b>All positions liquidated successfully.</b>")
        except Exception as e:
            log.error(f"Emergency liquidation failed: {e}")
            self.telegram.send_message(f"❌ <b>Emergency liquidation failed:</b> {e}")

    async def _log_execution(self, order: SmartOrder):
        """Log a completed trade in live mode to the database."""
        try:
            from src.database.db_manager import DatabaseManager
            from src.database.models import ExecutionRecord, TradeRecord
            
            # Calculate metrics
            target_price = order.price
            fill_price = order.average_fill_price
            slippage = abs(fill_price - target_price) / target_price * 100 if target_price > 0 else 0
            
            latency_ms = 0
            if order.signal_timestamp and order.fill_timestamp:
                latency_ms = (order.fill_timestamp - order.signal_timestamp) * 1000

            db = DatabaseManager()
            async with db.session() as session:
                # Find corresponding TradeRecord (this logic depends on how trades are linked)
                # For now, we assume a placeholder link or create a record
                execution = ExecutionRecord(
                    trade_id=0, # Need actual trade ID link
                    symbol=order.symbol,
                    side="buy" if order.is_buy else "sell",
                    target_price=target_price,
                    fill_price=fill_price,
                    slippage_pct=slippage,
                    latency_ms=latency_ms,
                    timestamp=datetime.fromtimestamp(order.fill_timestamp) if order.fill_timestamp else datetime.utcnow(),
                    meta_data=order.attribution_metadata
                )
                session.add(execution)
                await session.commit()
                
            log.info(f"⚡ MFT Execution Logged: {order.symbol} | Latency: {latency_ms:.2f}ms")
            
        except Exception as e:
            log.error("Failed to log execution metrics", error=str(e))

    async def _log_shadow_trade(self, order: SmartOrder):
        """Log a completed trade in shadow mode to the database."""
        if not self._shadow_mode:
            return

        try:
            from src.database.db_manager import DatabaseManager
            from src.database.models import ShadowTradeRecord
            
            # Calculate metrics
            target_price = order.price # In this context, what we wanted
            fill_price = order.average_fill_price
            slippage = abs(fill_price - target_price) / target_price * 100 if target_price > 0 else 0
            
            # Calculate latency
            latency_ms = 0
            if order.signal_timestamp and order.fill_timestamp:
                latency_ms = (order.fill_timestamp - order.signal_timestamp) * 1000

            exchange = order.attribution_metadata.get("exchange", "default") if order.attribution_metadata else "default"

            shadow_record = ShadowTradeRecord(
                symbol=order.symbol,
                side="buy" if order.is_buy else "sell",
                target_price=target_price,
                fill_price=fill_price,
                amount=order.quantity,
                slippage_pct=slippage,
                signal_timestamp=datetime.fromtimestamp(order.signal_timestamp) if order.signal_timestamp else None,
                submission_timestamp=datetime.fromtimestamp(order.submission_timestamp) if order.submission_timestamp else None,
                fill_timestamp=datetime.fromtimestamp(order.fill_timestamp) if order.fill_timestamp else None,
                latency_ms=latency_ms,
                strategy_name=order.attribution_metadata.get("strategy_name", "unknown") if order.attribution_metadata else "unknown",
                meta_data={**(order.attribution_metadata or {}), "exchange": exchange}
            )
            
            # Update Risk Manager positions
            self.risk_manager.record_entry(
                symbol=order.symbol,
                entry_price=fill_price,
                position_size=order.quantity,
                stop_loss_price=order.stop_price or 0,
                exchange=exchange
            )

            db = DatabaseManager()
            async with db.session() as session:
                session.add(shadow_record)
                await session.commit()
                
            log.info(f"📈 Shadow Trade Logged: {order.symbol} @ {fill_price} on {exchange}")
            
        except Exception as e:
            log.error("Failed to log shadow trade", error=str(e))
