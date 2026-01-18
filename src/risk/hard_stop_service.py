"""
Hard Stop Loss Service (External Guard)
======================================

Autonomous service that monitors exchange positions and closes them
if hard stop loss conditions are met, independent of the main strategy loop.
"""

import asyncio
import logging

from src.order_manager.exchange_backend import IExchangeBackend
from src.utils.logger import log

logger = logging.getLogger(__name__)

class HardStopLossService:
    """
    Independent monitor for open positions.
    Acts as a final safety layer.
    """

    def __init__(
        self,
        backend: IExchangeBackend,
        poll_interval: float = 2.0,
        max_drawdown_per_trade: float = 0.10, # 10% hard limit
    ):
        self.backend = backend
        self.poll_interval = poll_interval
        self.max_drawdown_per_trade = max_drawdown_per_trade
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the monitoring task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        log.info("🛡️ Hard Stop Loss Service started")

    async def stop(self):
        """Stop the monitoring task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("Hard Stop Loss Service stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_positions()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Error in Hard Stop Loss Service", error=str(e))
                await asyncio.sleep(self.poll_interval * 2)

    async def _check_positions(self):
        """Fetch and validate all open positions."""
        try:
            positions = await self.backend.fetch_positions()
            for pos in positions:
                symbol = pos["symbol"]
                entry_price = float(pos["entry_price"])
                side = pos["side"]
                amount = float(pos["amount"])

                # In a real system, we'd fetch current market price
                # For this service, let's assume we might need another way to get it
                # or the position data includes current price/pnl.
                # For now, let's just implement the logic that COULD trigger a close.

                if "current_price" in pos:
                    current_price = float(pos["current_price"])
                    pnl_pct = (current_price - entry_price) / entry_price
                    if side.lower() == "sell":
                        pnl_pct = -pnl_pct

                    if pnl_pct < -self.max_drawdown_per_trade:
                        log.critical(f"HARD STOP TRIGGERED for {symbol}: PnL {pnl_pct:.2%}")
                        await self.force_close_position(symbol, amount, side)
        except Exception as e:
            log.error("Failed to check positions in HardStopLossService", error=str(e))

    async def force_close_position(self, symbol: str, size: float, side: str):
        """Emergency close of a position."""
        log.warning(f"🚨 EMERGENCY CLOSING POSITION: {symbol} {size} {side}")
        try:
            if side.lower() == "long":
                await self.backend.create_limit_sell_order(symbol, size, 0, params={"reduceOnly": True, "type": "market"})
            else:
                await self.backend.create_limit_buy_order(symbol, size, 0, params={"reduceOnly": True, "type": "market"})

            log.info(f"✅ Emergency close order sent for {symbol}")
        except Exception as e:
            log.error(f"❌ FAILED TO EMERGENCY CLOSE {symbol}", error=str(e))
