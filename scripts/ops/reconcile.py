#!/usr/bin/env python3
"""
Reconciliation Script (Сверка Балансов)
=======================================

Periodically reconciles RiskManager positions with actual exchange data.
Halt trading if discrepancies are found.
"""

import asyncio
import logging
import os
import sys
import time
from decimal import Decimal

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.manager import config
from src.risk.risk_manager import RiskManager
from src.order_manager.exchange_backend import CCXTBackend
from src.notification.telegram import TelegramBot
from src.utils.logger import log

logger = logging.getLogger("reconciler")

class Reconciler:
    def __init__(self, mode="prod"):
        self.reconciler_mode = mode
        self.cfg = config()
        self.risk_manager = RiskManager()
        self.telegram = TelegramBot()
        
        # Initialize backend
        exchange_config = {
            "name": self.cfg.exchange.name,
            "key": getattr(self.cfg.exchange, "api_key", None),
            "secret": getattr(self.cfg.exchange, "api_secret", None),
            "enableRateLimit": True,
        }
        self.backend = CCXTBackend(exchange_config)
        self._running = False

    async def start(self):
        self._running = True
        log.info(f"Starting Reconciliation Service (Mode: {self.reconciler_mode})")
        await self.backend.initialize()
        
        while self._running:
            try:
                await self.reconcile()
                await asyncio.sleep(60) # Run once per minute
            except Exception as e:
                log.error(f"Reconciliation error: {e}")
                await asyncio.sleep(10)

    async def stop(self):
        self._running = False
        await self.backend.close()

    async def reconcile(self):
        log.info("Performing balance and position reconciliation...")
        
        # 1. Fetch data from exchange
        try:
            # Check if backend is initialized and has credentials
            if not self.backend.exchange.apiKey and not self.reconciler_mode == "test":
                log.warning("Skipping reconciliation: No API keys configured.")
                return

            exchange_positions = await self.backend.fetch_positions()
        except Exception as e:
            log.error(f"Failed to fetch data from exchange: {e}")
            return

        # 2. Get data from RiskManager
        bot_positions = self.risk_manager._positions
        
        discrepancies = []

        # 3. Compare positions
        # Check if bot thinks it's in position but exchange says no
        for symbol, bot_pos in bot_positions.items():
            ex_pos = next((p for p in exchange_positions if p['symbol'] == symbol), None)
            
            if not ex_pos or abs(ex_pos['contracts']) < 1e-8:
                discrepancies.append(f"MISSING POSITION: Bot thinks {symbol} is open, but exchange says closed.")
            elif abs(float(bot_pos['size']) - ex_pos['contracts']) / max(abs(float(bot_pos['size'])), 1e-8) > 0.01:
                discrepancies.append(f"SIZE MISMATCH: {symbol} Bot: {bot_pos['size']}, Exchange: {ex_pos['contracts']}")

        # Check if exchange has positions bot doesn't know about
        for ex_pos in exchange_positions:
            if abs(ex_pos['contracts']) > 1e-8:
                symbol = ex_pos['symbol']
                if symbol not in bot_positions:
                    discrepancies.append(f"GHOST POSITION: Exchange has {symbol} open, but bot thinks it is closed.")

        # 4. Handle discrepancies
        if discrepancies:
            msg = "🚨 <b>RECONCILIATION FAILURE!</b> 🚨\n\n" + "\n".join(discrepancies)
            log.critical(msg)
            
            # Emergency Stop
            self.risk_manager.emergency_stop()
            await self.telegram.send_message_async(msg + "\n\n🛑 <b>TRADING HALTED.</b>")
        else:
            log.info("✅ Reconciliation successful. No discrepancies found.")

async def main():
    reconciler = Reconciler()
    try:
        await reconciler.start()
    except KeyboardInterrupt:
        await reconciler.stop()

if __name__ == "__main__":
    asyncio.run(main())
