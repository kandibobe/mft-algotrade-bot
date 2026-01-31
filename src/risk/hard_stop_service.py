"""
Nuclear Option Daemon
=====================

Autonomous daemon that monitors total account equity. If a catastrophic
drawdown is detected, it revokes API keys and terminates the bot process
as the ultimate failsafe.
"""

import asyncio
import logging
import os
import signal
import time
from collections import deque
from datetime import timedelta

import ccxt.async_support as ccxt

from src.config.unified_config import load_config
from src.order_manager.exchange_backend import CCXTBackend
from src.utils.logger import log

logger = logging.getLogger(__name__)


class NuclearOptionDaemon:
    """
    Monitors account equity and acts as the ultimate circuit breaker.
    """

    def __init__(
        self,
        backend: CCXTBackend,
        pid_to_kill: int,
        poll_interval: float = 60.0,  # Check every 60 seconds
        drawdown_threshold: float = 0.10,  # 10%
        monitoring_window: timedelta = timedelta(hours=1),
    ):
        self.backend = backend
        self.pid_to_kill = pid_to_kill
        self.poll_interval = poll_interval
        self.drawdown_threshold = drawdown_threshold
        self.monitoring_window_seconds = monitoring_window.total_seconds()
        self._running = False
        self._task: asyncio.Task | None = None
        self.equity_history = deque(
            maxlen=int(self.monitoring_window_seconds / poll_interval)
        )

    async def start(self):
        """Start the monitoring task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        log.info("🛡️ Nuclear Option Daemon started. Monitoring account equity.")

    async def stop(self):
        """Stop the monitoring task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("Nuclear Option Daemon stopped.")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_account_equity()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Error in Nuclear Option Daemon", error=str(e))
                await asyncio.sleep(self.poll_interval * 2)

    async def _check_account_equity(self):
        """Fetch and evaluate account equity."""
        try:
            # We need to implement a way to fetch total equity
            # For now, let's assume the backend has a fetch_total_equity method
            balance = await self.backend.exchange.fetch_balance()
            # This is a simplification; a real implementation needs to handle
            # different quote currencies. We'll assume USDT for now.
            current_equity = balance["total"]["USDT"]
            
            if not current_equity:
                 log.warning("Could not determine account equity.")
                 return

            timestamp = time.time()
            self.equity_history.append((timestamp, current_equity))

            # Not enough data to make a decision yet
            if len(self.equity_history) < self.equity_history.maxlen * 0.5:
                return

            # Find peak equity in the window
            peak_equity = max(value for _, value in self.equity_history)

            drawdown = (peak_equity - current_equity) / peak_equity

            if drawdown > self.drawdown_threshold:
                log.critical(f"NUCLEAR OPTION TRIGGERED! Drawdown: {drawdown:.2%}")
                await self._nuke_it_all()
            else:
                log.info(f"Equity check OK. Current: {current_equity:.2f}, Peak: {peak_equity:.2f}, Drawdown: {drawdown:.2%}")

        except Exception as e:
            log.error("Failed to check account equity", error=str(e))

    async def _nuke_it_all(self):
        """The final failsafe."""
        log.warning("🚨 INITIATING NUCLEAR OPTION 🚨")

        # 1. Revoke API Keys (Conceptual - very few exchanges support this)
        try:
            log.warning("Attempting to revoke API keys...")
            # This is a placeholder for a real implementation
            # success = await self.backend.revoke_api_key()
            # if success:
            #     log.info("✅ API keys revoked.")
            # else:
            #     log.warning("⚠️  Could not revoke API keys via API.")
            log.warning("API key revocation is not supported by most exchanges. Please do this manually.")
        except Exception as e:
            log.error("Failed to revoke API keys", error=str(e))

        # 2. Kill the main bot process
        try:
            log.warning(f"Terminating bot process (PID: {self.pid_to_kill})...")
            if self.pid_to_kill > 0:
                os.kill(self.pid_to_kill, signal.SIGTERM)
                log.info(f"✅ SIGTERM signal sent to PID {self.pid_to_kill}.")
            else:
                log.error("Invalid PID to kill.")
        except ProcessLookupError:
            log.warning(f"Process with PID {self.pid_to_kill} not found. It might have already stopped.")
        except Exception as e:
            log.error("Failed to kill bot process", error=str(e))

        # 3. Stop this daemon
        await self.stop()


async def main():
    """Entry point for the standalone daemon."""
    # This should be configured via a lightweight config or env vars
    pid_file_path = "bot.pid"  # Assume the bot writes its PID here
    
    if not os.path.exists(pid_file_path):
        log.error(f"PID file not found at {pid_file_path}. Exiting.")
        return

    with open(pid_file_path, "r") as f:
        pid = int(f.read().strip())
        
    config = load_config()

    exchange_config = {
        "name": config.exchange.name,
        "key": config.exchange.api_key,
        "secret": config.exchange.api_secret,
    }

    backend = CCXTBackend(exchange_config)
    await backend.initialize()
    
    daemon = NuclearOptionDaemon(
        backend=backend,
        pid_to_kill=pid,
    )
    
    await daemon.start()
    
    # Keep the daemon running
    while daemon._running:
        await asyncio.sleep(1)
        
    await backend.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Daemon stopped by user.")

