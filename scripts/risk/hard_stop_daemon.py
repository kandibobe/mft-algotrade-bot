"""
Hard Stop Daemon - The "Nuclear Option"
========================================

A separate, tiny daemon process that watches the Exchange Account Balance
directly via API. If equity drops > 10% in 1 hour, it triggers a hard stop.
"""

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass

import ccxt.async_support as ccxt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class BalanceSnapshot:
    timestamp: float
    balance: float

class HardStopDaemon:
    def __init__(self, exchange_config: dict, pid_file: str):
        self.exchange_config = exchange_config
        self.pid_file = pid_file
        self.history = deque(maxlen=60)  # Store 1 hour of data (60 samples * 60s interval)
        self.check_interval_seconds = 60
        self.threshold_pct = -10.0 # -10% drop

    async def run(self):
        logging.info("Starting Hard Stop Daemon.")
        
        exchange_class = getattr(ccxt, self.exchange_config.get("name", "binance"))
        exchange = exchange_class({
            "apiKey": self.exchange_config.get("key"),
            "secret": self.exchange_config.get("secret"),
        })

        try:
            while True:
                await self.check_balance(exchange)
                await asyncio.sleep(self.check_interval_seconds)
        finally:
            await exchange.close()

    async def check_balance(self, exchange):
        try:
            balance_data = await exchange.fetch_balance()
            total_balance = balance_data['total']['USDT'] # Assuming USDT for now
            
            now = time.time()
            self.history.append(BalanceSnapshot(timestamp=now, balance=total_balance))
            logging.info(f"Current Balance: {total_balance:.2f} USDT")

            # Check for drawdown
            if len(self.history) > 1:
                one_hour_ago = now - 3600
                relevant_history = [s for s in self.history if s.timestamp >= one_hour_ago]
                
                if relevant_history:
                    max_balance_in_window = max(s.balance for s in relevant_history)
                    drawdown_pct = (total_balance - max_balance_in_window) / max_balance_in_window * 100
                    
                    logging.info(f"1-hour Drawdown: {drawdown_pct:.2f}% (Max balance in window: {max_balance_in_window:.2f})")

                    if drawdown_pct < self.threshold_pct:
                        logging.critical(f"HARD STOP TRIGGERED: Equity dropped by {abs(drawdown_pct):.2f}% in the last hour!")
                        await self.trigger_shutdown()

        except Exception as e:
            logging.error(f"Error checking balance: {e}")

    async def trigger_shutdown(self):
        logging.info("Attempting to shut down the main bot process...")
        try:
            if os.path.exists(self.pid_file):
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                logging.info(f"Found bot PID: {pid}. Sending termination signal.")
                # Use a method that can be caught for graceful shutdown if possible
                os.kill(pid, 15) # SIGTERM
                
                # Here you could add API key revocation logic if the exchange supports it
                logging.warning("API key revocation is NOT IMPLEMENTED.")

            else:
                logging.error("PID file not found. Cannot shut down bot.")

        except Exception as e:
            logging.error(f"Failed to trigger shutdown: {e}")
        
        # Stop the daemon itself
        asyncio.get_event_loop().stop()


async def main():
    # This would be loaded from a secure config in a real scenario
    from src.config.unified_config import load_config
    config = load_config()
    
    exchange_config = {
        "name": config.exchange.name,
        "key": config.exchange.api_key,
        "secret": config.exchange.api_secret,
    }
    
    pid_file = str(config.paths.user_data_dir / "freqtrade.pid")

    daemon = HardStopDaemon(exchange_config, pid_file)
    await daemon.run()

if __name__ == "__main__":
    # Note: This should be run as a separate process from the main bot.
    # e.g., using supervisor, systemd, or another process manager.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Hard Stop Daemon shut down manually.")

