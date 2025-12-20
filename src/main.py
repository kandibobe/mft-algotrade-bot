#!/usr/bin/env python3
"""
Main entry point for Stoic Citadel trading bot with graceful shutdown.

This module provides:
1. GracefulShutdown class for handling SIGTERM/SIGINT signals
2. TradingBot wrapper for Freqtrade integration
3. Main entry point with proper shutdown handling

Usage:
    python src/main.py --config user_data/config/config_production.json
"""

import signal
import sys
import time
import threading
from typing import Optional
from pathlib import Path

# Import structured logger
from src.utils.logger import log, setup_structured_logging


class TradingBot:
    """
    Wrapper around Freqtrade trading bot.
    
    This class provides a unified interface for the GracefulShutdown handler
    to control the trading bot during shutdown.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trading bot with configuration.
        
        Args:
            config_path: Path to Freqtrade configuration file
        """
        self.config_path = config_path
        self.is_trading = True
        self.positions = []  # Mock positions list
        log.info("trading_bot_initialized", config_path=config_path)
        
    def pause_trading(self) -> None:
        """
        Stop accepting new trades.
        
        This method should be called during graceful shutdown to prevent
        new trades from being opened while we're closing existing positions.
        """
        self.is_trading = False
        log.info("trading_paused")
        
    def close_all_positions(self, reason: str = "shutdown") -> None:
        """
        Close all open positions.
        
        Args:
            reason: Reason for closing positions (e.g., "shutdown", "risk_limit")
        """
        if not self.positions:
            log.info("no_open_positions", reason=reason)
            return
            
        log.info("closing_positions", 
                 count=len(self.positions), 
                 reason=reason)
        
        # In a real implementation, this would:
        # 1. Iterate through all open positions
        # 2. Create market orders to close each position
        # 3. Wait for orders to fill
        # 4. Update position tracking
        
        # Mock implementation
        self.positions.clear()
        log.info("all_positions_closed", reason=reason)
        
    def flush_buffers(self) -> None:
        """
        Flush all buffers (metrics, logs, cache) to persistent storage.
        
        This ensures no data is lost during shutdown.
        """
        log.info("flushing_buffers")
        
        # In a real implementation, this would:
        # 1. Flush metrics to Prometheus/InfluxDB
        # 2. Flush logs to file/ELK
        # 3. Flush Redis cache to disk
        # 4. Commit any pending database transactions
        
        time.sleep(0.1)  # Simulate flush time
        log.info("buffers_flushed")
        
    def save_state(self, filepath: str) -> None:
        """
        Save bot state to disk for recovery.
        
        Args:
            filepath: Path where state should be saved
        """
        log.info("saving_state", filepath=filepath)
        
        # In a real implementation, this would:
        # 1. Serialize current positions, orders, and bot state
        # 2. Save to pickle file or database
        # 3. Include timestamp and version information
        
        state = {
            "timestamp": time.time(),
            "positions": self.positions,
            "is_trading": self.is_trading,
            "config_path": self.config_path
        }
        
        # Mock implementation - would actually save to file
        log.info("state_saved", 
                 filepath=filepath, 
                 position_count=len(self.positions))
        
    def run(self) -> None:
        """
        Main trading loop.
        
        This method would normally start the Freqtrade bot and run
        the trading strategy continuously.
        """
        log.info("starting_trading_bot")
        
        try:
            # Mock trading loop
            while self.is_trading:
                log.info("trading_cycle", iteration=1)
                time.sleep(5)  # Simulate work
                
                # Break after a few iterations for demo
                break
                
        except KeyboardInterrupt:
            log.info("keyboard_interrupt_received")
        except Exception as e:
            log.error("trading_error", error=str(e), exc_info=e)
        finally:
            log.info("trading_bot_stopped")


class GracefulShutdown:
    """
    Handle graceful shutdown on SIGTERM/SIGINT signals.
    
    This class ensures that:
    1. No new trades are accepted during shutdown
    2. All open positions are closed properly
    3. Buffers (metrics, logs) are flushed to disk
    4. Bot state is saved for recovery
    5. Exit code is 0 (successful shutdown)
    """
    
    def __init__(self, trading_bot):
        """
        Initialize graceful shutdown handler.
        
        Args:
            trading_bot: Instance of TradingBot (or compatible object)
        """
        self.bot = trading_bot
        self.shutdown_initiated = False
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        
        log.info("graceful_shutdown_handler_initialized")
        
    def handle_shutdown(self, signum, frame):
        """
        Handle shutdown signal.
        
        Args:
            signum: Signal number (SIGTERM=15, SIGINT=2)
            frame: Current stack frame
        """
        if self.shutdown_initiated:
            log.debug("shutdown_already_in_progress", signal=signum)
            return
            
        self.shutdown_initiated = True
        
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        log.warning("shutdown_initiated", signal=signal_name, signum=signum)
        
        try:
            # 1. Stop accepting new trades
            self.bot.pause_trading()
            
            # 2. Close all open positions (or save state)
            self.bot.close_all_positions(reason="shutdown")
            
            # 3. Flush metrics/logs
            self.bot.flush_buffers()
            
            # 4. Save state to disk
            self.bot.save_state("user_data/state_checkpoint.pkl")
            
            log.info("shutdown_complete")
            
        except Exception as e:
            log.error("shutdown_error", error=str(e), exc_info=e)
            
        finally:
            # Exit with success code
            sys.exit(0)


def main():
    """
    Main entry point for trading bot.
    
    Sets up structured logging, initializes trading bot,
    and installs graceful shutdown handler.
    """
    # Setup structured logging
    setup_structured_logging(
        level="INFO",
        json_output=True,
        enable_console=True,
        enable_file=True,
        file_path="user_data/logs/trading_bot.log"
    )
    
    log.info("stoic_citadel_starting")
    
    # Default configuration path
    config_path = "user_data/config/config_production.json"
    
    # Check if config file exists
    if not Path(config_path).exists():
        log.error("config_file_not_found", path=config_path)
        print(f"❌ Configuration file not found: {config_path}")
        print("Please create the configuration file or specify a different path.")
        sys.exit(1)
    
    # Initialize trading bot
    bot = TradingBot(config_path)
    
    # Install graceful shutdown handler
    shutdown_handler = GracefulShutdown(bot)
    
    log.info("starting_main_trading_loop")
    
    try:
        # Start trading
        bot.run()
        
    except Exception as e:
        log.error("fatal_error", error=str(e), exc_info=e)
        sys.exit(1)
        
    log.info("stoic_citadel_stopped")


if __name__ == "__main__":
    main()
