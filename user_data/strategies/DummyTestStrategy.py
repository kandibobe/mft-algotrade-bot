"""
Dummy Test Strategy for Infrastructure Testing (Track A)

This is a simple strategy used to test the bot infrastructure in dry-run mode.
It makes random trades or uses simple RSI to verify:
1. WebSocket stability
2. Telegram alerts
3. Log rotation
4. Order execution logic

Goal: Find bugs in the ENGINE, not the model.

Author: CTO & Lead Quantitative Researcher
Version: 1.0.0
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DummyTestStrategy(IStrategy):
    """
    Dummy strategy for infrastructure testing.
    
    Two modes:
    1. Random mode: Makes random buy/sell decisions (good for stress testing)
    2. Simple RSI mode: Uses basic RSI strategy (good for logic testing)
    """
    
    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================
    
    INTERFACE_VERSION = 3
    
    # Strategy mode: "random" or "rsi"
    MODE = "random"  # Change to "rsi" for RSI-based testing
    
    # Minimal ROI - very conservative for testing
    minimal_roi = {
        "0": 0.05,    # 5% profit
        "30": 0.03,   # 3% profit after 30 candles
        "60": 0.01,   # 1% profit after 60 candles
        "120": 0.0    # Break even after 120 candles
    }
    
    # Stoploss - very tight for testing
    stoploss = -0.02  # 2% stoploss
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '5m'
    
    # Processing
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 100
    
    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60
    }
    
    # ==========================================================================
    # PROTECTIONS
    # ==========================================================================
    
    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 60,
                "trade_limit": 5,
                "stop_duration_candles": 30,
                "required_profit": 0.0
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,
                "trade_limit": 10,
                "stop_duration_candles": 60,
                "max_allowed_drawdown": 0.10  # 10% max drawdown
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            }
        ]
    
    # ==========================================================================
    # INDICATORS
    # ==========================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators for the strategy.
        """
        # Simple RSI for RSI mode
        dataframe['rsi'] = self.calculate_rsi(dataframe['close'], 14)
        
        # Simple moving averages
        dataframe['sma_20'] = dataframe['close'].rolling(window=20).mean()
        dataframe['sma_50'] = dataframe['close'].rolling(window=50).mean()
        
        # Volume indicator
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # For random mode: add a random signal column
        if self.MODE == "random":
            # Generate reproducible random signals based on timestamp
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            dataframe['random_signal'] = np.random.randint(0, 100, size=len(dataframe))
        
        logger.debug(f"Populated indicators for {metadata['pair']}, mode: {self.MODE}")
        return dataframe
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions.
        """
        if self.MODE == "random":
            # Random entry: 5% chance to buy
            dataframe.loc[
                (dataframe['random_signal'] < 5) &  # 5% chance
                (dataframe['volume_ratio'] > 0.5) &  # Some volume
                (dataframe.index % 10 == 0),  # Only every 10th candle
                'enter_long'
            ] = 1
            
        elif self.MODE == "rsi":
            # Simple RSI strategy: buy when RSI < 30 (oversold)
            dataframe.loc[
                (dataframe['rsi'] < 30) &  # Oversold
                (dataframe['close'] > dataframe['sma_20']) &  # Above short MA
                (dataframe['volume_ratio'] > 1.0),  # Above average volume
                'enter_long'
            ] = 1
        
        logger.debug(f"Entry signals for {metadata['pair']}: {dataframe['enter_long'].sum()}")
        return dataframe
    
    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions.
        """
        if self.MODE == "random":
            # Random exit: 10% chance to sell
            dataframe.loc[
                (dataframe['random_signal'] > 90) &  # 10% chance
                (dataframe.index % 5 == 0),  # Only every 5th candle
                'exit_long'
            ] = 1
            
        elif self.MODE == "rsi":
            # Simple RSI strategy: sell when RSI > 70 (overbought)
            dataframe.loc[
                (dataframe['rsi'] > 70) &  # Overbought
                (dataframe['close'] < dataframe['sma_20']),  # Below short MA
                'exit_long'
            ] = 1
        
        logger.debug(f"Exit signals for {metadata['pair']}: {dataframe['exit_long'].sum()}")
        return dataframe
    
    # ==========================================================================
    # CUSTOM METHODS
    # ==========================================================================
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Final validation before entry.
        
        For testing purposes, we add some random rejection to test
        the order management system.
        """
        # Reject 10% of trades randomly to test error handling
        import random
        if random.random() < 0.1:
            logger.info(f"Randomly rejecting trade for {pair} to test error handling")
            return False
        
        # Don't trade during low liquidity hours (for testing time filters)
        hour = current_time.hour
        if hour in [0, 1, 2, 3, 4, 5]:  # 12am-6am UTC
            logger.info(f"Skipping {pair}: Low liquidity hours ({hour}:00 UTC)")
            return False
        
        return True
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Fixed stake amount for testing.
        """
        # Use a fixed small stake for testing
        test_stake = 15.0  # Small amount for testing
        
        # Apply bounds
        if min_stake and test_stake < min_stake:
            test_stake = min_stake
        if test_stake > max_stake:
            test_stake = max_stake
        
        logger.debug(f"Using test stake: {test_stake} for {pair}")
        return test_stake
    
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        No leverage for testing.
        """
        return 1.0  # No leverage for testing
    
    # ==========================================================================
    # LOGGING AND MONITORING
    # ==========================================================================
    
    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of each bot loop iteration.
        Good for logging and monitoring.
        """
        logger.info(f"Dummy strategy loop start - Mode: {self.MODE}")
        
        # Log some stats
        if hasattr(self, 'dp'):
            try:
                pairs = self.dp.current_whitelist()
                logger.info(f"Tracking {len(pairs)} pairs: {pairs}")
            except:
                pass
    
    def bot_start(self, **kwargs) -> None:
        """
        Called when the bot starts.
        """
        logger.info("=" * 60)
        logger.info("DUMMY TEST STRATEGY STARTED")
        logger.info(f"Mode: {self.MODE}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info("Goal: Test infrastructure, not make profits")
        logger.info("=" * 60)
    
    def bot_stop(self, **kwargs) -> None:
        """
        Called when the bot stops.
        """
        logger.info("=" * 60)
        logger.info("DUMMY TEST STRATEGY STOPPED")
        logger.info("=" * 60)


# ==============================================================================
# STRATEGY CONFIGURATION HELPER
# ==============================================================================

def get_dummy_strategy_config(mode: str = "random") -> Dict:
    """
    Get configuration for the dummy strategy.
    
    Args:
        mode: "random" or "rsi"
    
    Returns:
        Dictionary with strategy configuration
    """
    return {
        "strategy": "DummyTestStrategy",
        "strategy_params": {
            "MODE": mode
        },
        "timeframe": "5m",
        "dry_run": True,
        "dry_run_wallet": 1000.0,
        "max_open_trades": 3,
        "stake_amount": 15.0,
        "stake_currency": "USDT",
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": [
                "BTC/USDT",
                "ETH/USDT"
            ],
            "pair_blacklist": []
        },
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1
            }
        },
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "order_book_top": 1
        },
        "telegram": {
            "enabled": True,
            "token": "",
            "chat_id": "",
            "notification_settings": {
                "status": "on",
                "warning": "on",
                "startup": "on",
                "entry": "on",
                "entry_fill": "on",
                "exit": "on",
                "exit_fill": "on",
                "protection_trigger": "on",
                "protection_trigger_global": "on"
            }
        },
        "api_server": {
            "enabled": True,
            "listen_ip_address": "0.0.0.0",
            "listen_port": 8080,
            "verbosity": "error",
            "enable_openapi": False,
            "jwt_secret_key": "dummy-test-key-change-in-production",
            "CORS_origins": [],
            "username": "dummy",
            "password": "test123"
        }
    }
