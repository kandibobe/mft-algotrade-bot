import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class StrategyTemplate(IStrategy):
    """
    Stoic Citadel Strategy Template
    
    This is a boilerplate for creating your own proprietary strategies.
    Implement your 'Alpha' logic in the populate_indicators, 
    populate_entry_trend, and populate_exit_trend methods.
    """
    
    # Strategy interface version - do not change
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.10
    }

    # Optimal stoploss
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # Use exit_tag for custom exit reasons
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before operating
    startup_candle_count: int = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate technical indicators for the strategy
        """
        # Example: Add RSI
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals
        """
        dataframe.loc[
            (
                # (dataframe['rsi'] < 30) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals
        """
        dataframe.loc[
            (
                # (dataframe['rsi'] > 70) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        return dataframe
