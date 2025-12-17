"""
Stoic Citadel - Ensemble Strategy Template
===========================================

Philosophy: "The wise man accepts losses with equanimity."

This is a professional-grade strategy template implementing:
1. Trend Filter (EMA 200) - Don't fight the macro trend
2. Entry Oscillator (RSI + Stochastic) - Buy pullbacks in trends
3. Strict Risk Management - Capital preservation first
4. ML Validator (placeholder) - Filter false signals

Author: Stoic Citadel Team
Version: 1.0.0
License: MIT
"""

from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import talib.abstract as ta
import pandas_ta as pta
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StoicEnsembleStrategy(IStrategy):
    """
    Ensemble strategy combining trend following with mean reversion entries.

    The Strategy:
    -------------
    1. TREND FILTER: Only long when price > EMA200 (bull market)
    2. ENTRY SIGNAL: RSI oversold (<30) + Stochastic crossover
    3. EXIT SIGNAL: RSI overbought (>70) or trailing stop
    4. RISK: Hard stop at -5%, trailing stop at +1%

    Timeframe: 5m (execution), 1h (trend confirmation)
    """

    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================

    INTERFACE_VERSION = 3

    # Minimal ROI - Take profits at these levels
    minimal_roi = {
        "0": 0.15,   # 15% profit - close immediately
        "30": 0.08,  # 8% profit after 30 minutes
        "60": 0.05,  # 5% profit after 1 hour
        "120": 0.03  # 3% profit after 2 hours
    }

    # Stoploss - HARD LIMIT (The Stoic Guard)
    stoploss = -0.05  # -5% maximum loss per trade

    # Trailing stop - Lock in profits
    trailing_stop = True
    trailing_stop_positive = 0.01      # Start trailing at +1%
    trailing_stop_positive_offset = 0.015  # Trail by 1.5%
    trailing_only_offset_is_reached = True

    # Timeframes
    timeframe = '5m'

    # Run "populate_indicators()" only for new candles
    process_only_new_candles = True

    # Don't use signals from previous candles
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60
    }

    # Optional order time in force
    order_time_in_force = {
        'entry': 'GTC',  # Good Till Cancelled
        'exit': 'GTC'
    }

    # ==========================================================================
    # PROTECTIONS (The Stoic Cooldown)
    # ==========================================================================

    @property
    def protections(self):
        """
        Protection mechanisms to prevent revenge trading and overtrading.
        """
        return [
            {
                # Stop trading after 3 consecutive losses
                "method": "StoplossGuard",
                "lookback_period_candles": 60,  # Last 5 hours (5m tf)
                "trade_limit": 3,
                "stop_duration_candles": 24,    # Cooldown: 2 hours
                "required_profit": 0.0
            },
            {
                # Prevent trading on pairs with low win rate
                "method": "LowProfitPairs",
                "lookback_period_candles": 360,  # Last 30 hours
                "trade_limit": 2,
                "stop_duration_candles": 60,     # Cooldown: 5 hours
                "required_profit": -0.05         # If losing > 5%
            },
            {
                # Max drawdown protection
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,  # Last 24 hours
                "trade_limit": 5,
                "stop_duration_candles": 48,     # Cooldown: 4 hours
                "max_allowed_drawdown": 0.15     # 15% max DD
            }
        ]

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators needed for the strategy.

        This runs on EVERY candle, so keep it efficient!
        """

        # ------------------------------------------------------------------
        # TREND INDICATORS
        # ------------------------------------------------------------------

        # EMA - Exponential Moving Averages (trend direction)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # ADX - Average Directional Index (trend strength)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # ------------------------------------------------------------------
        # OSCILLATORS (Entry Signals)
        # ------------------------------------------------------------------

        # RSI - Relative Strength Index
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Stochastic - Another oscillator for confirmation
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        # MACD - Moving Average Convergence Divergence
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # ------------------------------------------------------------------
        # VOLATILITY INDICATORS
        # ------------------------------------------------------------------

        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_width'] = (
            (dataframe['bb_upperband'] - dataframe['bb_lowerband']) /
            dataframe['bb_middleband']
        )

        # ATR - Average True Range (for position sizing)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # ------------------------------------------------------------------
        # VOLUME INDICATORS
        # ------------------------------------------------------------------

        # Volume trend
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        # ------------------------------------------------------------------
        # CUSTOM FEATURES (For ML Model - Future Implementation)
        # ------------------------------------------------------------------

        # Price change percentages
        dataframe['pct_change_1'] = dataframe['close'].pct_change(1)
        dataframe['pct_change_3'] = dataframe['close'].pct_change(3)
        dataframe['pct_change_5'] = dataframe['close'].pct_change(5)

        # Trend strength score (custom)
        dataframe['trend_score'] = (
            (dataframe['ema_50'] > dataframe['ema_100']).astype(int) +
            (dataframe['ema_100'] > dataframe['ema_200']).astype(int) +
            (dataframe['close'] > dataframe['ema_50']).astype(int)
        )

        return dataframe

    # ==========================================================================
    # ENTRY LOGIC (The Wise Man's Patience)
    # ==========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the conditions for ENTERING a trade.

        Philosophy: "Enter when others are fearful, but only in a bull trend."
        """

        conditions = []

        # ------------------------------------------------------------------
        # CONDITION 1: TREND FILTER (Mandatory)
        # ------------------------------------------------------------------
        # Only trade when price is above EMA200 (bull market)
        trend_filter = (
            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['ema_50'] > dataframe['ema_100']) &
            (dataframe['adx'] > 20)  # Trend has strength
        )
        conditions.append(trend_filter)

        # ------------------------------------------------------------------
        # CONDITION 2: ENTRY SIGNAL (Oscillator Oversold)
        # ------------------------------------------------------------------
        # Buy on pullbacks within the uptrend
        entry_signal = (
            (dataframe['rsi'] < 35) &  # RSI oversold
            (dataframe['slowk'] < 30) &  # Stochastic oversold
            (dataframe['slowk'] > dataframe['slowd'])  # Stoch crossover
        )
        conditions.append(entry_signal)

        # ------------------------------------------------------------------
        # CONDITION 3: VOLUME CONFIRMATION
        # ------------------------------------------------------------------
        # Ensure there's actual interest in the pair
        volume_filter = (
            dataframe['volume'] > dataframe['volume_mean'] * 0.8
        )
        conditions.append(volume_filter)

        # ------------------------------------------------------------------
        # CONDITION 4: VOLATILITY FILTER
        # ------------------------------------------------------------------
        # Don't trade in dead markets or extreme volatility
        volatility_filter = (
            (dataframe['bb_width'] > 0.02) &  # Minimum volatility
            (dataframe['bb_width'] < 0.20)    # Maximum volatility
        )
        conditions.append(volatility_filter)

        # ------------------------------------------------------------------
        # COMBINE ALL CONDITIONS
        # ------------------------------------------------------------------
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    # ==========================================================================
    # EXIT LOGIC (The Wise Man's Profit Taking)
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the conditions for EXITING a trade.

        Philosophy: "Take profits when others are greedy."
        """

        conditions = []

        # ------------------------------------------------------------------
        # EXIT CONDITION 1: Overbought Signal
        # ------------------------------------------------------------------
        overbought_exit = (
            (dataframe['rsi'] > 75) &  # RSI overbought
            (dataframe['slowk'] > 80)  # Stochastic overbought
        )
        conditions.append(overbought_exit)

        # ------------------------------------------------------------------
        # EXIT CONDITION 2: Trend Reversal
        # ------------------------------------------------------------------
        trend_reversal = (
            (dataframe['close'] < dataframe['ema_50']) &
            (dataframe['macd'] < dataframe['macdsignal'])
        )
        conditions.append(trend_reversal)

        # ------------------------------------------------------------------
        # COMBINE ALL CONDITIONS
        # ------------------------------------------------------------------
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),  # OR logic for exits
                'exit_long'
            ] = 1

        return dataframe

    # ==========================================================================
    # CUSTOM METHODS (Optional Enhancements)
    # ==========================================================================

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
        Dynamic position sizing based on volatility (ATR).

        Higher volatility = smaller position (risk parity).
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # Get ATR (volatility measure)
        atr = current_candle['atr']
        close = current_candle['close']

        # Calculate volatility percentage
        volatility_pct = atr / close

        # Adjust stake based on volatility
        # High volatility = smaller position
        if volatility_pct > 0.05:  # > 5% volatility
            adjusted_stake = proposed_stake * 0.5
        elif volatility_pct > 0.03:  # > 3% volatility
            adjusted_stake = proposed_stake * 0.75
        else:  # Low volatility
            adjusted_stake = proposed_stake

        return max(min_stake or 0, min(adjusted_stake, max_stake))

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
        Final gate before entering a trade.

        This is where you'd plug in an ML model to validate the signal.
        For now, it's just a basic check.
        """

        # Example: Don't trade during low liquidity hours (placeholder)
        hour = current_time.hour

        # Avoid trading during typical low liquidity periods
        if hour in [0, 1, 2, 3, 4, 5]:
            logger.info(f"Rejecting entry for {pair} - Low liquidity hours")
            return False

        # ML Model would go here:
        # ml_score = self.ml_model.predict(features)
        # if ml_score < 0.65:
        #     return False

        return True

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """
        Custom exit logic (emergency exits, time-based exits, etc.)
        """

        # Emergency exit: If trade is open for > 24 hours and losing
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600

        if trade_duration > 24 and current_profit < -0.02:
            return "emergency_exit_24h"

        # Take profit on strong moves
        if current_profit > 0.10:  # 10% profit
            return "take_profit_10pct"

        return None


# Required import for reduce function
from functools import reduce
