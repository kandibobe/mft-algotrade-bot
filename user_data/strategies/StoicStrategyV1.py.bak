"""
Stoic Citadel - Strategy V1 (Production Grade)
==============================================

Market Regime-Aware Trend Following Strategy with HyperOpt Integration

Key Features:
-------------
1. Market Regime Filter: BTC/USDT 1d timeframe (only long when BTC > EMA200)
2. HyperOptimizable Parameters: RSI thresholds, stoploss, ROI
3. Strict Risk Management: -5% hard stop, scalping ROI profile
4. Walk-Forward Compatible: Designed for train/test validation

Author: Stoic Citadel Team
Version: 1.0.0
License: MIT
"""

from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import talib.abstract as ta
from typing import Optional, Dict
from datetime import datetime
from functools import reduce
import logging

# HyperOpt imports
from freqtrade.optimize.space import Categorical, Decimal, Integer

logger = logging.getLogger(__name__)


class StoicStrategyV1(IStrategy):
    """
    Production-grade strategy with market regime awareness.

    Strategy Logic:
    ---------------
    1. REGIME FILTER: Check if BTC is bullish (BTC > EMA200 on 1d)
    2. ENTRY: RSI oversold + uptrend confirmation on main pair
    3. EXIT: RSI overbought OR stop loss OR ROI
    4. RISK: -5% hard stop, tight ROI for scalping
    """

    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================

    INTERFACE_VERSION = 3

    # Timeframe
    timeframe = '5m'

    # Startup candle count
    startup_candle_count: int = 200

    # Can short? (Set to False for long-only)
    can_short: bool = False

    # ==========================================================================
    # HYPEROPTIMIZABLE PARAMETERS
    # ==========================================================================

    # ROI - Take profit targets (optimizable)
    minimal_roi = {
        "0": 0.06,   # 6% immediate profit
        "20": 0.04,  # 4% after 100 minutes (20 * 5m)
        "40": 0.02,  # 2% after 200 minutes
        "60": 0.01   # 1% after 300 minutes
    }

    # Stoploss (optimizable)
    stoploss = -0.05  # -5% hard stop

    # Trailing stop
    trailing_stop = False  # Keep it simple for now
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Exit signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False

    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Process only new candles
    process_only_new_candles = True

    # ==========================================================================
    # HYPEROPT PARAMETER SPACES
    # ==========================================================================

    # Entry parameters (optimizable)
    buy_rsi_threshold = Integer(20, 40, default=30, space='buy', optimize=True)
    buy_rsi_enabled = Categorical([True, False], default=True, space='buy', optimize=False)

    # Exit parameters (optimizable)
    sell_rsi_threshold = Integer(60, 80, default=70, space='sell', optimize=True)
    sell_rsi_enabled = Categorical([True, False], default=True, space='sell', optimize=False)

    # Market regime parameters
    regime_ema_period = Integer(150, 250, default=200, space='buy', optimize=True)

    # ==========================================================================
    # INFORMATIVE PAIRS (MARKET REGIME)
    # ==========================================================================

    def informative_pairs(self):
        """
        Define additional pairs to fetch for market regime analysis.

        Returns BTC/USDT on 1d timeframe to determine overall market trend.
        """
        return [
            ('BTC/USDT', '1d'),  # Market regime indicator
        ]

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================

    @informative('1d', 'BTC/USDT')
    def populate_indicators_btc(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators for BTC/USDT 1d (market regime).

        This runs automatically due to @informative decorator.
        """
        # EMA for regime detection
        dataframe['ema_regime'] = ta.EMA(
            dataframe,
            timeperiod=self.regime_ema_period.value
        )

        # Regime flag: 1 = bull, 0 = bear/neutral
        dataframe['regime_bull'] = (
            dataframe['close'] > dataframe['ema_regime']
        ).astype(int)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate technical indicators for the main trading pair.

        This runs on the strategy's primary timeframe (5m).
        """

        # ------------------------------------------------------------------
        # TREND INDICATORS
        # ------------------------------------------------------------------

        # EMA - Exponential Moving Averages
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA - Simple Moving Averages
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)

        # ADX - Trend Strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # ------------------------------------------------------------------
        # OSCILLATORS
        # ------------------------------------------------------------------

        # RSI - Relative Strength Index (primary signal)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # ------------------------------------------------------------------
        # VOLATILITY
        # ------------------------------------------------------------------

        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']

        # ATR - Average True Range
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # ------------------------------------------------------------------
        # VOLUME
        # ------------------------------------------------------------------

        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        return dataframe

    # ==========================================================================
    # ENTRY LOGIC (BUY)
    # ==========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define conditions for entering a trade (LONG only).

        Multi-layered approach:
        1. REGIME CHECK: BTC must be bullish (> EMA200 on 1d)
        2. TREND CHECK: Price above key EMAs
        3. ENTRY SIGNAL: RSI oversold + other confirmations
        """

        conditions = []

        # ------------------------------------------------------------------
        # CONDITION 1: MARKET REGIME FILTER (MANDATORY)
        # ------------------------------------------------------------------
        # Only trade when BTC is bullish on 1d timeframe
        # This prevents trading during bear markets

        regime_bull = (
            dataframe['regime_bull_BTC/USDT_1d'] == 1
        )

        conditions.append(regime_bull)

        # ------------------------------------------------------------------
        # CONDITION 2: LOCAL TREND (Main Pair)
        # ------------------------------------------------------------------
        # Price should be in an uptrend on the main timeframe

        local_uptrend = (
            (dataframe['close'] > dataframe['ema_50']) &
            (dataframe['ema_50'] > dataframe['ema_100']) &
            (dataframe['adx'] > 20)  # Trend has strength
        )

        conditions.append(local_uptrend)

        # ------------------------------------------------------------------
        # CONDITION 3: ENTRY SIGNAL (RSI Oversold)
        # ------------------------------------------------------------------

        if self.buy_rsi_enabled.value:
            entry_signal = (
                (dataframe['rsi'] < self.buy_rsi_threshold.value) &
                (dataframe['volume'] > dataframe['volume_mean'] * 0.8)
            )
            conditions.append(entry_signal)

        # ------------------------------------------------------------------
        # CONDITION 4: ADDITIONAL CONFIRMATIONS
        # ------------------------------------------------------------------

        # Stochastic confirmation
        stoch_confirm = (
            (dataframe['slowk'] < 30) &
            (dataframe['slowk'] > dataframe['slowd'])  # Crossover
        )
        conditions.append(stoch_confirm)

        # ------------------------------------------------------------------
        # COMBINE ALL CONDITIONS (AND LOGIC)
        # ------------------------------------------------------------------

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    # ==========================================================================
    # EXIT LOGIC (SELL)
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define conditions for exiting a trade.

        Exit when:
        1. RSI overbought
        2. Trend reversal detected
        """

        conditions = []

        # ------------------------------------------------------------------
        # EXIT CONDITION 1: Overbought
        # ------------------------------------------------------------------

        if self.sell_rsi_enabled.value:
            overbought_exit = (
                dataframe['rsi'] > self.sell_rsi_threshold.value
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
        # COMBINE CONDITIONS (OR LOGIC)
        # ------------------------------------------------------------------

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1

        return dataframe

    # ==========================================================================
    # CUSTOM METHODS (OPTIONAL ENHANCEMENTS)
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
        Dynamic position sizing based on ATR (volatility).

        Higher volatility = smaller position (risk parity).
        """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        atr = current_candle['atr']
        close = current_candle['close']

        # Calculate volatility percentage
        volatility_pct = atr / close

        # Adjust position size based on volatility
        if volatility_pct > 0.05:  # High volatility
            adjusted_stake = proposed_stake * 0.6
        elif volatility_pct > 0.03:  # Medium volatility
            adjusted_stake = proposed_stake * 0.8
        else:  # Low volatility
            adjusted_stake = proposed_stake

        # Ensure we stay within limits
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
        Final validation before entering a trade.

        Reject trades during:
        - Low liquidity hours
        - Extreme volatility
        """

        hour = current_time.hour

        # Avoid low liquidity hours (UTC timezone)
        if hour in [0, 1, 2, 3, 4, 5]:
            logger.info(f"Entry rejected for {pair} - Low liquidity hours")
            return False

        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # Reject if ATR is too high (extreme volatility)
        atr_pct = current_candle['atr'] / current_candle['close']
        if atr_pct > 0.10:  # More than 10% ATR
            logger.info(f"Entry rejected for {pair} - Extreme volatility (ATR: {atr_pct:.2%})")
            return False

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
        Custom exit logic for emergency situations.
        """

        # Emergency exit if trade has been open too long and losing
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600

        if trade_duration_hours > 24 and current_profit < -0.02:
            return "emergency_timeout_exit"

        # Take profit on strong moves
        if current_profit > 0.08:  # 8% profit
            return "strong_profit_exit"

        return None

    # ==========================================================================
    # HYPEROPT LOSS FUNCTION (OPTIONAL CUSTOM)
    # ==========================================================================

    # You can define a custom loss function for HyperOpt here
    # For now, we use the default SharpeHyperOptLoss
