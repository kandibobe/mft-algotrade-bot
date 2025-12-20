"""
Stoic Strategy V1 - Professional Core Strategy
===============================================

Core professional trading strategy implementing:
1. Multi-timeframe analysis (5m execution, 1h trend)
2. Advanced risk management with adaptive position sizing
3. Machine learning signal validation (placeholder)
4. Market regime detection and adaptation
5. Portfolio correlation management

Philosophy: "Trade with discipline, adapt with wisdom."

Author: Stoic Citadel Team
Version: 1.0.0
License: MIT
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame, Series
import pandas as pd
import numpy as np

# Add src to path for shared libraries
src_path = Path(__file__).parents[2] / 'src'
sys.path.insert(0, str(src_path))

try:
    from utils.regime_detection import calculate_regime_score, get_regime_parameters
    from risk.correlation import CorrelationManager, DrawdownMonitor
    from signals.indicators import SignalGenerator
    USE_SHARED_LIBS = True
except ImportError:
    USE_SHARED_LIBS = False

import talib.abstract as ta

logger = logging.getLogger(__name__)


class StoicStrategyV1(IStrategy):
    """
    Professional core strategy with multi-timeframe analysis and ML integration.
    
    Key Features:
    -------------
    1. MULTI-TIMEFRAME: 5m for execution, 1h for trend confirmation
    2. ADVANCED RISK: Adaptive position sizing, correlation management
    3. ML READY: Architecture for machine learning signal validation
    4. REGIME ADAPTIVE: Adjusts to market conditions
    5. PORTFOLIO AWARE: Manages correlation and portfolio heat
    
    Trading Logic:
    --------------
    - Entry: Price > EMA200 (bull trend) + RSI oversold + Volume confirmation
    - Exit: RSI overbought OR trend reversal OR time-based exit
    - Risk: -5% hard stop, trailing stop at +1%
    """

    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================

    INTERFACE_VERSION = 3

    # Hyperopt parameters
    buy_rsi = IntParameter(25, 40, default=30, space="buy")
    buy_adx = IntParameter(15, 30, default=20, space="buy")
    sell_rsi = IntParameter(65, 85, default=75, space="sell")
    volatility_threshold = DecimalParameter(0.02, 0.10, default=0.05, space="buy")
    max_trade_duration = IntParameter(12, 48, default=24, space="sell")  # hours

    # Minimal ROI with progressive take-profit
    minimal_roi = {
        "0": 0.15,   # 15% - immediate take profit
        "30": 0.08,  # 8% after 30 minutes
        "60": 0.05,  # 5% after 1 hour
        "120": 0.03, # 3% after 2 hours
        "240": 0.01  # 1% after 4 hours
    }

    # Stoploss - HARD LIMIT
    stoploss = -0.05  # -5% maximum loss per trade

    # Trailing stop - Lock in profits
    trailing_stop = True
    trailing_stop_positive = 0.01      # Start trailing at +1%
    trailing_stop_positive_offset = 0.015  # Trail by 1.5%
    trailing_only_offset_is_reached = True

    # Timeframes
    timeframe = '5m'  # Execution timeframe
    informative_timeframe = '1h'  # Trend confirmation timeframe

    # Process only new candles
    process_only_new_candles = True

    # Use exit signals
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200
    informative_startup_candle_count: int = 50

    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60
    }

    order_time_in_force = {
        'entry': 'GTC',  # Good Till Cancelled
        'exit': 'GTC'
    }

    # Maximum open trades based on regime
    max_open_trades = 5

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def __init__(self, config: Dict) -> None:
        """Initialize strategy with shared libraries and state management."""
        super().__init__(config)

        # Shared libraries
        if USE_SHARED_LIBS:
            self.signal_generator = SignalGenerator()
            self.correlation_manager = CorrelationManager(
                correlation_window=24,
                max_correlation=0.7,
                max_portfolio_heat=0.15
            )
            self.drawdown_monitor = DrawdownMonitor(
                max_drawdown=0.15,
                stop_duration_minutes=240
            )
        else:
            self.signal_generator = None
            self.correlation_manager = None
            self.drawdown_monitor = None

        # Regime detection state
        self._regime_mode = 'normal'
        self._regime_params = {}
        self._last_regime_update = None
        self._regime_score = 50.0

        # ML inference placeholder
        self.ml_confidence_threshold = 0.65
        self._ml_predictions = {}

        # Performance tracking
        self._trade_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0
        }

        logger.info("✅ StoicStrategyV1 initialized with professional features")

    # ==========================================================================
    # PROTECTIONS
    # ==========================================================================

    @property
    def protections(self):
        """Protection mechanisms with regime adaptation."""
        base_protections = [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 60,
                "trade_limit": 3,
                "stop_duration_candles": 24,
                "required_profit": 0.0
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 360,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": -0.05
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,
                "trade_limit": 5,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.15
            }
        ]

        # Add regime-specific protections
        if self._regime_mode == 'defensive':
            base_protections.append({
                "method": "CooldownPeriod",
                "stop_duration_candles": 10
            })

        return base_protections

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================

    def informative_pairs(self):
        """Define additional informative pairs for multi-timeframe analysis."""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators for both execution and informative timeframes.
        """
        # Get informative dataframe (1h for trend)
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe=self.informative_timeframe
        )
        
        # Calculate indicators on informative timeframe
        informative = self._calculate_informative_indicators(informative)
        
        # Merge informative indicators into execution dataframe
        dataframe = self._merge_informative_data(dataframe, informative)
        
        # Calculate execution indicators
        dataframe = self._calculate_execution_indicators(dataframe)
        
        # Update regime if we have enough data
        if len(dataframe) >= 200 and USE_SHARED_LIBS:
            self._update_regime(dataframe, metadata)
        
        return dataframe

    def _calculate_informative_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate indicators on informative timeframe."""
        # Trend indicators
        dataframe['ema_50_1h'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200_1h'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['adx_1h'] = ta.ADX(dataframe, timeperiod=14)
        
        # Support/Resistance
        dataframe['resistance_1h'] = dataframe['high'].rolling(20).max()
        dataframe['support_1h'] = dataframe['low'].rolling(20).min()
        
        # Volume analysis
        dataframe['volume_mean_1h'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio_1h'] = dataframe['volume'] / dataframe['volume_mean_1h']
        
        return dataframe

    def _merge_informative_data(self, dataframe: DataFrame, informative: DataFrame) -> DataFrame:
        """Merge informative indicators into execution dataframe."""
        # Resample informative to match execution timeframe
        informative_resampled = informative.resample('5min').ffill()
        
        # Merge key indicators
        merge_columns = ['ema_50_1h', 'ema_200_1h', 'adx_1h', 'resistance_1h', 
                        'support_1h', 'volume_ratio_1h']
        
        for col in merge_columns:
            if col in informative_resampled.columns:
                dataframe[col] = informative_resampled[col]
        
        return dataframe

    def _calculate_execution_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate indicators on execution timeframe."""
        # Use shared library if available
        if self.signal_generator:
            dataframe = self.signal_generator.populate_all_indicators(dataframe)
        else:
            # Fallback to TA-Lib
            dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
            dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
            
            # MACD
            macd = ta.MACD(dataframe)
            dataframe['macd'] = macd['macd']
            dataframe['macdsignal'] = macd['macdsignal']
            dataframe['macdhist'] = macd['macdhist']
            
            # Bollinger Bands
            bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
            dataframe['bb_upper'] = bollinger['upperband']
            dataframe['bb_middle'] = bollinger['middleband']
            dataframe['bb_lower'] = bollinger['lowerband']
            dataframe['bb_width'] = (
                (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
            )
            
            # ATR for volatility
            dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
            dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
            
            # Volume
            dataframe['volume_mean'] = dataframe['volume'].rolling(20).mean()
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Multi-timeframe trend alignment
        dataframe['trend_aligned'] = (
            (dataframe['close'] > dataframe['ema_200']) &
            (dataframe['close'] > dataframe['ema_200_1h'])
        ).astype(int)
        
        # Trend strength composite
        dataframe['trend_strength'] = (
            (dataframe['adx'] > 20).astype(int) +
            (dataframe['adx_1h'] > 20).astype(int) +
            dataframe['trend_aligned']
        )
        
        return dataframe

    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions with multi-timeframe confirmation.
        """
        conditions = []
        
        # 1. Multi-timeframe trend filter
        trend_condition = (
            (dataframe['close'] > dataframe['ema_200']) &  # 5m trend
            (dataframe['close'] > dataframe['ema_200_1h']) &  # 1h trend
            (dataframe['trend_strength'] >= 2)  # At least 2/3 trend signals
        )
        conditions.append(trend_condition)
        
        # 2. Entry oscillator (RSI oversold)
        entry_condition = (
            (dataframe['rsi'] < self.buy_rsi.value) &
            (dataframe['close'] < dataframe['bb_lower'] * 1.02)  # Near lower BB
        )
        conditions.append(entry_condition)
        
        # 3. Volume confirmation
        volume_condition = (
            (dataframe['volume_ratio'] > 0.8) &
            (dataframe['volume_ratio_1h'] > 0.7)
        )
        conditions.append(volume_condition)
        
        # 4. Volatility filter
        volatility_condition = (
            (dataframe['bb_width'] > self.volatility_threshold.value) &
            (dataframe['bb_width'] < 0.15)
        )
        conditions.append(volatility_condition)
        
        # 5. Regime-specific adjustments
        if self._regime_mode == 'defensive':
            # More conservative in defensive mode
            defensive_condition = (
                (dataframe['adx'] > self.buy_adx.value + 5) &
                (dataframe['rsi'] < self.buy_rsi.value - 5)
            )
            conditions.append(defensive_condition)
        elif self._regime_mode == 'aggressive':
            # More permissive in aggressive mode
            aggressive_condition = (
                (dataframe['adx'] > max(self.buy_adx.value - 5, 15))
            )
            conditions.append(aggressive_condition)
        
        # Combine all conditions
        if conditions:
            dataframe.loc[
                pd.Series(np.logical_and.reduce(conditions)),
                'enter_long'
            ] = 1
        
        # Log entry signals
        signal_count = dataframe['enter_long'].sum()
        if signal_count > 0:
            logger.info(
                f"📊 {metadata['pair']}: {signal_count} entry signals "
                f"(regime: {self._regime_mode}, trend strength: {dataframe['trend_strength'].iloc[-1]}/3)"
            )
        
        return dataframe

    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions with multi-timeframe confirmation.
        """
        conditions = []
        
        # 1. Overbought exit
        overbought_exit = (
            (dataframe['rsi'] > self.sell_rsi.value) &
            (dataframe['close'] > dataframe['bb_upper'] * 0.98)  # Near upper BB
        )
        conditions.append(overbought_exit)
        
        # 2. Trend reversal
        trend_reversal = (
            (dataframe['close'] < dataframe['ema_50']) &
            (dataframe['macd_hist'] < 0) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))
        )
        conditions.append(trend_reversal)
        
        # 3. Multi-timeframe trend breakdown
        trend_breakdown = (
            (dataframe['close'] < dataframe['ema_200']) |
            (dataframe['close'] < dataframe['ema_200_1h'])
        )
        conditions.append(trend_breakdown)
        
        # 4. Regime-specific exits
        if self._regime_mode == 'defensive':
            # Exit earlier in defensive mode
            defensive_exit = (
                (dataframe['rsi'] > self.sell_rsi.value - 5) |
                (dataframe['adx'] < 15)
            )
            conditions.append(defensive_exit)
        elif self._regime_mode == 'aggressive':
            # Hold longer in aggressive mode
            aggressive_exit = (
                (dataframe['rsi'] > self.sell_rsi.value + 5) |
                ((dataframe['close'] < dataframe['ema_50']) & (dataframe['macd'] < dataframe['macdsignal']))
            )
            conditions.append(aggressive_exit)
        
        # Combine with OR logic
        if conditions:
            dataframe.loc[
                pd.Series(np.logical_or.reduce(conditions)),
                'exit_long'
            ] = 1
        
        # Log exit signals
        exit_count = dataframe['exit_long'].sum()
        if exit_count > 0:
            logger.info(
                f"📉 {metadata['pair']}: {exit_count} exit signals "
                f"(regime: {self._regime_mode})"
            )
        
        return dataframe

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _update_regime(self, dataframe: DataFrame, metadata: dict) -> None:
        """Update market regime based on current data."""
        try:
            regime_data = calculate_regime_score(
                dataframe['high'],
                dataframe['low'],
                dataframe['close'],
                dataframe['volume']
            )
            
            current_score = regime_data['regime_score'].iloc[-1]
            self._regime_score = float(current_score)
            
            self._regime_params = get_regime_parameters(
                current_score,
                base_risk=0.02
            )
            
            self._regime_mode = self._regime_params.get('mode', 'normal')
            self._last_regime_update = datetime.now()
            
            logger.info(
                f"📈 Regime updated: {self._regime_mode} "
                f"(score: {current_score:.1f}) for {metadata['pair']}"
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Regime detection failed: {e}")
            self._regime_mode = 'normal'

    # ==========================================================================
    # CUSTOM METHODS
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
        Dynamic position sizing based on volatility and regime.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return proposed_stake
                
            current_candle = dataframe.iloc[-1].squeeze()

            # Get ATR (volatility measure)
            atr = current_candle.get('atr', 0)
            close = current_candle.get('close', current_rate)

            # Calculate volatility percentage
            volatility_pct = atr / close if close > 0 else 0.02

            # Base volatility adjustment
            if volatility_pct > self.volatility_threshold.value:  # High volatility
                vol_factor = 0.5
                vol_msg = "High volatility, reducing stake by 50%"
            elif volatility_pct > self.volatility_threshold.value * 0.6:  # Medium volatility
                vol_factor = 0.75
                vol_msg = "Medium volatility, reducing stake by 25%"
            else:  # Low volatility
                vol_factor = 1.0
                vol_msg = "Low volatility, full stake"

            # Regime adjustment
            regime_factor = self._regime_params.get('risk_per_trade', 0.02) / 0.02

            # Calculate adjusted stake
            adjusted_stake = proposed_stake * vol_factor * regime_factor

            # Apply bounds
            adjusted_stake = max(min_stake or 0, min(adjusted_stake, max_stake))

            logger.info(
                f"💰 {pair}: {vol_msg} | "
                f"Regime factor: {regime_factor:.2f} | "
                f"Final stake: {adjusted_stake:.2f}"
            )

            return adjusted_stake

        except Exception as e:
            logger.error(f"❌ Error calculating stake: {e}")
            return proposed_stake

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
        Final gate before entering a trade with correlation check.
        """
        # 1. Check circuit breaker if available
        if self.drawdown_monitor and self.drawdown_monitor.is_circuit_breaker_active():
            logger.warning(f"🔒 {pair}: Entry blocked by circuit breaker")
            return False

        # 2. Check low liquidity hours
        hour = current_time.hour
        if hour in [0, 1, 2, 3, 4, 5]:
            logger.info(f"⏰ {pair}: Rejecting entry - low liquidity hours")
            return False

        # 3. Correlation check if manager available
        if self.correlation_manager:
            try:
                open_trades = self.dp.get_open_trades()
                if open_trades:
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    
                    all_pairs_data = {}
                    for trade in open_trades:
                        open_pair = trade['pair']
                        open_df, _ = self.dp.get_analyzed_dataframe(
                            open_pair,
                            self.timeframe
                        )
                        all_pairs_data[open_pair] = open_df

                    correlation_ok = self.correlation_manager.check_entry_correlation(
                        new_pair=pair,
                        new_pair_data=dataframe,
                        open_positions=[{'pair': t['pair']} for t in open_trades],
                        all_pairs_data=all_pairs_data
                    )

                    if not correlation_ok:
                        logger.warning(f"❌ {pair}: Entry blocked by correlation check")
                        return False
            except Exception as e:
                logger.warning(f"⚠️ Correlation check failed: {e}, allowing entry")

        # 4. Regime-specific checks
        if self._regime_mode == 'defensive':
            # In defensive mode, require stronger signals
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current = dataframe.iloc[-1]
                if current.get('trend_strength', 0) < 3:  # Require max trend strength
                    logger.info(f"⚠️ {pair}: Weak trend in defensive mode")
                    return False

        logger.info(f"✅ {pair}: Entry confirmed (passed all checks)")
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
        Custom exit logic with regime adaptation and emergency exits.
        """
        # Trade duration in hours
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Emergency exit: trade open > max duration and losing
        if trade_duration > self.max_trade_duration.value and current_profit < -0.02:
            logger.warning(f"⚠️ {pair}: Emergency exit - open {self.max_trade_duration.value}h+ with -2% loss")
            return f"emergency_exit_{self.max_trade_duration.value}h"

        # Regime-specific exits
        if self._regime_mode == 'defensive':
            # Exit earlier in defensive mode
            if trade_duration > 12 and current_profit > 0.03:
                logger.info(f"💰 {pair}: Early take profit at +3% in defensive mode")
                return "early_take_profit_defensive"
                
            if trade_duration > 6 and current_profit < -0.01:
                logger.info(f"📉 {pair}: Quick stop loss at -1% in defensive mode")
                return "quick_stop_defensive"

        elif self._regime_mode == 'aggressive':
            # Hold longer in aggressive mode
            if current_profit > 0.15:
                logger.info(f"💰 {pair}: Take profit at +15% in aggressive mode")
                return "take_profit_aggressive"
                
            if trade_duration > 36 and current_profit < -0.03:
                logger.warning(f"⚠️ {pair}: Extended emergency exit in aggressive mode")
                return "extended_emergency_aggressive"

        else:  # normal/cautious
            # Standard exits
            if current_profit > 0.10:
                logger.info(f"💰 {pair}: Take profit at +10%")
                return "take_profit_10pct"
                
            if trade_duration > 18 and 0 < current_profit < 0.02:
                logger.info(f"⏰ {pair}: Time-based exit with small profit")
                return "time_exit_small_profit"

        # Protect profits: if was up 5%+ and now dropping
        if hasattr(trade, 'max_rate') and trade.max_rate:
            max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
            if max_profit > 0.05 and current_profit < max_profit * 0.5:
                logger.info(f"🛡️ {pair}: Profit protection exit")
                return "profit_protection"

        return None

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
        Dynamic leverage based on regime and volatility.
        """
        # Get current volatility
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current = dataframe.iloc[-1]
                atr_pct = current.get('atr_pct', 2.0)
                
                # Reduce leverage in high volatility
                if atr_pct > 5:
                    vol_factor = 0.5
                elif atr_pct > 3:
                    vol_factor = 0.75
                else:
                    vol_factor = 1.0
            else:
                vol_factor = 1.0
        except:
            vol_factor = 1.0

        # Regime-based leverage
        if self._regime_mode == 'defensive':
            regime_leverage = min(1.0, max_leverage)
        elif self._regime_mode == 'aggressive':
            regime_leverage = min(2.0, max_leverage)
        else:
            regime_leverage = min(1.5, max_leverage)

        # Apply both adjustments
        final_leverage = min(regime_leverage * vol_factor, max_leverage)
        
        logger.info(
            f"⚖️ {pair}: Leverage - Regime: {regime_leverage:.1f}x, "
            f"Vol factor: {vol_factor:.2f}, Final: {final_leverage:.1f}x"
        )
        
        return final_leverage

    def update_trade_stats(self, trade, profit: float) -> None:
        """Update trade statistics."""
        self._trade_stats['total_trades'] += 1
        self._trade_stats['total_profit'] += profit
        
        if profit > 0:
            self._trade_stats['winning_trades'] += 1
        else:
            self._trade_stats['losing_trades'] += 1
        
        win_rate = (self._trade_stats['winning_trades'] / self._trade_stats['total_trades'] * 100 
                   if self._trade_stats['total_trades'] > 0 else 0)
        
        logger.info(
            f"📊 Trade Stats: {self._trade_stats['winning_trades']}/"
            f"{self._trade_stats['total_trades']} wins "
            f"({win_rate:.1f}%), Total P&L: {self._trade_stats['total_profit']:.2%}"
        )
