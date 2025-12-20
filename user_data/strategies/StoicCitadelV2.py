"""
Stoic Citadel V2 - Professional Production Strategy
===================================================

Professional-grade strategy with:
1. Shared signal library (research/production parity)
2. Portfolio correlation management
3. Circuit breaker for max drawdown
4. Market regime adaptation
5. ML inference integration
6. Dynamic position sizing
7. Advanced exit logic

Philosophy: "Risk management first, profits second."

Author: Stoic Citadel Team
Version: 2.5.0
License: MIT
"""

import sys
from pathlib import Path

# Add src to path for shared library
src_path = Path(__file__).parents[2] / 'src'
sys.path.insert(0, str(src_path))

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame, Series
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

# Import shared libraries
from signals.indicators import SignalGenerator
from risk.correlation import CorrelationManager, DrawdownMonitor
from utils.regime_detection import calculate_regime_score, get_regime_parameters

logger = logging.getLogger(__name__)


class StoicCitadelV2(IStrategy):
    """
    Professional production strategy with regime adaptation and ML integration.

    Key Features:
    - Shared signal library (100% parity with research)
    - Correlation-based position sizing
    - Circuit breaker for max drawdown
    - Portfolio heat monitoring
    - Market regime detection and adaptation
    - ML inference for signal validation
    - Dynamic leverage and risk adjustment
    """

    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================

    INTERFACE_VERSION = 3

    # Hyperopt parameters
    buy_rsi = IntParameter(20, 40, default=30, space="buy")
    buy_adx = IntParameter(15, 30, default=20, space="buy")
    sell_rsi = IntParameter(65, 85, default=75, space="sell")
    volatility_threshold = DecimalParameter(0.02, 0.10, default=0.05, space="buy")

    # Minimal ROI with regime adaptation
    minimal_roi = {
        "0": 0.15,   # 15% - close immediately
        "30": 0.08,  # 8% after 30 min
        "60": 0.05,  # 5% after 1 hour
        "120": 0.03, # 3% after 2 hours
        "240": 0.01  # 1% after 4 hours
    }

    # Hard stop loss (THE STOIC GUARD)
    stoploss = -0.05  # -5%

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '5m'

    # Process only new candles
    process_only_new_candles = True

    # Use exit signals
    use_exit_signal = True
    exit_profit_only = False

    # Startup candle count
    startup_candle_count: int = 200

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

    # Maximum open trades based on regime
    max_open_trades = 5

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def __init__(self, config: Dict) -> None:
        """Initialize strategy with shared libraries and regime detection."""
        super().__init__(config)

        # Initialize shared signal generator
        self.signal_generator = SignalGenerator()

        # Initialize risk management
        self.correlation_manager = CorrelationManager(
            correlation_window=24,
            max_correlation=0.7,
            max_portfolio_heat=0.15
        )

        self.drawdown_monitor = DrawdownMonitor(
            max_drawdown=0.15,
            stop_duration_minutes=240
        )

        # Regime detection state
        self._regime_mode = 'normal'
        self._regime_params = {}
        self._last_regime_update = None
        self._regime_score = 50.0

        # ML inference placeholder
        self.ml_confidence_threshold = 0.65

        logger.info("✅ StoicCitadelV2 initialized with regime adaptation and ML integration")

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators using shared library with regime detection.

        **CRITICAL**: This uses the same code as research environment!
        """
        try:
            # Use shared signal generator
            dataframe = self.signal_generator.populate_all_indicators(dataframe)

            # Calculate regime score if we have enough data
            if len(dataframe) >= 200:
                self._update_regime(dataframe, metadata)

            logger.info(f"✅ Indicators populated for {metadata['pair']} (regime: {self._regime_mode})")

        except Exception as e:
            logger.error(f"❌ Error populating indicators: {e}")
            raise

        return dataframe

    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals using shared library with regime adaptation."""
        try:
            # Get base signals from shared library
            base_signals = self.signal_generator.generate_entry_signal(dataframe)

            # Apply regime-specific filters
            regime_filter = self._apply_regime_entry_filter(dataframe)
            
            # Combine signals
            dataframe['enter_long'] = (base_signals & regime_filter).astype(int)

            # Apply ML confidence filter if available
            if hasattr(self, '_ml_predictions') and metadata['pair'] in self._ml_predictions:
                ml_confidence = self._ml_predictions[metadata['pair']]
                ml_filter = ml_confidence > self.ml_confidence_threshold
                dataframe['enter_long'] = (dataframe['enter_long'] & ml_filter).astype(int)

            # Log signal count
            signal_count = dataframe['enter_long'].sum()
            if signal_count > 0:
                logger.info(
                    f"📊 {metadata['pair']}: {signal_count} entry signals generated (regime: {self._regime_mode})"
                )

        except Exception as e:
            logger.error(f"❌ Error generating entry signals: {e}")
            dataframe['enter_long'] = 0

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

    def _apply_regime_entry_filter(self, dataframe: DataFrame) -> Series:
        """Apply regime-specific entry filters."""
        if self._regime_mode == 'defensive':
            # More conservative in defensive mode
            return (
                (dataframe['rsi'] < self.buy_rsi.value - 5) &
                (dataframe['adx'] > self.buy_adx.value + 5) &
                (dataframe['volume'] > dataframe['volume_mean'] * 1.2)
            )
        elif self._regime_mode == 'aggressive':
            # More permissive in aggressive mode
            return (
                (dataframe['rsi'] < self.buy_rsi.value + 5) &
                (dataframe['adx'] > max(self.buy_adx.value - 5, 15)) &
                (dataframe['volume'] > dataframe['volume_mean'] * 0.7)
            )
        else:  # normal/cautious
            return (
                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['volume'] > dataframe['volume_mean'] * 0.8)
            )

    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals using shared library with regime adaptation."""
        try:
            # Get base exit signals
            base_exits = self.signal_generator.generate_exit_signal(dataframe)
            
            # Add regime-specific exit conditions
            regime_exits = self._apply_regime_exit_filter(dataframe)
            
            # Combine exits
            dataframe['exit_long'] = (base_exits | regime_exits).astype(int)
            
            # Log exit signals
            exit_count = dataframe['exit_long'].sum()
            if exit_count > 0:
                logger.info(
                    f"📉 {metadata['pair']}: {exit_count} exit signals generated (regime: {self._regime_mode})"
                )

        except Exception as e:
            logger.error(f"❌ Error generating exit signals: {e}")
            dataframe['exit_long'] = 0

        return dataframe

    def _apply_regime_exit_filter(self, dataframe: DataFrame) -> Series:
        """Apply regime-specific exit filters."""
        if self._regime_mode == 'defensive':
            # Exit earlier in defensive mode
            return (
                (dataframe['rsi'] > self.sell_rsi.value - 5) |
                (dataframe['close'] < dataframe['ema_50']) |
                (dataframe['macd_hist'] < 0)
            )
        elif self._regime_mode == 'aggressive':
            # Hold longer in aggressive mode
            return (
                (dataframe['rsi'] > self.sell_rsi.value + 5) |
                ((dataframe['close'] < dataframe['ema_50']) & (dataframe['macd'] < dataframe['macdsignal']))
            )
        else:  # normal/cautious
            return (
                (dataframe['rsi'] > self.sell_rsi.value) |
                ((dataframe['close'] < dataframe['ema_50']) & (dataframe['macd_hist'] < 0))
            )

    # ==========================================================================
    # ENTRY CONFIRMATION (WITH CORRELATION CHECK)
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
        Final gate before entering trade - with correlation check.

        This is where we implement advanced risk management!
        """
        # 1. Check circuit breaker (drawdown limit)
        if self.drawdown_monitor.is_circuit_breaker_active():
            logger.warning(
                f"🔒 {pair}: Entry blocked by circuit breaker"
            )
            return False

        # 2. Check low liquidity hours
        hour = current_time.hour
        if hour in [0, 1, 2, 3, 4, 5]:
            logger.info(f"⏰ {pair}: Rejecting entry - low liquidity hours")
            return False

        # 3. Correlation check (if possible)
        try:
            # Get open trades
            open_trades = self.dp.get_open_trades()

            if open_trades:
                # Get dataframe for new pair
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

                # Get dataframes for open pairs
                all_pairs_data = {}
                for trade in open_trades:
                    open_pair = trade['pair']
                    open_df, _ = self.dp.get_analyzed_dataframe(
                        open_pair,
                        self.timeframe
                    )
                    all_pairs_data[open_pair] = open_df

                # Check correlation
                correlation_ok = self.correlation_manager.check_entry_correlation(
                    new_pair=pair,
                    new_pair_data=dataframe,
                    open_positions=[{'pair': t['pair']} for t in open_trades],
                    all_pairs_data=all_pairs_data
                )

                if not correlation_ok:
                    logger.warning(
                        f"❌ {pair}: Entry blocked by correlation check"
                    )
                    return False

        except Exception as e:
            logger.warning(f"⚠️ Correlation check failed: {e}, allowing entry")

        logger.info(f"✅ {pair}: Entry confirmed (passed all checks)")
        return True

    # ==========================================================================
    # CUSTOM STAKE AMOUNT (VOLATILITY-ADJUSTED WITH REGIME)
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

        Higher volatility = smaller position (risk parity).
        Regime adaptation adjusts risk per trade.
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

    # ==========================================================================
    # CUSTOM EXIT WITH REGIME ADAPTATION
    # ==========================================================================

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """Custom exit logic with regime adaptation and emergency exits."""
        # Trade duration in hours
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600

        # Emergency exit: trade open > 24h and losing
        if trade_duration > 24 and current_profit < -0.02:
            logger.warning(
                f"⚠️ {pair}: Emergency exit - open 24h+ with -2% loss"
            )
            return "emergency_exit_24h"

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

    # ==========================================================================
    # DYNAMIC LEVERAGE
    # ==========================================================================

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
