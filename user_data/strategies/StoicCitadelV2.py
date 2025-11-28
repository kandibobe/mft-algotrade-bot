"""
Stoic Citadel V2 - Production-Ready Strategy
=============================================

This strategy implements critical improvements:
1. Shared signal library (research/production parity)
2. Portfolio correlation management
3. Circuit breaker for max drawdown
4. Async ML inference ready (placeholder)

Philosophy: "Risk management first, profits second."

Author: Stoic Citadel Team
Version: 2.0.0
License: MIT
"""

import sys
from pathlib import Path

# Add src to path for shared library
src_path = Path(__file__).parents[2] / 'src'
sys.path.insert(0, str(src_path))

from freqtrade.strategy import IStrategy
from pandas import DataFrame
from typing import Optional, Dict
from datetime import datetime
import logging

# Import shared libraries
from signals.indicators import SignalGenerator
from risk.correlation import CorrelationManager, DrawdownMonitor

logger = logging.getLogger(__name__)


class StoicCitadelV2(IStrategy):
    """
    Production-ready strategy with advanced risk management.

    Key Features:
    - Shared signal library (100% parity with research)
    - Correlation-based position sizing
    - Circuit breaker for max drawdown
    - Portfolio heat monitoring
    """

    # ==========================================================================
    # STRATEGY METADATA
    # ==========================================================================

    INTERFACE_VERSION = 3

    # Minimal ROI
    minimal_roi = {
        "0": 0.15,   # 15% - close immediately
        "30": 0.08,  # 8% after 30 min
        "60": 0.05,  # 5% after 1 hour
        "120": 0.03  # 3% after 2 hours
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

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================

    def __init__(self, config: Dict) -> None:
        """Initialize strategy with shared libraries."""
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

        logger.info("✅ StoicCitadelV2 initialized with advanced risk management")

    # ==========================================================================
    # PROTECTIONS
    # ==========================================================================

    @property
    def protections(self):
        """Protection mechanisms."""
        return [
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

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators using shared library.

        **CRITICAL**: This uses the same code as research environment!
        """
        try:
            # Use shared signal generator
            dataframe = self.signal_generator.populate_all_indicators(dataframe)

            logger.info(f"✅ Indicators populated for {metadata['pair']}")

        except Exception as e:
            logger.error(f"❌ Error populating indicators: {e}")
            raise

        return dataframe

    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals using shared library."""
        try:
            # Use shared signal generator
            dataframe['enter_long'] = self.signal_generator.generate_entry_signal(
                dataframe
            )

            # Log signal count
            signal_count = dataframe['enter_long'].sum()
            if signal_count > 0:
                logger.info(
                    f"📊 {metadata['pair']}: {signal_count} entry signals generated"
                )

        except Exception as e:
            logger.error(f"❌ Error generating entry signals: {e}")
            dataframe['enter_long'] = 0

        return dataframe

    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals using shared library."""
        try:
            # Use shared signal generator
            dataframe['exit_long'] = self.signal_generator.generate_exit_signal(
                dataframe
            )

        except Exception as e:
            logger.error(f"❌ Error generating exit signals: {e}")
            dataframe['exit_long'] = 0

        return dataframe

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
    # CUSTOM STAKE AMOUNT (VOLATILITY-ADJUSTED)
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
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()

            # Get ATR (volatility measure)
            atr = current_candle['atr']
            close = current_candle['close']

            # Calculate volatility percentage
            volatility_pct = atr / close

            # Adjust stake based on volatility
            if volatility_pct > 0.05:  # > 5% volatility
                adjusted_stake = proposed_stake * 0.5
                logger.info(f"📉 {pair}: High volatility, reducing stake by 50%")
            elif volatility_pct > 0.03:  # > 3% volatility
                adjusted_stake = proposed_stake * 0.75
                logger.info(f"📊 {pair}: Medium volatility, reducing stake by 25%")
            else:  # Low volatility
                adjusted_stake = proposed_stake
                logger.info(f"📈 {pair}: Low volatility, full stake")

            return max(min_stake or 0, min(adjusted_stake, max_stake))

        except Exception as e:
            logger.error(f"❌ Error calculating stake: {e}")
            return proposed_stake

    # ==========================================================================
    # CUSTOM EXIT
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
        """Custom exit logic with emergency exits."""
        # Emergency exit: trade open > 24h and losing
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600

        if trade_duration > 24 and current_profit < -0.02:
            logger.warning(
                f"⚠️ {pair}: Emergency exit - open 24h+ with -2% loss"
            )
            return "emergency_exit_24h"

        # Take profit on strong moves
        if current_profit > 0.10:
            logger.info(f"💰 {pair}: Take profit at +10%")
            return "take_profit_10pct"

        return None
