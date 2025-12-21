"""
Market Regime Filter
=====================

Filters trades based on market conditions. If the market is "unclear",
the bot should NOT trade, even if ML signals "BUY".

Key insight: A model trained on trending markets will FAIL in sideways markets.
Solution: Detect market regime and only trade when conditions match the model.

Regimes:
1. BULL - Price above EMA-200, ADX > 25
2. BEAR - Price below EMA-200, ADX > 25
3. SIDEWAYS - ADX < 20 (no clear trend)
4. HIGH_VOLATILITY - Volatility spike (danger zone)

Usage:
    filter = MarketRegimeFilter(config)

    if not filter.should_trade(dataframe, side='buy'):
        return  # Skip trade
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""

    BULL = "bull"  # Strong uptrend
    BEAR = "bear"  # Strong downtrend
    SIDEWAYS = "sideways"  # No clear trend (choppy)
    HIGH_VOLATILITY = "high_volatility"  # Dangerous conditions
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class RegimeFilterConfig:
    """Configuration for market regime filter."""

    # EMA for trend direction
    ema_period: int = 200

    # ADX for trend strength
    adx_period: int = 14
    adx_trend_threshold: float = 25.0  # ADX > 25 = trending
    adx_sideways_threshold: float = 20.0  # ADX < 20 = sideways

    # Volatility thresholds
    volatility_lookback: int = 24  # Hours for volatility calc
    volatility_spike_multiplier: float = 2.0  # 2x normal vol = spike

    # Trading rules per regime
    allow_long_in_bull: bool = True
    allow_short_in_bull: bool = False  # Risky to short in bull
    allow_long_in_bear: bool = False  # Risky to long in bear
    allow_short_in_bear: bool = True
    allow_trade_in_sideways: bool = False  # Don't trade in choppy market
    allow_trade_in_high_vol: bool = False  # Don't trade during vol spikes

    # Confirmation requirements
    require_volume_confirmation: bool = True
    volume_ma_period: int = 20
    min_volume_ratio: float = 0.8  # Volume must be 80% of average

    # Safety margin (price must be X% above/below EMA)
    ema_buffer_pct: float = 0.005  # 0.5% buffer


class MarketRegimeFilter:
    """
    Filters trades based on current market regime.

    Problem: ML model trained on trends fails in sideways markets.
    Solution: Only trade when market conditions match model assumptions.

    Usage in Freqtrade:
        def confirm_trade_entry(self, pair, ..., **kwargs):
            if not self.regime_filter.should_trade(dataframe, side='buy'):
                return False
            return True
    """

    def __init__(self, config: Optional[RegimeFilterConfig] = None):
        """Initialize regime filter."""
        self.config = config or RegimeFilterConfig()
        self.regime_history: list = []
        self.blocked_trades: int = 0
        self.allowed_trades: int = 0

    def detect_regime(self, dataframe: pd.DataFrame) -> Tuple[MarketRegime, Dict[str, Any]]:
        """
        Detect current market regime.

        Args:
            dataframe: OHLCV dataframe

        Returns:
            (regime, details_dict)
        """
        if len(dataframe) < self.config.ema_period:
            return MarketRegime.UNKNOWN, {"reason": "insufficient_data"}

        details = {}

        # Calculate indicators
        close = dataframe["close"].iloc[-1]
        ema_200 = dataframe["close"].ewm(span=self.config.ema_period, adjust=False).mean().iloc[-1]
        adx = self._calculate_adx(dataframe)

        details["close"] = close
        details["ema_200"] = ema_200
        details["adx"] = adx
        details["price_vs_ema"] = (close - ema_200) / ema_200

        # Calculate volatility
        current_vol, avg_vol = self._calculate_volatility(dataframe)
        details["current_volatility"] = current_vol
        details["avg_volatility"] = avg_vol
        details["volatility_ratio"] = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Check for volatility spike
        if details["volatility_ratio"] > self.config.volatility_spike_multiplier:
            regime = MarketRegime.HIGH_VOLATILITY
            details["reason"] = f"Volatility spike: {details['volatility_ratio']:.1f}x normal"
            return regime, details

        # Check ADX for trend strength
        if adx < self.config.adx_sideways_threshold:
            regime = MarketRegime.SIDEWAYS
            details["reason"] = f"ADX {adx:.1f} < {self.config.adx_sideways_threshold} (no trend)"
            return regime, details

        # Determine trend direction
        ema_buffer = ema_200 * self.config.ema_buffer_pct

        if close > (ema_200 + ema_buffer) and adx > self.config.adx_trend_threshold:
            regime = MarketRegime.BULL
            details["reason"] = f"Price above EMA-200 + buffer, ADX {adx:.1f} (strong uptrend)"
        elif close < (ema_200 - ema_buffer) and adx > self.config.adx_trend_threshold:
            regime = MarketRegime.BEAR
            details["reason"] = f"Price below EMA-200 - buffer, ADX {adx:.1f} (strong downtrend)"
        else:
            regime = MarketRegime.SIDEWAYS
            details["reason"] = f"Price near EMA-200 or weak ADX {adx:.1f}"

        return regime, details

    def should_trade(
        self, dataframe: pd.DataFrame, side: str, entry_tag: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if trade should be allowed based on regime.

        Args:
            dataframe: OHLCV dataframe
            side: 'buy' or 'sell'
            entry_tag: Optional entry tag for logging

        Returns:
            (should_trade, reason)
        """
        regime, details = self.detect_regime(dataframe)

        # Store for analysis
        self.regime_history.append({"regime": regime.value, "side": side, **details})

        # Check volume if required
        if self.config.require_volume_confirmation:
            volume_ok, vol_reason = self._check_volume(dataframe)
            if not volume_ok:
                self.blocked_trades += 1
                return False, f"Volume filter: {vol_reason}"

        # Check regime rules
        should_trade = False
        reason = ""

        if regime == MarketRegime.BULL:
            if side == "buy" and self.config.allow_long_in_bull:
                should_trade = True
                reason = "BULL regime: LONG allowed"
            elif side == "sell" and self.config.allow_short_in_bull:
                should_trade = True
                reason = "BULL regime: SHORT allowed"
            else:
                reason = f"BULL regime: {side.upper()} blocked"

        elif regime == MarketRegime.BEAR:
            if side == "buy" and self.config.allow_long_in_bear:
                should_trade = True
                reason = "BEAR regime: LONG allowed"
            elif side == "sell" and self.config.allow_short_in_bear:
                should_trade = True
                reason = "BEAR regime: SHORT allowed"
            else:
                reason = f"BEAR regime: {side.upper()} blocked"

        elif regime == MarketRegime.SIDEWAYS:
            if self.config.allow_trade_in_sideways:
                should_trade = True
                reason = "SIDEWAYS regime: Trading allowed (risky)"
            else:
                reason = f"SIDEWAYS regime: All trades blocked (ADX={details.get('adx', 0):.1f})"

        elif regime == MarketRegime.HIGH_VOLATILITY:
            if self.config.allow_trade_in_high_vol:
                should_trade = True
                reason = "HIGH_VOL regime: Trading allowed (dangerous)"
            else:
                vol_ratio = details.get("volatility_ratio", 0)
                reason = f"HIGH_VOL regime: All trades blocked (vol={vol_ratio:.1f}x)"

        else:  # UNKNOWN
            reason = "UNKNOWN regime: Insufficient data"

        if should_trade:
            self.allowed_trades += 1
            logger.info(f"Trade ALLOWED: {side.upper()} | {reason}")
        else:
            self.blocked_trades += 1
            logger.info(f"Trade BLOCKED: {side.upper()} | {reason}")

        return should_trade, reason

    def _calculate_adx(self, dataframe: pd.DataFrame) -> float:
        """Calculate ADX (Average Directional Index)."""
        period = self.config.adx_period

        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]

        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Smoothed TR, +DM, -DM
        atr = true_range.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0

    def _calculate_volatility(self, dataframe: pd.DataFrame) -> Tuple[float, float]:
        """Calculate current and average volatility."""
        returns = dataframe["close"].pct_change().dropna()

        if len(returns) < self.config.volatility_lookback * 2:
            return 0.0, 0.0

        # Current volatility (recent period)
        current_vol = returns.tail(self.config.volatility_lookback).std()

        # Average volatility (longer period)
        avg_vol = returns.tail(self.config.volatility_lookback * 5).std()

        return current_vol, avg_vol

    def _check_volume(self, dataframe: pd.DataFrame) -> Tuple[bool, str]:
        """Check if volume confirms the move."""
        if "volume" not in dataframe.columns:
            return True, "No volume data"

        volume = dataframe["volume"].iloc[-1]
        volume_ma = dataframe["volume"].rolling(self.config.volume_ma_period).mean().iloc[-1]

        if pd.isna(volume_ma) or volume_ma <= 0:
            return True, "Insufficient volume history"

        volume_ratio = volume / volume_ma

        if volume_ratio < self.config.min_volume_ratio:
            return False, f"Low volume: {volume_ratio:.1%} of average"

        return True, f"Volume OK: {volume_ratio:.1%} of average"

    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics."""
        total = self.allowed_trades + self.blocked_trades

        if self.regime_history:
            regime_counts = {}
            for entry in self.regime_history:
                regime = entry["regime"]
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
        else:
            regime_counts = {}

        return {
            "total_checks": total,
            "allowed": self.allowed_trades,
            "blocked": self.blocked_trades,
            "block_rate": self.blocked_trades / total if total > 0 else 0,
            "regime_distribution": regime_counts,
        }

    def get_current_regime_summary(self, dataframe: pd.DataFrame) -> str:
        """Get human-readable regime summary."""
        regime, details = self.detect_regime(dataframe)

        summary = f"""
╔══════════════════════════════════════════════════════════╗
║               MARKET REGIME ANALYSIS                     ║
╠══════════════════════════════════════════════════════════╣
║ Regime: {regime.value.upper():^48} ║
╠══════════════════════════════════════════════════════════╣
║ Close:     {details.get('close', 0):>15.2f}                             ║
║ EMA-200:   {details.get('ema_200', 0):>15.2f}                             ║
║ Price/EMA: {details.get('price_vs_ema', 0)*100:>14.2f}%                             ║
║ ADX:       {details.get('adx', 0):>15.1f}                             ║
║ Vol Ratio: {details.get('volatility_ratio', 0):>15.2f}x                            ║
╠══════════════════════════════════════════════════════════╣
║ Reason: {details.get('reason', 'N/A'):<48} ║
╚══════════════════════════════════════════════════════════╝
"""
        return summary


def create_freqtrade_confirm_entry(config: Optional[RegimeFilterConfig] = None):
    """
    Create a confirm_trade_entry function for Freqtrade.

    Usage in strategy:
        from src.strategies.market_regime import create_freqtrade_confirm_entry

        class MyStrategy(IStrategy):
            confirm_trade_entry = create_freqtrade_confirm_entry()
    """
    regime_filter = MarketRegimeFilter(config)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """
        Freqtrade confirm_trade_entry callback.

        Checks market regime before allowing trade.
        """
        try:
            # Get dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

            if dataframe.empty or len(dataframe) < 200:
                logger.warning(f"Insufficient data for {pair}, allowing trade")
                return True

            # Check regime
            should_trade, reason = regime_filter.should_trade(
                dataframe=dataframe, side=side, entry_tag=entry_tag
            )

            if not should_trade:
                logger.info(f"BLOCKED {side.upper()} on {pair}: {reason}")

            return should_trade

        except Exception as e:
            logger.error(f"Error in confirm_trade_entry: {e}")
            return True  # Allow on error (fail-open)

    return confirm_trade_entry
