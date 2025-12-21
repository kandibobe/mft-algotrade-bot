"""
Stoic Citadel - Base Strategy Template
=======================================

Provides common functionality for all strategies:
- Standard indicator set
- Risk management integration
- Regime-aware behavior
- Logging and debugging
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""

    # Risk parameters
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_positions: int = 3
    max_drawdown: float = 0.15  # 15% max drawdown

    # Entry parameters
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    min_adx: float = 20.0
    min_volume_ratio: float = 0.8

    # Exit parameters
    stoploss: float = -0.05  # -5%
    trailing_stop_positive: float = 0.01
    trailing_stop_offset: float = 0.015

    # ROI targets
    roi_immediate: float = 0.15  # 15%
    roi_30min: float = 0.08
    roi_60min: float = 0.05
    roi_120min: float = 0.03

    # Regime adjustment
    regime_aware: bool = True
    aggressive_mode_threshold: float = 70.0
    defensive_mode_threshold: float = 40.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_per_trade": self.risk_per_trade,
            "max_positions": self.max_positions,
            "max_drawdown": self.max_drawdown,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "min_adx": self.min_adx,
            "min_volume_ratio": self.min_volume_ratio,
            "stoploss": self.stoploss,
            "trailing_stop_positive": self.trailing_stop_positive,
            "trailing_stop_offset": self.trailing_stop_offset,
            "regime_aware": self.regime_aware,
        }


class BaseStrategy(ABC):
    """
    Base class for all Stoic Citadel strategies.

    Provides:
    - Standard indicator calculation
    - Risk management
    - Regime detection
    - Logging
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self._current_regime = None
        self._regime_params = {}

    def calculate_indicators(self, df: pd.DataFrame, include_advanced: bool = True) -> pd.DataFrame:
        """
        Calculate standard indicators.

        Args:
            df: OHLCV DataFrame
            include_advanced: Include regime detection

        Returns:
            DataFrame with indicators
        """
        from src.utils.indicators import calculate_all_indicators

        result = calculate_all_indicators(df, include_advanced)

        if include_advanced and self.config.regime_aware:
            self._update_regime(result)

        return result

    def _update_regime(self, df: pd.DataFrame) -> None:
        """
        Update current market regime.
        """
        from src.utils.regime_detection import calculate_regime_score, get_regime_parameters

        if len(df) < 200:  # Need enough data
            return

        # Calculate regime score
        regime_data = calculate_regime_score(df["high"], df["low"], df["close"], df["volume"])

        current_score = regime_data["regime_score"].iloc[-1]

        # Get adjusted parameters
        self._regime_params = get_regime_parameters(
            current_score, base_risk=self.config.risk_per_trade
        )

        self._current_regime = self._regime_params.get("mode", "normal")

        logger.info(f"Regime: {self._current_regime} (score: {current_score:.1f})")

    def get_adjusted_risk(self) -> float:
        """
        Get risk adjusted for current regime.
        """
        if not self._regime_params:
            return self.config.risk_per_trade
        return self._regime_params.get("risk_per_trade", self.config.risk_per_trade)

    def get_adjusted_rsi_threshold(self) -> float:
        """
        Get RSI entry threshold adjusted for regime.
        """
        if not self._regime_params:
            return self.config.rsi_oversold
        return self._regime_params.get("min_rsi_entry", self.config.rsi_oversold)

    @abstractmethod
    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Define entry signals. Must be implemented by subclass.

        Args:
            df: DataFrame with indicators

        Returns:
            DataFrame with 'enter_long' column
        """
        pass

    @abstractmethod
    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Define exit signals. Must be implemented by subclass.

        Args:
            df: DataFrame with indicators

        Returns:
            DataFrame with 'exit_long' column
        """
        pass

    def validate_entry(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Additional validation before entry.

        Override in subclass for custom validation.
        """
        row = df.iloc[idx]

        # Volume check
        if row.get("volume", 0) < row.get("volume_sma", 1) * self.config.min_volume_ratio:
            logger.debug("Entry rejected: Low volume")
            return False

        # Volatility check (BB width)
        if row.get("bb_width", 0) < 0.02 or row.get("bb_width", 0) > 0.20:
            logger.debug("Entry rejected: Volatility filter")
            return False

        return True

    def calculate_position_size(
        self, account_balance: float, entry_price: float, stop_price: float
    ) -> float:
        """
        Calculate position size using fixed risk.
        """
        from src.utils.risk import calculate_position_size_fixed_risk

        return calculate_position_size_fixed_risk(
            account_balance=account_balance,
            risk_per_trade=self.get_adjusted_risk(),
            entry_price=entry_price,
            stop_loss_price=stop_price,
        )

    def get_roi_table(self) -> Dict[str, float]:
        """
        Get ROI table for Freqtrade.
        """
        return {
            "0": self.config.roi_immediate,
            "30": self.config.roi_30min,
            "60": self.config.roi_60min,
            "120": self.config.roi_120min,
        }

    def get_stoploss(self) -> float:
        """Get stoploss value."""
        return self.config.stoploss

    def log_trade_entry(self, pair: str, price: float, amount: float, reason: str = "") -> None:
        """Log trade entry for debugging."""
        logger.info(
            f"ENTRY: {pair} @ {price:.2f} | Amount: {amount:.4f} | "
            f"Regime: {self._current_regime} | Reason: {reason}"
        )

    def log_trade_exit(self, pair: str, price: float, profit_pct: float, reason: str = "") -> None:
        """Log trade exit for debugging."""
        emoji = "✅" if profit_pct > 0 else "❌"
        logger.info(
            f"{emoji} EXIT: {pair} @ {price:.2f} | Profit: {profit_pct:.2f}% | " f"Reason: {reason}"
        )
