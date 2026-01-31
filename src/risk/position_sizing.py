"""
Stoic Citadel - Advanced Position Sizing
=========================================

This module implements institutional-grade position sizing logic.

Financial Logic:
----------------
Position sizing is the primary determinant of long-term survival and profitability.
- **Fixed Risk**: Risk a fixed % of account per trade (e.g., 1%).
- **Volatility Adjusted**: Reduce size when asset volatility is high to maintain constant risk.
- **VaR (Value at Risk)**: Limit size so that the 95% worst-case daily loss is within limits.
- **Kelly Criterion**: Mathematically optimal size for maximum geometric growth (fractional).

We default to a "Dynamic" method that takes the most conservative of Volatility and ATR-based sizing.

Author: Stoic Citadel Team
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.risk.hrp import get_hrp_weights
from src.risk.volatility_scaler import VolatilityScaler
from src.utils.logger import log

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """Position sizing configuration."""

    # Base parameters
    max_position_pct: float = 0.10  # Max 10% per position
    max_portfolio_risk_pct: float = 0.20  # Max 20% portfolio at risk
    max_correlation_exposure: float = 0.60  # Max 60% correlated exposure

    # Risk parameters
    base_risk_per_trade: float = 0.02  # 2% risk per trade
    kelly_fraction: float = 0.25  # Use 25% of Kelly

    # Volatility adjustment
    target_volatility: float = 0.15  # 15% annual target vol
    volatility_lookback_days: int = 20
    min_vol_scalar: float = 0.5  # Min 50% of base size
    max_vol_scalar: float = 2.0  # Max 200% of base size

    # VaR parameters
    var_confidence: float = 0.95  # 95% VaR
    var_horizon_days: int = 1
    max_position_var_pct: float = 0.02  # Max 2% VaR per position


class PositionSizer:
    """
    Advanced position sizing with multiple methods.
    """

    def __init__(self, config: PositionSizingConfig | None = None) -> None:
        """
        Initialize PositionSizer.

        Args:
            config: Configuration object.
        """
        self.config = config or PositionSizingConfig()
        self._correlation_matrix: pd.DataFrame | None = None
        self._current_positions: dict[str, float] = {}
        from src.config.unified_config import load_config
        u_cfg = load_config()
        self._volatility_scaler = VolatilityScaler(u_cfg.volatility_scaler)

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        method: str = "fixed_risk",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Calculate position size using specified method.

        Args:
            account_balance: Total account balance.
            entry_price: Planned entry price.
            stop_loss_price: Stop loss price.
            method: Sizing method to use ('fixed_risk', 'volatility', 'var', 'kelly', 'hrp', 'optimal').
            **kwargs: Additional parameters for specific methods.

        Returns:
            Dictionary with position details.
        """
        methods: dict[str, Callable] = {
            "fixed_risk": self._fixed_risk_size,
            "volatility": self._volatility_adjusted_size,
            "var": self._var_based_size,
            "kelly": self._kelly_size,
            "hrp": self._hrp_size,
            "optimal": self._optimal_size,  # Combines all methods
        }

        if method not in methods:
            raise ValueError(f"Unknown sizing method: {method}")

        size_func = methods[method]
        result = size_func(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            **kwargs,
        )

        # Phase 2: Dynamic Volatility Scaling
        returns = kwargs.get("returns")
        if returns is not None and not returns.empty:
            # Calculate annualized volatility from returns
            # Assuming daily returns, 252 trading days
            # If returns are hourly, this would be different.
            # The calling context must provide appropriate returns series.
            current_volatility = float(returns.std() * np.sqrt(252)) # Annualized
            
            volatility_multiplier = self._volatility_scaler.calculate_multiplier(current_volatility)

            result["position_size"] *= volatility_multiplier
            result["position_value"] *= volatility_multiplier
            result["volatility_multiplier"] = volatility_multiplier
            result["current_volatility"] = current_volatility

        # Apply max position limit
        max_position_value = account_balance * self.config.max_position_pct
        if result["position_value"] > max_position_value:
            scale_factor = max_position_value / result["position_value"]
            result["position_size"] *= scale_factor
            result["position_value"] = max_position_value
            result["limited_by"] = "max_position_pct"

        return result


    def _fixed_risk_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        risk_pct: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Fixed risk position sizing.

        Position size = (Account * Risk%) / (Entry - Stop)
        """
        risk_pct = risk_pct or self.config.base_risk_per_trade
        risk_amount = account_balance * risk_pct

        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            raise ValueError("Stop loss cannot equal entry price")

        position_size = risk_amount / risk_per_unit
        position_value = position_size * entry_price

        return {
            "method": "fixed_risk",
            "position_size": position_size,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_pct": risk_pct,
            "risk_per_unit": risk_per_unit,
            "limited_by": None,
        }

    def _volatility_adjusted_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        current_volatility: float | None = None,
        returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Volatility-adjusted position sizing.

        Reduces size in high volatility, increases in low volatility.
        """
        # Get base size first
        base_result = self._fixed_risk_size(account_balance, entry_price, stop_loss_price)

        # Calculate volatility scalar
        if current_volatility is None and returns is not None:
            current_volatility = float(returns.std() * np.sqrt(252))  # Annualized
        elif current_volatility is None:
            current_volatility = self.config.target_volatility

        vol_scalar = self.config.target_volatility / max(current_volatility, 0.001)
        vol_scalar = float(
            np.clip(vol_scalar, self.config.min_vol_scalar, self.config.max_vol_scalar)
        )

        adjusted_size = base_result["position_size"] * vol_scalar
        adjusted_value = adjusted_size * entry_price

        return {
            "method": "volatility_adjusted",
            "position_size": adjusted_size,
            "position_value": adjusted_value,
            "risk_amount": base_result["risk_amount"],
            "risk_pct": base_result["risk_pct"],
            "vol_scalar": vol_scalar,
            "current_volatility": current_volatility,
            "target_volatility": self.config.target_volatility,
            "limited_by": None,
        }

    def _var_based_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        returns: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        VaR-based position sizing.

        Sizes position so that VaR doesn't exceed threshold.
        """
        if returns is None or len(returns) < 20:
            # Fallback to fixed risk if no returns data
            return self._fixed_risk_size(account_balance, entry_price, stop_loss_price)

        # Calculate VaR
        var_pct = self._calculate_var(returns)

        # Max position value based on VaR
        max_var_amount = account_balance * self.config.max_position_var_pct
        max_position_value = max_var_amount / var_pct if var_pct > 0 else account_balance

        # Calculate position size
        position_size = max_position_value / entry_price

        # Check against fixed risk
        fixed_result = self._fixed_risk_size(account_balance, entry_price, stop_loss_price)

        # Use smaller of VaR and fixed risk
        limiting_factor = "var"
        if position_size > fixed_result["position_size"]:
            position_size = fixed_result["position_size"]
            max_position_value = position_size * entry_price
            limiting_factor = "fixed_risk"

        return {
            "method": "var_based",
            "position_size": position_size,
            "position_value": max_position_value,
            "var_pct": var_pct,
            "var_confidence": self.config.var_confidence,
            "max_var_amount": max_var_amount,
            "limited_by": limiting_factor,
        }

    def _kelly_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        win_rate: float = 0.50,
        avg_win: float = 0.03,
        avg_loss: float = 0.02,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Kelly Criterion position sizing.

        f* = (p*b - q) / b
        where:
            p = win probability
            q = loss probability (1-p)
            b = win/loss ratio
        """
        if avg_loss == 0:
            avg_loss = 0.01  # Prevent division by zero

        win_loss_ratio = avg_win / avg_loss
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Use fractional Kelly
        kelly_pct = max(0.0, kelly_pct * self.config.kelly_fraction)
        kelly_pct = min(kelly_pct, self.config.max_position_pct)

        position_value = account_balance * kelly_pct
        position_size = position_value / entry_price

        return {
            "method": "kelly",
            "position_size": position_size,
            "position_value": position_value,
            "kelly_full_pct": (
                kelly_pct / self.config.kelly_fraction if self.config.kelly_fraction > 0 else 0
            ),
            "kelly_fractional_pct": kelly_pct,
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "kelly_fraction_used": self.config.kelly_fraction,
            "limited_by": None,
        }

    def _hrp_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        prices: pd.DataFrame,
        symbol: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Hierarchical Risk Parity (HRP) position sizing.
        """
        if prices is None or prices.empty:
            raise ValueError("HRP sizing requires a `prices` dataframe.")

        weights = get_hrp_weights(prices)
        symbol_weight = weights.get(symbol, 0.0)

        position_value = account_balance * symbol_weight
        position_size = position_value / entry_price

        return {
            "method": "hrp",
            "position_size": position_size,
            "position_value": position_value,
            "symbol_weight": symbol_weight,
            "hrp_weights": weights,
            "limited_by": None,
        }

    def _optimal_size(
        self, account_balance: float, entry_price: float, stop_loss_price: float, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Optimal position sizing combining all methods.

        Takes the minimum of all sizing methods for safety.
        """
        results = {}

        # Calculate all sizes
        results["fixed_risk"] = self._fixed_risk_size(
            account_balance, entry_price, stop_loss_price, **kwargs
        )

        if "returns" in kwargs or "current_volatility" in kwargs:
            results["volatility"] = self._volatility_adjusted_size(
                account_balance, entry_price, stop_loss_price, **kwargs
            )

        if "returns" in kwargs:
            results["var"] = self._var_based_size(
                account_balance, entry_price, stop_loss_price, **kwargs
            )

        if "win_rate" in kwargs:
            results["kelly"] = self._kelly_size(
                account_balance, entry_price, stop_loss_price, **kwargs
            )

        if "prices" in kwargs and "symbol" in kwargs:
            results["hrp"] = self._hrp_size(account_balance, entry_price, stop_loss_price, **kwargs)

        # Find minimum
        min_size = float("inf")
        min_method = "fixed_risk"
        for method, result in results.items():
            if result["position_size"] < min_size:
                min_size = result["position_size"]
                min_method = method

        final_result = results[min_method].copy()
        final_result["method"] = "optimal"
        final_result["selected_method"] = min_method
        final_result["all_methods"] = {k: v["position_size"] for k, v in results.items()}

        return final_result

    def _calculate_var(self, returns: pd.Series, confidence: float | None = None) -> float:
        """Calculate VaR from returns."""
        confidence = confidence or self.config.var_confidence
        return float(abs(np.percentile(returns, (1 - confidence) * 100)))

    def check_portfolio_risk(
        self, new_position: dict[str, Any], symbol: str, account_balance: float
    ) -> tuple[bool, str]:
        """
        Check if new position fits within portfolio risk limits.

        Returns:
            Tuple of (allowed, reason)
        """
        current_exposure = sum(self._current_positions.values())
        new_exposure = current_exposure + new_position.get("position_value", 0)

        max_exposure = account_balance * self.config.max_portfolio_risk_pct

        if new_exposure > max_exposure:
            msg = f"Portfolio exposure {new_exposure / account_balance:.1%} exceeds limit {self.config.max_portfolio_risk_pct:.1%}"
            log.info(
                "risk_rejection", symbol=symbol, reason=msg, rejection_type="portfolio_exposure"
            )
            return (False, msg)

        # Check correlation exposure if matrix available
        if self._correlation_matrix is not None and symbol in self._correlation_matrix.columns:
            correlated_exposure = self._calculate_correlated_exposure(
                symbol, new_position.get("position_value", 0)
            )
            max_correlated = account_balance * self.config.max_correlation_exposure

            if correlated_exposure > max_correlated:
                msg = (
                    f"Correlated exposure {correlated_exposure / account_balance:.1%} exceeds limit"
                )
                log.info(
                    "risk_rejection",
                    symbol=symbol,
                    reason=msg,
                    rejection_type="correlation_exposure",
                )
                return (False, msg)

        return True, "OK"

    def update_positions(self, positions: dict[str, float]) -> None:
        """Update current positions for portfolio risk checks."""
        self._current_positions = positions.copy()

    def update_correlation_matrix(self, matrix: pd.DataFrame) -> None:
        """Update correlation matrix for portfolio risk."""
        self._correlation_matrix = matrix

    def _calculate_correlated_exposure(self, symbol: str, position_value: float) -> float:
        """Calculate total correlated exposure."""
        if self._correlation_matrix is None:
            return position_value

        total_correlated = position_value

        for other_symbol, other_value in self._current_positions.items():
            if other_symbol in self._correlation_matrix.columns:
                correlation = abs(self._correlation_matrix.loc[symbol, other_symbol])
                if correlation > 0.5:  # Only count highly correlated
                    total_correlated += other_value * correlation

        return total_correlated

    def calculate_atr_based_size(
        self,
        account_balance: float,
        entry_price: float,
        dataframe: pd.DataFrame,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ) -> dict[str, Any]:
        """
        Calculate position size based on ATR for stop-loss sizing.

        This method sizes positions so that a 2xATR move equals your
        risk per trade. More volatile = smaller position.

        Args:
            account_balance: Total account balance
            entry_price: Current/planned entry price
            dataframe: OHLCV dataframe with high, low, close
            atr_period: Period for ATR calculation
            atr_multiplier: Stop-loss distance in ATR multiples
            risk_per_trade: Maximum risk per trade (e.g., 0.01 = 1%)

        Returns:
            Dictionary with position details and calculated stop-loss
        """
        # Calculate ATR
        high_low = dataframe["high"] - dataframe["low"]
        high_close = np.abs(dataframe["high"] - dataframe["close"].shift())
        low_close = np.abs(dataframe["low"] - dataframe["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(atr_period).mean().iloc[-1]

        if pd.isna(atr) or atr <= 0:
            logger.warning("Invalid ATR, using fixed risk sizing")
            return self._fixed_risk_size(account_balance, entry_price, entry_price * 0.98)

        # Calculate stop-loss price
        stop_loss_distance = atr * atr_multiplier
        stop_loss_price = entry_price - stop_loss_distance

        # Calculate position size
        # Risk = Position * (Entry - Stop) / Entry = risk_per_trade * Balance
        # Position = (risk_per_trade * Balance) / (Stop Distance / Entry)
        stop_loss_pct = stop_loss_distance / entry_price
        position_value = (risk_per_trade * account_balance) / stop_loss_pct
        position_size = position_value / entry_price

        # Apply max position limit
        max_position_value = account_balance * self.config.max_position_pct
        limiting_factor = None
        if position_value > max_position_value:
            position_value = max_position_value
            position_size = position_value / entry_price
            limiting_factor = "max_position_pct"

        return {
            "method": "atr_based",
            "position_size": position_size,
            "position_value": position_value,
            "atr": atr,
            "atr_pct": atr / entry_price,
            "stop_loss_price": stop_loss_price,
            "stop_loss_distance": stop_loss_distance,
            "stop_loss_pct": stop_loss_pct,
            "risk_amount": position_value * stop_loss_pct,
            "risk_pct": risk_per_trade,
            "atr_multiplier": atr_multiplier,
            "limited_by": limiting_factor,
        }

    def calculate_dynamic_stake(
        self,
        account_balance: float,
        entry_price: float,
        dataframe: pd.DataFrame,
        current_drawdown: float = 0.0,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
    ) -> dict[str, Any]:
        """
        Calculate dynamic stake using best available method.

        Combines volatility targeting, ATR sizing, and optional Kelly
        to determine optimal position size.

        This is the recommended method for Freqtrade integration.

        Args:
            account_balance: Total account balance
            entry_price: Current asset price
            dataframe: OHLCV dataframe
            current_drawdown: Current portfolio drawdown (0.05 = 5%)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average winning trade profit
            avg_loss: Average losing trade loss

        Returns:
            Dictionary with optimal stake details
        """
        results = {}

        # 1. Calculate volatility-adjusted size using the new VolatilityScaler
        returns = dataframe["close"].pct_change().dropna()
        if len(returns) >= 20:
            # Annualize volatility based on the timeframe of the data
            # This is a placeholder, a more robust solution would analyze the dataframe's timeframe
            time_diff = (dataframe["date"].iloc[1] - dataframe["date"].iloc[0]).total_seconds()
            candles_per_day = 86400 / time_diff
            annualization_factor = np.sqrt(252 * candles_per_day)
            current_vol = returns.std() * annualization_factor
            
            volatility_multiplier = self._volatility_scaler.calculate_multiplier(current_vol)

            # Get a base size from fixed risk
            base_size = self._fixed_risk_size(
                account_balance, entry_price, entry_price * 0.98 # Placeholder SL
            )["position_value"]
            
            results["volatility"] = base_size * volatility_multiplier

        # 2. Calculate ATR-based size
        atr_result = self.calculate_atr_based_size(
            account_balance=account_balance, entry_price=entry_price, dataframe=dataframe
        )
        results["atr"] = atr_result["position_value"]

        # 3. Calculate Kelly if stats available
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_result = self._kelly_size(
                account_balance=account_balance,
                entry_price=entry_price,
                stop_loss_price=entry_price * 0.98,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )
            results["kelly"] = kelly_result["position_value"]

        # Take minimum (most conservative)
        min_method = min(results.keys(), key=lambda k: results[k])
        optimal_value = results[min_method]

        # Apply drawdown reduction
        if current_drawdown >= 0.05:  # 5% drawdown threshold
            reduction_factor = max(0.5, 1.0 - current_drawdown)
            optimal_value *= reduction_factor
            logger.info(
                f"Drawdown {current_drawdown:.1%} - reducing stake by {1 - reduction_factor:.0%}"
            )

        return {
            "method": "dynamic",
            "selected_method": min_method,
            "position_value": optimal_value,
            "position_size": optimal_value / entry_price,
            "all_methods": results,
            "atr_stop_loss": atr_result.get("stop_loss_price"),
            "drawdown_applied": current_drawdown >= 0.05,
        }


def create_freqtrade_stake_function(config: PositionSizingConfig | None = None) -> Callable:
    """
    Create a custom_stake_amount function for Freqtrade strategies.

    Usage in strategy:
        from src.risk.position_sizing import create_freqtrade_stake_function

        class MyStrategy(IStrategy):
            custom_stake_amount = create_freqtrade_stake_function()
    """
    sizer = PositionSizer(config)

    def custom_stake_amount(
        self: Any,
        pair: str,
        current_time: Any,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        """
        Freqtrade custom_stake_amount callback.

        Implements dynamic position sizing based on volatility and ATR.
        """
        try:
            # Get dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty or len(dataframe) < 50:
                return proposed_stake

            # Get wallet balance
            wallet_balance = self.wallets.get_total_stake_amount()

            # Calculate current drawdown
            starting_balance = self.wallets.get_starting_balance()
            if starting_balance > 0:
                current_drawdown = max(0.0, (starting_balance - wallet_balance) / starting_balance)
            else:
                current_drawdown = 0.0

            # Calculate dynamic stake
            result = sizer.calculate_dynamic_stake(
                account_balance=wallet_balance,
                entry_price=current_rate,
                dataframe=dataframe,
                current_drawdown=current_drawdown,
            )

            stake = result["position_value"]

            # Respect min/max
            if min_stake is not None:
                stake = max(min_stake, stake)
            stake = min(max_stake, stake)

            logger.info(
                f"Dynamic stake for {pair}: {stake:.2f} "
                f"(method: {result['selected_method']}, "
                f"proposed: {proposed_stake:.2f})"
            )

            return stake

        except Exception as e:
            logger.error(f"Error in custom_stake_amount: {e}")
            return proposed_stake

    return custom_stake_amount
