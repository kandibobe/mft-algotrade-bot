"""
Stoic Citadel - Risk Management Utilities
==========================================

Position sizing, risk calculations, and portfolio management.

"The wise man does not expose himself to unnecessary risk."
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_position_size_fixed_risk(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    min_position: float = 0.0,
    max_position: Optional[float] = None,
) -> float:
    """
    Calculate position size using fixed risk method.

    Args:
        account_balance: Total account balance
        risk_per_trade: Risk per trade as decimal (e.g., 0.02 = 2%)
        entry_price: Planned entry price
        stop_loss_price: Stop loss price
        min_position: Minimum position size
        max_position: Maximum position size

    Returns:
        Position size in base currency
    """
    # Calculate risk amount in currency
    risk_amount = account_balance * risk_per_trade

    # Calculate risk per unit (distance to stop loss)
    risk_per_unit = abs(entry_price - stop_loss_price)

    if risk_per_unit == 0:
        logger.warning("Risk per unit is zero, returning minimum position")
        return min_position

    # Position size
    position_size = risk_amount / risk_per_unit

    # Apply limits
    position_size = max(min_position, position_size)
    if max_position:
        position_size = min(max_position, position_size)

    return position_size


def calculate_position_size_kelly(
    account_balance: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.25,
    max_exposure: float = 0.25,
) -> float:
    """
    Calculate position size using Kelly Criterion.

    Uses fractional Kelly (default 25%) for safety.

    Args:
        account_balance: Total account balance
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)
        kelly_fraction: Fraction of full Kelly to use (default 0.25)
        max_exposure: Maximum position as fraction of balance

    Returns:
        Position size as fraction of balance
    """
    if avg_loss == 0:
        logger.warning("Average loss is zero, returning 0")
        return 0.0

    # Win/Loss ratio
    win_loss_ratio = avg_win / avg_loss

    # Full Kelly
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

    # Fractional Kelly with bounds
    position_fraction = max(0, min(kelly * kelly_fraction, max_exposure))

    return account_balance * position_fraction


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Series of portfolio values over time

    Returns:
        Tuple of (max_drawdown_pct, peak_idx, trough_idx)
    """
    # Running maximum
    running_max = equity_curve.expanding().max()

    # Drawdown at each point
    drawdowns = (equity_curve - running_max) / running_max

    # Maximum drawdown
    max_dd = drawdowns.min()

    # Find indices
    trough_idx = drawdowns.idxmin()
    peak_idx = equity_curve[:trough_idx].idxmax()

    return abs(max_dd), peak_idx, trough_idx


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24 * 12,  # 5-minute candles
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Number of periods per year (for annualization)

    Returns:
        Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0

    # Remove NaN
    returns = returns.dropna()

    # Mean and std of returns
    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)

    # Sharpe
    sharpe = (annualized_return - risk_free_rate) / annualized_std

    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252 * 24 * 12
) -> float:
    """
    Calculate annualized Sortino Ratio (downside deviation only).

    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0

    returns = returns.dropna()
    mean_return = returns.mean()

    # Downside returns only
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float("inf")  # No downside = perfect

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_downside_std = downside_std * np.sqrt(periods_per_year)

    sortino = (annualized_return - risk_free_rate) / annualized_downside_std

    return sortino


def calculate_calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252 * 24 * 12) -> float:
    """
    Calculate Calmar Ratio (return / max drawdown).

    Args:
        equity_curve: Series of portfolio values
        periods_per_year: Number of periods per year

    Returns:
        Calmar Ratio
    """
    if len(equity_curve) < 2:
        return 0.0

    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Annualize
    num_periods = len(equity_curve)
    annualized_return = ((1 + total_return) ** (periods_per_year / num_periods)) - 1

    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return float("inf")

    return annualized_return / max_dd


def calculate_risk_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Dictionary of risk metrics
    """
    returns = equity_curve.pct_change().dropna()

    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity_curve)

    return {
        "total_return": (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        "max_drawdown": max_dd,
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "sortino_ratio": calculate_sortino_ratio(returns),
        "calmar_ratio": calculate_calmar_ratio(equity_curve),
        "volatility": returns.std() * np.sqrt(252 * 24 * 12),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "var_95": returns.quantile(0.05),  # 95% VaR
        "win_rate": (returns > 0).mean(),
    }
