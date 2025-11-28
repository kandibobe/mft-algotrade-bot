"""
Portfolio Correlation & Risk Management
========================================

Prevents opening correlated positions that amplify portfolio risk.

Key Concept:
If BTC/USDT is falling, don't open ETH/USDT long even if strategy signals.

Author: Stoic Citadel Team
License: MIT
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CorrelationManager:
    """
    Manage portfolio correlation to prevent concentration risk.

    Design Philosophy:
    - Calculate rolling correlation between assets
    - Block entries if portfolio heat exceeds threshold
    - Force-close highly correlated losing positions
    """

    def __init__(
        self,
        correlation_window: int = 24,  # hours
        max_correlation: float = 0.7,
        max_portfolio_heat: float = 0.15  # 15%
    ):
        """
        Initialize correlation manager.

        Args:
            correlation_window: Rolling window for correlation (hours)
            max_correlation: Maximum allowed correlation with open positions
            max_portfolio_heat: Maximum portfolio exposure
        """
        self.correlation_window = correlation_window
        self.max_correlation = max_correlation
        self.max_portfolio_heat = max_portfolio_heat

        # Cache for price data
        self.price_cache: Dict[str, pd.DataFrame] = {}

    def calculate_correlation(
        self,
        pair1_data: pd.DataFrame,
        pair2_data: pd.DataFrame,
        method: str = 'pearson'
    ) -> float:
        """
        Calculate correlation between two pairs.

        Args:
            pair1_data: OHLCV dataframe for pair 1
            pair2_data: OHLCV dataframe for pair 2
            method: Correlation method ('pearson', 'spearman')

        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Align indices
        common_index = pair1_data.index.intersection(pair2_data.index)

        if len(common_index) < 10:
            logger.warning("Insufficient data for correlation calculation")
            return 0.0

        # Calculate returns
        returns1 = pair1_data.loc[common_index, 'close'].pct_change().dropna()
        returns2 = pair2_data.loc[common_index, 'close'].pct_change().dropna()

        # Calculate correlation
        if len(returns1) < 10 or len(returns2) < 10:
            return 0.0

        corr = returns1.corr(returns2, method=method)
        return corr if not np.isnan(corr) else 0.0

    def check_entry_correlation(
        self,
        new_pair: str,
        new_pair_data: pd.DataFrame,
        open_positions: List[Dict],
        all_pairs_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Check if new entry would create excessive correlation.

        Args:
            new_pair: Pair to enter
            new_pair_data: OHLCV data for new pair
            open_positions: List of currently open positions
            all_pairs_data: Dict of {pair: ohlcv_dataframe}

        Returns:
            True if entry allowed, False if blocked by correlation
        """
        if not open_positions:
            return True  # No correlation risk with empty portfolio

        # Check correlation with each open position
        for position in open_positions:
            open_pair = position.get('pair')

            if open_pair not in all_pairs_data:
                logger.warning(f"No data for {open_pair}, skipping correlation check")
                continue

            open_pair_data = all_pairs_data[open_pair]

            # Calculate correlation
            corr = self.calculate_correlation(new_pair_data, open_pair_data)

            logger.info(f"Correlation {new_pair} vs {open_pair}: {corr:.2f}")

            # Block if correlation too high
            if abs(corr) > self.max_correlation:
                logger.warning(
                    f"❌ Blocking {new_pair} entry: "
                    f"correlation {corr:.2f} > {self.max_correlation} "
                    f"with {open_pair}"
                )
                return False

        return True

    def calculate_portfolio_heat(
        self,
        open_positions: List[Dict],
        current_prices: Dict[str, float]
    ) -> float:
        """
        Calculate current portfolio exposure (heat).

        Portfolio heat = sum of (position_size * current_unrealized_loss)

        Args:
            open_positions: List of open positions
            current_prices: Dict of {pair: current_price}

        Returns:
            Portfolio heat as percentage (0.0 to 1.0)
        """
        total_heat = 0.0
        total_capital = 0.0

        for position in open_positions:
            pair = position.get('pair')
            entry_price = position.get('open_rate', 0)
            stake_amount = position.get('stake_amount', 0)

            current_price = current_prices.get(pair, entry_price)

            # Calculate unrealized P&L
            pnl_pct = (current_price - entry_price) / entry_price

            # Only count losing positions
            if pnl_pct < 0:
                heat = abs(pnl_pct) * stake_amount
                total_heat += heat

            total_capital += stake_amount

        if total_capital == 0:
            return 0.0

        portfolio_heat = total_heat / total_capital
        return portfolio_heat

    def check_portfolio_heat(
        self,
        open_positions: List[Dict],
        current_prices: Dict[str, float]
    ) -> bool:
        """
        Check if portfolio heat is within limits.

        Args:
            open_positions: List of open positions
            current_prices: Dict of {pair: current_price}

        Returns:
            True if heat acceptable, False if exceeded
        """
        heat = self.calculate_portfolio_heat(open_positions, current_prices)

        logger.info(f"Portfolio heat: {heat:.2%} (max: {self.max_portfolio_heat:.2%})")

        if heat > self.max_portfolio_heat:
            logger.error(
                f"❌ Portfolio heat {heat:.2%} exceeds "
                f"maximum {self.max_portfolio_heat:.2%}"
            )
            return False

        return True


class DrawdownMonitor:
    """
    Monitor and enforce maximum drawdown limits.

    Implements circuit breaker pattern.
    """

    def __init__(
        self,
        max_drawdown: float = 0.15,  # 15%
        stop_duration_minutes: int = 240  # 4 hours
    ):
        """
        Initialize drawdown monitor.

        Args:
            max_drawdown: Maximum allowed drawdown
            stop_duration_minutes: How long to stop trading after breach
        """
        self.max_drawdown = max_drawdown
        self.stop_duration_minutes = stop_duration_minutes
        self.circuit_breaker_until: Optional[datetime] = None

    def check_drawdown(
        self,
        current_balance: float,
        peak_balance: float
    ) -> bool:
        """
        Check if drawdown exceeds limit.

        Args:
            current_balance: Current account balance
            peak_balance: Historical peak balance

        Returns:
            True if trading allowed, False if circuit breaker triggered
        """
        # Check if circuit breaker is active
        if self.circuit_breaker_until:
            if datetime.now() < self.circuit_breaker_until:
                logger.warning(
                    f"🔒 Circuit breaker active until "
                    f"{self.circuit_breaker_until.strftime('%Y-%m-%d %H:%M')}"
                )
                return False
            else:
                logger.info("✅ Circuit breaker expired, resuming trading")
                self.circuit_breaker_until = None

        # Calculate drawdown
        drawdown = (peak_balance - current_balance) / peak_balance

        logger.info(
            f"Drawdown: {drawdown:.2%} "
            f"(current: ${current_balance:.2f}, peak: ${peak_balance:.2f})"
        )

        # Trigger circuit breaker if exceeded
        if drawdown > self.max_drawdown:
            self.circuit_breaker_until = datetime.now() + timedelta(
                minutes=self.stop_duration_minutes
            )
            logger.error(
                f"⚠️ CIRCUIT BREAKER TRIGGERED! "
                f"Drawdown {drawdown:.2%} > {self.max_drawdown:.2%}. "
                f"Trading stopped until {self.circuit_breaker_until.strftime('%Y-%m-%d %H:%M')}"
            )
            return False

        return True

    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        if not self.circuit_breaker_until:
            return False

        return datetime.now() < self.circuit_breaker_until
