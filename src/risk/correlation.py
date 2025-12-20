"""
Portfolio Correlation & Risk Management
========================================

Prevents opening correlated positions that amplify portfolio risk.

Key Concept:
If BTC/USDT is falling, don't open ETH/USDT long even if strategy signals.

Author: Stoic Citadel Team
License: MIT
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy.cluster import hierarchy

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


class CorrelationAnalyzer:
    """
    Monitor correlation between open positions and detect concentration risk.
    
    Uses hierarchical clustering to identify groups of highly correlated assets
    and warns if portfolio is too concentrated in any single cluster.
    """
    
    def __init__(self, window: int = 100):
        """
        Initialize correlation analyzer.
        
        Args:
            window: Rolling window for correlation calculation
        """
        self.correlation_matrix = None
        self.window = window
        
    def calculate_portfolio_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation between all positions.
        
        Args:
            returns_df: DataFrame with returns for all symbols.
                       Columns: symbols, index: datetime
                       
        Returns:
            Correlation matrix (rolling window correlation)
        """
        if len(returns_df) < self.window:
            logger.warning(
                f"Insufficient data for correlation calculation: "
                f"have {len(returns_df)} rows, need {self.window}"
            )
            # Use simple correlation if insufficient data
            self.correlation_matrix = returns_df.corr()
        else:
            # Calculate rolling correlation
            self.correlation_matrix = returns_df.rolling(self.window).corr()
            
        logger.info(f"Correlation matrix calculated: {self.correlation_matrix.shape}")
        return self.correlation_matrix
        
    def check_concentration_risk(
        self, 
        positions: List['Position'], 
        correlation_threshold: float = 0.7,
        max_cluster_exposure: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Detect if portfolio is too concentrated in correlated assets.
        
        Args:
            positions: List of Position objects
            correlation_threshold: Threshold for high correlation (default: 0.7)
            max_cluster_exposure: Maximum allowed exposure to a single cluster (default: 0.5 = 50%)
            
        Returns:
            Tuple of (has_risk: bool, message: str)
        """
        if self.correlation_matrix is None:
            return False, "No correlation matrix available"
            
        if not positions:
            return False, "No positions to analyze"
            
        # Extract unique symbols from positions
        symbols = [pos.symbol for pos in positions]
        if len(symbols) < 2:
            return False, "Insufficient positions for correlation analysis"
            
        # Get the latest correlation matrix
        corr_matrix = self._get_latest_correlation_matrix(symbols)
        
        # Ensure we only have correlations for symbols we have positions in
        available_symbols = [s for s in symbols if s in corr_matrix.index]
        if len(available_symbols) < 2:
            return False, f"Insufficient correlation data for symbols: {symbols}"
            
        # Subset correlation matrix
        corr_subset = corr_matrix.loc[available_symbols, available_symbols]
        
        # Fill NaN values with 0 (no correlation)
        corr_subset = corr_subset.fillna(0)
        
        # Convert to numpy array for clustering
        corr_array = corr_subset.values
        
        # Perform hierarchical clustering
        try:
            # Calculate distance matrix (1 - absolute correlation)
            distance_matrix = 1 - np.abs(corr_array)
            
            # Perform hierarchical clustering
            linkage_matrix = hierarchy.linkage(distance_matrix, method='ward')
            
            # Form clusters based on correlation threshold
            clusters = hierarchy.fcluster(
                linkage_matrix, 
                t=1 - correlation_threshold,  # Convert correlation to distance
                criterion='distance'
            )
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return False, f"Clustering failed: {e}"
        
        # Map symbols to cluster IDs
        symbol_to_cluster = dict(zip(available_symbols, clusters))
        
        # Calculate total portfolio value
        total_value = 0.0
        cluster_values = {}
        
        for position in positions:
            if position.symbol not in symbol_to_cluster:
                continue
                
            # Calculate position value (quantity * current price)
            position_value = position.quantity * position.current_price
            total_value += position_value
            
            cluster_id = symbol_to_cluster[position.symbol]
            cluster_values[cluster_id] = cluster_values.get(cluster_id, 0.0) + position_value
        
        if total_value == 0:
            return False, "Portfolio has zero value"
        
        # Check if any cluster exceeds max exposure
        for cluster_id, cluster_value in cluster_values.items():
            cluster_exposure = cluster_value / total_value
            
            if cluster_exposure > max_cluster_exposure:
                # Get symbols in this cluster
                cluster_symbols = [
                    sym for sym, cid in symbol_to_cluster.items() 
                    if cid == cluster_id
                ]
                
                message = (
                    f"Concentration risk in cluster {cluster_id}: "
                    f"{cluster_exposure:.1%} exposure > {max_cluster_exposure:.1%} limit. "
                    f"Symbols in cluster: {cluster_symbols}"
                )
                logger.warning(f"⚠️ {message}")
                return True, message
        
        return False, "OK - No concentration risk detected"
    
    def _get_latest_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """
        Extract the latest correlation matrix from stored correlation data.
        
        Args:
            symbols: List of symbols to include
            
        Returns:
            Correlation matrix DataFrame
        """
        if self.correlation_matrix is None:
            # Return identity matrix if no correlation data
            return pd.DataFrame(
                np.eye(len(symbols)), 
                index=symbols, 
                columns=symbols
            )
        
        # Check if we have a MultiIndex (rolling correlation)
        if hasattr(self.correlation_matrix.index, 'levels') and len(self.correlation_matrix.index.levels) > 1:
            # This is a rolling correlation with MultiIndex
            try:
                # Get the last timestamp with complete data
                last_idx = self.correlation_matrix.index.get_level_values(0)[-1]
                corr_matrix = self.correlation_matrix.loc[last_idx]
                
                # Ensure it's a DataFrame with symbols as index
                if isinstance(corr_matrix, pd.DataFrame):
                    return corr_matrix
                else:
                    # Might be a Series, convert to DataFrame
                    logger.warning("Unexpected correlation matrix format, using identity")
                    return pd.DataFrame(
                        np.eye(len(symbols)), 
                        index=symbols, 
                        columns=symbols
                    )
            except (AttributeError, KeyError, IndexError) as e:
                logger.warning(f"Could not extract correlation matrix: {e}, using identity")
                return pd.DataFrame(
                    np.eye(len(symbols)), 
                    index=symbols, 
                    columns=symbols
                )
        else:
            # Simple correlation matrix
            return self.correlation_matrix
        
    def get_correlation_summary(self) -> Dict:
        """
        Get summary of correlation matrix.
        
        Returns:
            Dictionary with correlation statistics
        """
        if self.correlation_matrix is None:
            return {"status": "No correlation matrix available"}
        
        # Get a sample of symbols to extract correlation matrix
        # We need some symbols to call _get_latest_correlation_matrix
        # Try to get symbols from the correlation matrix itself
        symbols = []
        if hasattr(self.correlation_matrix, 'columns'):
            symbols = list(self.correlation_matrix.columns)
        elif hasattr(self.correlation_matrix.index, 'levels') and len(self.correlation_matrix.index.levels) > 1:
            # MultiIndex case - get symbols from second level
            symbols = list(self.correlation_matrix.index.get_level_values(1).unique())
        
        if not symbols:
            return {"status": "Could not extract symbols from correlation matrix"}
        
        try:
            # Get the latest correlation matrix
            corr_matrix = self._get_latest_correlation_matrix(symbols[:min(10, len(symbols))])
            
            # Get correlation values
            corr_values = corr_matrix.values.flatten()
            
            # Filter out diagonal (self-correlation = 1) and NaN
            corr_values = corr_values[~np.isnan(corr_values)]
            corr_values = corr_values[np.abs(corr_values - 1) > 1e-10]
            
            if len(corr_values) == 0:
                return {"status": "No valid correlation values"}
                
            return {
                "status": "OK",
                "mean_correlation": float(np.mean(corr_values)),
                "median_correlation": float(np.median(corr_values)),
                "std_correlation": float(np.std(corr_values)),
                "min_correlation": float(np.min(corr_values)),
                "max_correlation": float(np.max(corr_values)),
                "high_correlation_count": int(np.sum(np.abs(corr_values) > 0.7)),
                "negative_correlation_count": int(np.sum(corr_values < -0.3)),
            }
        except Exception as e:
            logger.error(f"Error getting correlation summary: {e}")
            return {"status": f"Error: {str(e)}"}
