"""
Stoic Citadel - Advanced Position Sizing
=========================================

Multiple position sizing methods:
- Fixed Risk (Kelly Criterion variant)
- Volatility-adjusted sizing
- VaR-based sizing
- Correlation-adjusted portfolio sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """Position sizing configuration."""
    # Base parameters
    max_position_pct: float = 0.10          # Max 10% per position
    max_portfolio_risk_pct: float = 0.20    # Max 20% portfolio at risk
    max_correlation_exposure: float = 0.60  # Max 60% correlated exposure
    
    # Risk parameters
    base_risk_per_trade: float = 0.02       # 2% risk per trade
    kelly_fraction: float = 0.25            # Use 25% of Kelly
    
    # Volatility adjustment
    target_volatility: float = 0.15         # 15% annual target vol
    volatility_lookback_days: int = 20
    min_vol_scalar: float = 0.5             # Min 50% of base size
    max_vol_scalar: float = 2.0             # Max 200% of base size
    
    # VaR parameters
    var_confidence: float = 0.95            # 95% VaR
    var_horizon_days: int = 1
    max_position_var_pct: float = 0.02      # Max 2% VaR per position


class PositionSizer:
    """
    Advanced position sizing with multiple methods.
    """
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._current_positions: Dict[str, float] = {}
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        method: str = "fixed_risk",
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate position size using specified method.
        
        Args:
            account_balance: Total account balance
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            method: Sizing method to use
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary with position details
        """
        methods = {
            "fixed_risk": self._fixed_risk_size,
            "volatility": self._volatility_adjusted_size,
            "var": self._var_based_size,
            "kelly": self._kelly_size,
            "optimal": self._optimal_size  # Combines all methods
        }
        
        if method not in methods:
            raise ValueError(f"Unknown sizing method: {method}")
        
        size_func = methods[method]
        result = size_func(
            account_balance=account_balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            **kwargs
        )
        
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
        risk_pct: Optional[float] = None,
        **kwargs
    ) -> Dict[str, float]:
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
            "limited_by": None
        }
    
    def _volatility_adjusted_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        current_volatility: Optional[float] = None,
        returns: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Volatility-adjusted position sizing.
        
        Reduces size in high volatility, increases in low volatility.
        """
        # Get base size first
        base_result = self._fixed_risk_size(
            account_balance, entry_price, stop_loss_price
        )
        
        # Calculate volatility scalar
        if current_volatility is None and returns is not None:
            current_volatility = returns.std() * np.sqrt(252)  # Annualized
        elif current_volatility is None:
            current_volatility = self.config.target_volatility
        
        vol_scalar = self.config.target_volatility / max(current_volatility, 0.001)
        vol_scalar = np.clip(
            vol_scalar,
            self.config.min_vol_scalar,
            self.config.max_vol_scalar
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
            "limited_by": None
        }
    
    def _var_based_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        returns: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        VaR-based position sizing.
        
        Sizes position so that VaR doesn't exceed threshold.
        """
        if returns is None or len(returns) < 20:
            # Fallback to fixed risk if no returns data
            return self._fixed_risk_size(
                account_balance, entry_price, stop_loss_price
            )
        
        # Calculate VaR
        var_pct = self._calculate_var(returns)
        
        # Max position value based on VaR
        max_var_amount = account_balance * self.config.max_position_var_pct
        max_position_value = max_var_amount / var_pct if var_pct > 0 else account_balance
        
        # Calculate position size
        position_size = max_position_value / entry_price
        
        # Check against fixed risk
        fixed_result = self._fixed_risk_size(
            account_balance, entry_price, stop_loss_price
        )
        
        # Use smaller of VaR and fixed risk
        if position_size > fixed_result["position_size"]:
            position_size = fixed_result["position_size"]
            max_position_value = position_size * entry_price
            limiting_factor = "fixed_risk"
        else:
            limiting_factor = "var"
        
        return {
            "method": "var_based",
            "position_size": position_size,
            "position_value": max_position_value,
            "var_pct": var_pct,
            "var_confidence": self.config.var_confidence,
            "max_var_amount": max_var_amount,
            "limited_by": limiting_factor
        }
    
    def _kelly_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        win_rate: float = 0.50,
        avg_win: float = 0.03,
        avg_loss: float = 0.02,
        **kwargs
    ) -> Dict[str, float]:
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
        kelly_pct = max(0, kelly_pct * self.config.kelly_fraction)
        kelly_pct = min(kelly_pct, self.config.max_position_pct)
        
        position_value = account_balance * kelly_pct
        position_size = position_value / entry_price
        
        return {
            "method": "kelly",
            "position_size": position_size,
            "position_value": position_value,
            "kelly_full_pct": kelly_pct / self.config.kelly_fraction if self.config.kelly_fraction > 0 else 0,
            "kelly_fractional_pct": kelly_pct,
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "kelly_fraction_used": self.config.kelly_fraction,
            "limited_by": None
        }
    
    def _optimal_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        **kwargs
    ) -> Dict[str, float]:
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
        
        # Find minimum
        min_size = float('inf')
        min_method = "fixed_risk"
        for method, result in results.items():
            if result["position_size"] < min_size:
                min_size = result["position_size"]
                min_method = method
        
        final_result = results[min_method].copy()
        final_result["method"] = "optimal"
        final_result["selected_method"] = min_method
        final_result["all_methods"] = {
            k: v["position_size"] for k, v in results.items()
        }
        
        return final_result
    
    def _calculate_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> float:
        """Calculate VaR from returns."""
        confidence = confidence or self.config.var_confidence
        return abs(np.percentile(returns, (1 - confidence) * 100))
    
    def check_portfolio_risk(
        self,
        new_position: Dict[str, float],
        symbol: str,
        account_balance: float
    ) -> Tuple[bool, str]:
        """
        Check if new position fits within portfolio risk limits.
        
        Returns:
            Tuple of (allowed, reason)
        """
        current_exposure = sum(self._current_positions.values())
        new_exposure = current_exposure + new_position.get("position_value", 0)
        
        max_exposure = account_balance * self.config.max_portfolio_risk_pct
        
        if new_exposure > max_exposure:
            return False, f"Portfolio exposure {new_exposure/account_balance:.1%} exceeds limit {self.config.max_portfolio_risk_pct:.1%}"
        
        # Check correlation exposure if matrix available
        if self._correlation_matrix is not None and symbol in self._correlation_matrix.columns:
            correlated_exposure = self._calculate_correlated_exposure(
                symbol, new_position.get("position_value", 0)
            )
            max_correlated = account_balance * self.config.max_correlation_exposure
            
            if correlated_exposure > max_correlated:
                return False, f"Correlated exposure {correlated_exposure/account_balance:.1%} exceeds limit"
        
        return True, "OK"
    
    def update_positions(
        self,
        positions: Dict[str, float]
    ) -> None:
        """Update current positions for portfolio risk checks."""
        self._current_positions = positions.copy()
    
    def update_correlation_matrix(
        self,
        matrix: pd.DataFrame
    ) -> None:
        """Update correlation matrix for portfolio risk."""
        self._correlation_matrix = matrix
    
    def _calculate_correlated_exposure(
        self,
        symbol: str,
        position_value: float
    ) -> float:
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
