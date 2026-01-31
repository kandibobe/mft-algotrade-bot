"""
Stoic Citadel - Volatility Scaler
===================================

This module provides a continuous, curve-based approach to scaling position sizes
based on market volatility, replacing discrete, hardcoded buckets.

Financial Logic:
----------------
- In low volatility, risk can be increased to capture smaller moves.
- In high volatility, risk must be decreased to survive sharp swings.

This scaler uses a sigmoid function to map volatility to a multiplier, providing a
smooth transition between risk-on and risk-off sizing.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class VolatilityScalerConfig:
    """Configuration for the VolatilityScaler."""
    target_volatility: float = 0.15  # Target annualized volatility
    min_multiplier: float = 0.5    # Minimum position size multiplier
    max_multiplier: float = 2.0    # Maximum position size multiplier
    curve_aggressiveness: float = 10.0 # How quickly the multiplier adjusts

class VolatilityScaler:
    """
    Calculates a position size multiplier based on current market volatility.
    """

    def __init__(self, config: VolatilityScalerConfig | None = None) -> None:
        """
        Initialize VolatilityScaler.

        Args:
            config: Configuration object.
        """
        self.config = config or VolatilityScalerConfig()

    def _sigmoid(self, x: float) -> float:
        """Normalized sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def calculate_multiplier(self, current_volatility: float) -> float:
        """
        Calculate the size multiplier based on volatility.

        Args:
            current_volatility: The current annualized volatility.

        Returns:
            A sizing multiplier.
        """
        if current_volatility <= 0:
            return self.config.max_multiplier

        # Normalized difference from target volatility
        # Positive value means volatility is higher than target (reduce size)
        # Negative value means volatility is lower than target (increase size)
        vol_diff = (self.config.target_volatility - current_volatility) / self.config.target_volatility

        # Use sigmoid to map the difference to a value between 0 and 1
        # The aggressiveness factor controls the steepness of the curve
        curve = self._sigmoid(vol_diff * self.config.curve_aggressiveness)

        # Scale the 0-1 value to the min/max multiplier range
        multiplier = self.config.min_multiplier + curve * (self.config.max_multiplier - self.config.min_multiplier)

        return float(np.clip(multiplier, self.config.min_multiplier, self.config.max_multiplier))
