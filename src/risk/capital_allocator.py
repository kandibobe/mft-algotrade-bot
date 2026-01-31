"""
Stoic Citadel - Dynamic Capital Allocator
===========================================

This module implements the logic for dynamic alpha rotation, allocating capital
to the most efficient pairs based on a "Volatility / Fee" ratio.

Financial Logic:
----------------
- Pairs that move a lot (high volatility) but have low fees offer more alpha opportunities.
- By allocating more capital to these pairs, we increase the probability of capturing
  profitable trades.
- This is a form of dynamic portfolio optimization based on market efficiency.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CapitalAllocatorConfig:
    """Configuration for the CapitalAllocator."""
    reallocation_interval_hours: int = 4
    volatility_lookback_days: int = 7
    min_allocation_pct: float = 0.05  # Min 5% allocation per pair
    max_allocation_pct: float = 0.25  # Max 25% allocation per pair


class CapitalAllocator:
    """
    Dynamically allocates capital across a portfolio of assets.
    """

    def __init__(self, config: CapitalAllocatorConfig | None = None):
        """
        Initialize CapitalAllocator.

        Args:
            config: Configuration object.
        """
        self.config = config or CapitalAllocatorConfig()
        self._allocations: Dict[str, float] = {}

    def calculate_allocations(self, pair_data: Dict[str, pd.DataFrame], fees: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate new capital allocations.

        Args:
            pair_data: Dictionary mapping pair to its OHLCV dataframe.
            fees: Dictionary mapping pair to its taker fee.

        Returns:
            A dictionary mapping pair to its new capital allocation percentage.
        """
        efficiency_scores = {}

        for pair, df in pair_data.items():
            if df.empty or len(df) < 2:
                continue

            # 1. Calculate Realized Volatility
            returns = df["close"].pct_change().dropna()
            # Annualized volatility
            annualized_vol = returns.std() * (365**0.5) # Assuming daily data for now

            # 2. Get Fee
            fee = fees.get(pair, 0.001) # Default fee if not found

            # 3. Calculate Efficiency Ratio
            # Add a small epsilon to fee to avoid division by zero
            efficiency_scores[pair] = annualized_vol / (fee + 1e-9)

        if not efficiency_scores:
            return {}

        # Rank pairs by efficiency
        total_score = sum(efficiency_scores.values())
        if total_score == 0:
            # If all scores are 0, allocate equally
            num_pairs = len(efficiency_scores)
            return {pair: 1.0 / num_pairs for pair in efficiency_scores}

        # Normalize scores to get allocations
        allocations = {pair: score / total_score for pair, score in efficiency_scores.items()}
        
        # Clip allocations to min/max and re-normalize
        clipped_allocations = {
            pair: max(self.config.min_allocation_pct, min(self.config.max_allocation_pct, alloc))
            for pair, alloc in allocations.items()
        }
        
        # Re-normalize to ensure they sum to 1.0
        total_clipped = sum(clipped_allocations.values())
        final_allocations = {pair: alloc / total_clipped for pair, alloc in clipped_allocations.items()}

        self._allocations = final_allocations
        logger.info(f"New capital allocations calculated: {final_allocations}")
        return final_allocations

    def get_allocation(self, pair: str) -> float | None:
        """Get the current allocation for a specific pair."""
        return self._allocations.get(pair)

