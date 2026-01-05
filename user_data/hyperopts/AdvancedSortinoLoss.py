"""
AdvancedSortinoHyperOptLoss - Improved Loss Function for Stoic Citadel
=====================================================================

Optimizes for Sortino Ratio with a heavy penalty for Max Drawdown.
Ideal for MFT systems where capital preservation is paramount.
"""

from datetime import datetime
from typing import Any, Dict

from pandas import DataFrame
import numpy as np

from freqtrade.optimize.hyperopt import IHyperOptLoss


class AdvancedSortinoHyperOptLoss(IHyperOptLoss):
    """
    Advanced loss function for Hyperopt.
    """

    def hyperopt_loss_function(self, results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               *args, **kwargs) -> float:
        """
        Objective function for hyperopt to minimize.
        """
        if results.empty:
            return 100.0

        # 1. Basic Stats
        total_profit = results['profit_ratio'].sum()
        
        # Approximate daily returns for Sortino
        results['close_date_only'] = results['close_date'].dt.date
        daily_returns = results.groupby('close_date_only')['profit_ratio'].sum()
        
        # 2. Sortino Calculation
        mean_return = daily_returns.mean()
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 1e-9
        
        # Annualized Sortino
        sortino_ratio = (mean_return / downside_deviation) * np.sqrt(365) if downside_deviation > 0 else 0
        
        # 3. Max Drawdown Calculation
        # Calculate cumulative profit
        cumulative_profit = results['profit_ratio'].cumsum()
        peak = cumulative_profit.expanding(min_periods=1).max()
        drawdown = (peak - cumulative_profit)
        max_drawdown = drawdown.max()
        
        # 4. Penalties & Weighting
        # We want to minimize this value
        # Base loss is -Sortino
        loss = -sortino_ratio
        
        # Add Drawdown Penalty (Exponential penalty for drawdown > 10%)
        if max_drawdown > 0.10:
            loss += (max_drawdown * 50) # Heavy penalty
        else:
            loss += (max_drawdown * 10) # Light penalty
            
        # Statistical Significance Penalty
        min_trades = config.get('hyperopt_min_trades', 50)
        if trade_count < min_trades:
            loss += (min_trades - trade_count) * 2
            
        # Profitability Check
        if total_profit <= 0:
            loss += 20.0 # Huge penalty for negative total profit

        return float(loss)
