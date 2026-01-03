"""
SortinoHyperOptLoss - Custom Loss Function for Freqtrade Hyperopt
=================================================================

Optimizes for Sortino Ratio: (Mean Excess Return) / (Downside Deviation)
Sortino Ratio is better than Sharpe because it only penalizes downside volatility.

Formula:
    Loss = 1 - (Sortino Ratio / Scale Factor)

We normalize the Sortino Ratio to ensure the loss is reasonable.
"""

from datetime import datetime
from typing import Any, Dict

from pandas import DataFrame
import numpy as np

from freqtrade.optimize.hyperopt import IHyperOptLoss


class SortinoHyperOptLoss(IHyperOptLoss):
    """
    Custom loss function for Hyperopt which calculates the Sortino Ratio.
    """

    def hyperopt_loss_function(self, results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               *args, **kwargs) -> float:
        """
        Objective function for hyperopt to minimize.
        """
        
        # 1. Total Profit
        total_profit = results['profit_ratio'].sum()
        
        # 2. Daily Returns (Approximate)
        # We group by close date to approximate daily returns
        # Note: This is a simplification. For precise daily returns, we'd need resampling.
        # But for hyperopt speed, this is standard practice.
        results['close_date'] = results['close_date'].dt.date
        daily_returns = results.groupby('close_date')['profit_ratio'].sum()
        
        # 3. Calculate Sortino Components
        # Target Return (Risk Free Rate, usually 0 or small positive)
        target_return = 0.0
        
        # Excess Returns
        excess_returns = daily_returns - target_return
        
        # Average Daily Return
        mean_return = excess_returns.mean()
        
        # Downside Deviation (Standard deviation of negative returns only)
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns)
        else:
            # If no losing days, downside deviation is 0. 
            # To avoid division by zero, we use a small number.
            # Or effectively, the ratio is infinite (very good).
            downside_deviation = 1e-9
            
        # Avoid division by zero
        if downside_deviation == 0:
             downside_deviation = 1e-9
             
        # Annualization Factor (Crypto 24/7 = 365 days)
        # Although Sortino is often calculated on daily data, standardizing to annual is good for comparison.
        annualization_factor = np.sqrt(365)
        
        # Sortino Ratio
        sortino_ratio = (mean_return / downside_deviation) * annualization_factor
        
        # 4. Penalties
        # Penalty for too few trades (Statistical significance)
        min_trades = config.get('hyperopt_min_trades', 50)
        if trade_count < min_trades:
            # Heavy penalty
            return 100.0
            
        # Penalty for negative profit
        if total_profit < 0:
            # Return absolute value of loss + penalty
            return abs(total_profit) + 10.0
            
        # 5. Calculate Loss
        # We want to MAXIMIZE Sortino, so we MINIMIZE (1 - Sortino)
        # Or more robustly: -Sortino
        # However, Freqtrade expects a positive float usually, but minimizes it.
        # Common pattern: Return -Sortino.
        
        # To make it more comparable to other loss functions (like SharpeLoss), 
        # we can use a sigmoid or just negative.
        
        # Simple inversion:
        return -sortino_ratio
