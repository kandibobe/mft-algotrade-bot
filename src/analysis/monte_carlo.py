"""
Monte Carlo Simulation for Trading Strategy
===========================================

Analyzes strategy robustness by simulating 1000+ variations of trade sequences.
Calculates Probability of Ruin and Confidence Intervals for Drawdown.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from scipy.stats import t

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Runs a Monte Carlo simulation on a series of trades to test robustness.
    """

    def __init__(self, trades_df: pd.DataFrame, iterations: int = 1000, initial_capital: float = 10000.0, max_drawdown_limit: float = 0.5):
        self.trades_df = trades_df
        self.iterations = iterations
        self.initial_capital = initial_capital
        self.max_drawdown_limit = max_drawdown_limit
        self.results = None

    def run(self, noise_distribution: str = 't', noise_std: float = 0.001):
        """
        Run the Monte Carlo simulation.

        Args:
            noise_distribution: 'normal' or 't' for the noise distribution.
            noise_std: Standard deviation of the noise to add to returns.
        """
        logger.info(f"Starting Monte Carlo Simulation ({self.iterations} iterations)...")
        
        profits = self.trades_df['profit_ratio'].values
        
        simulation_results = []
        ruin_count = 0
        all_equity_curves = []

        for i in range(self.iterations):
            shuffled_profits = np.random.permutation(profits)
            
            if noise_distribution == 't':
                # Student's t-distribution with 5 degrees of freedom (for fat tails)
                noise = t.rvs(df=5, scale=noise_std, size=len(profits))
            else: # 'normal'
                noise = np.random.normal(0, noise_std, size=len(profits))
                
            simulated_profits = shuffled_profits + noise
            
            cumulative_returns = np.cumprod(1 + simulated_profits)
            equity_curve = self.initial_capital * cumulative_returns
            all_equity_curves.append(equity_curve)
            
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_dd = abs(np.min(drawdown))
            
            if max_dd > self.max_drawdown_limit:
                ruin_count += 1
                
            total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
            simulation_results.append({
                'max_drawdown': max_dd,
                'total_return': total_return
            })
            
        self.results = pd.DataFrame(simulation_results)
        self.prob_ruin = (ruin_count / self.iterations) * 100
        self.all_equity_curves = all_equity_curves

        logger.info("Monte Carlo simulation completed.")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the simulation results.
        """
        if self.results is None:
            raise RuntimeError("Simulation has not been run yet.")

        dd_95 = self.results['max_drawdown'].quantile(0.95)
        dd_99 = self.results['max_drawdown'].quantile(0.99)
        dd_mean = self.results['max_drawdown'].mean()

        return {
            "iterations": self.iterations,
            "trades_per_iteration": len(self.trades_df),
            "probability_of_ruin": self.prob_ruin,
            "mean_max_drawdown": dd_mean,
            "95th_percentile_drawdown": dd_95,
            "99th_percentile_drawdown": dd_99,
        }

    def plot_equity_curves(self, num_curves_to_plot: int = 100, output_path: str = None):
        """
        Plot the equity curves from the simulation.

        Args:
            num_curves_to_plot: The number of simulated equity curves to plot.
            output_path: If provided, save the plot to this file.
        """
        if self.results is None:
            raise RuntimeError("Simulation has not been run yet.")

        plt.figure(figsize=(12, 8))
        
        # Plot a subset of the curves
        indices_to_plot = np.random.choice(len(self.all_equity_curves), size=num_curves_to_plot, replace=False)
        for i in indices_to_plot:
            plt.plot(self.all_equity_curves[i], color='gray', alpha=0.2)

        # Plot the median curve
        median_curve = np.median(self.all_equity_curves, axis=0)
        plt.plot(median_curve, color='blue', linewidth=2, label='Median Outcome')
        
        plt.title(f'Monte Carlo Simulation of Equity Curves ({self.iterations} iterations)')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()
