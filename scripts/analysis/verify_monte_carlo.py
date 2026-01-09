
import pandas as pd
import logging
import os
from src.backtesting.vectorized_backtester import VectorizedBacktester, BacktestConfig
from src.analysis.monte_carlo import MonteCarloSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MonteCarloIntegrationCheck")

def verify_monte_carlo():
    logger.info("Verifying Monte Carlo Integration...")
    
    # 1. Simulate trade results (profit_ratio)
    # In a real scenario, these come from VectorizedBacktester
    data = {
        'profit_ratio': [0.01, -0.005, 0.02, 0.015, -0.01, 0.03, -0.002, 0.01, 0.005, -0.008] * 10
    }
    trades_df = pd.DataFrame(data)
    
    # 2. Initialize Simulator
    simulator = MonteCarloSimulator(
        trades_df=trades_df,
        iterations=500,
        initial_capital=1000.0,
        max_drawdown_limit=0.20
    )
    
    # 3. Run Simulation
    simulator.run(noise_distribution='t', noise_std=0.002)
    
    # 4. Check results
    summary = simulator.get_summary()
    logger.info(f"Prob. of Ruin: {summary['probability_of_ruin']:.2f}%")
    logger.info(f"Mean Max Drawdown: {summary['mean_max_drawdown']:.2%}")
    logger.info(f"95th Percentile Drawdown: {summary['95th_percentile_drawdown']:.2%}")
    
    # 5. Export plot
    plot_path = "user_data/plot/monte_carlo_test.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    simulator.plot_equity_curves(num_curves_to_plot=50, output_path=plot_path)
    
    if os.path.exists(plot_path):
        logger.info(f"✅ Monte Carlo plot successfully generated at {plot_path}")
    else:
        logger.error("❌ Monte Carlo plot was not generated!")

if __name__ == "__main__":
    verify_monte_carlo()
