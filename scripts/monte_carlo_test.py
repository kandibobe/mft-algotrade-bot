#!/usr/bin/env python3
"""
Monte Carlo Simulation for Trading Strategy
===========================================

Analyzes strategy robustness by simulating 1000+ variations of trade sequences.
Calculates Probability of Ruin and Confidence Intervals for Drawdown.

Usage:
    python scripts/monte_carlo_test.py --file user_data/backtest_results/backtest-result-....json
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from src.analysis.monte_carlo import MonteCarloSimulator

def load_backtest_data(filepath):
    """Load trades from Freqtrade backtest JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    trades = []
    if "strategy" in data:
        for strat_name, strat_data in data["strategy"].items():
            if "trades" in strat_data:
                trades.extend(strat_data["trades"])
                break
    elif "trades" in data:
        trades = data["trades"]
        
    if not trades:
        print("No trades found in file.")
        sys.exit(1)
        
    return pd.DataFrame(trades)

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation')
    parser.add_argument('--file', required=True, help='Backtest result JSON file')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of simulations')
    parser.add_argument('--ruin', type=float, default=0.5, help='Drawdown limit for Ruin (0.5 = 50%%)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curves')
    parser.add_argument('--output', type=str, default='monte_carlo.png', help='Output file for plot')
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"File not found: {args.file}")
        sys.exit(1)
        
    df = load_backtest_data(args.file)
    
    simulator = MonteCarloSimulator(df, iterations=args.iterations, max_drawdown_limit=args.ruin)
    simulator.run()
    
    summary = simulator.get_summary()
    
    print("\n" + "="*50)
    print("MONTE CARLO RESULTS")
    print("="*50)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("="*50)

    prob_ruin = summary['probability_of_ruin']
    dd_95 = summary['95th_percentile_drawdown']

    if prob_ruin > 5.0:
        print("❌ DANGER: Risk of Ruin is high (>5%). Reduce position size.")
    elif dd_95 > 0.35:
        print("⚠️ WARNING: 95% chance of >35% drawdown. Consider defensive protections.")
    else:
        print("✅ PASS: Strategy appears robust.")

    if args.plot:
        simulator.plot_equity_curves(output_path=args.output)

if __name__ == "__main__":
    main()
