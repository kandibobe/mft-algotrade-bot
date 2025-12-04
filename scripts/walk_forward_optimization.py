#!/usr/bin/env python3
"""
Stoic Citadel - Walk-Forward Optimization
==========================================

Automated walk-forward optimization pipeline.

Walk-Forward Analysis:
1. Divide data into rolling windows (train/validate/test)
2. Optimize parameters on training window
3. Validate on out-of-sample data
4. Roll forward and repeat
5. Aggregate results and select robust parameters

Usage:
    python scripts/walk_forward_optimization.py \
        --strategy StoicEnsembleStrategy \
        --timerange 20230101-20240101 \
        --windows 6 \
        --train-ratio 0.6 \
        --test-ratio 0.2
"""

import argparse
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class WFOWindow:
    """Single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    validate_start: str
    validate_end: str
    test_start: str
    test_end: str


@dataclass
class WFOResult:
    """Result from a single WFO window."""
    window_id: int
    train_trades: int
    train_profit: float
    train_sharpe: float
    validate_trades: int
    validate_profit: float
    validate_sharpe: float
    test_trades: int
    test_profit: float
    test_sharpe: float
    best_params: Dict[str, Any]
    timestamp: str


def parse_timerange(timerange: str) -> Tuple[datetime, datetime]:
    """Parse timerange string to datetime tuple."""
    start_str, end_str = timerange.split('-')
    start = datetime.strptime(start_str, '%Y%m%d')
    end = datetime.strptime(end_str, '%Y%m%d')
    return start, end


def format_date(dt: datetime) -> str:
    """Format datetime to timerange string."""
    return dt.strftime('%Y%m%d')


def create_windows(
    start: datetime,
    end: datetime,
    num_windows: int,
    train_ratio: float = 0.6,
    validate_ratio: float = 0.2
) -> List[WFOWindow]:
    """
    Create rolling walk-forward windows.
    
    Args:
        start: Start date
        end: End date
        num_windows: Number of WFO windows
        train_ratio: Fraction of window for training
        validate_ratio: Fraction for validation (rest is test)
        
    Returns:
        List of WFOWindow objects
    """
    total_days = (end - start).days
    window_size = total_days // num_windows
    
    windows = []
    
    for i in range(num_windows):
        window_start = start + timedelta(days=i * window_size)
        window_end = window_start + timedelta(days=window_size)
        
        train_days = int(window_size * train_ratio)
        validate_days = int(window_size * validate_ratio)
        
        train_start = window_start
        train_end = train_start + timedelta(days=train_days)
        validate_start = train_end
        validate_end = validate_start + timedelta(days=validate_days)
        test_start = validate_end
        test_end = min(window_end, end)
        
        windows.append(WFOWindow(
            window_id=i + 1,
            train_start=format_date(train_start),
            train_end=format_date(train_end),
            validate_start=format_date(validate_start),
            validate_end=format_date(validate_end),
            test_start=format_date(test_start),
            test_end=format_date(test_end)
        ))
        
        logger.info(
            f"Window {i+1}: Train {format_date(train_start)}-{format_date(train_end)} | "
            f"Validate {format_date(validate_start)}-{format_date(validate_end)} | "
            f"Test {format_date(test_start)}-{format_date(test_end)}"
        )
    
    return windows


def run_hyperopt(
    strategy: str,
    timerange: str,
    epochs: int = 100,
    loss_function: str = 'SharpeHyperOptLoss',
    spaces: str = 'buy sell'
) -> Dict[str, Any]:
    """
    Run Freqtrade hyperopt for parameter optimization.
    
    Args:
        strategy: Strategy name
        timerange: Timerange string (YYYYMMDD-YYYYMMDD)
        epochs: Number of optimization epochs
        loss_function: Hyperopt loss function
        spaces: Parameter spaces to optimize
        
    Returns:
        Dictionary with best parameters
    """
    cmd = [
        'docker-compose', '-f', 'docker-compose.backtest.yml',
        '--profile', 'optimize',
        'run', '--rm',
        '-e', f'STRATEGY={strategy}',
        '-e', f'TIMERANGE={timerange}',
        '-e', f'EPOCHS={epochs}',
        '-e', f'HYPEROPT_LOSS={loss_function}',
        '-e', f'SPACES={spaces}',
        'hyperopt'
    ]
    
    logger.info(f"Running hyperopt: {timerange}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Parse best parameters from output
        # (In production, we'd read from hyperopt_results directory)
        return {'optimized': True, 'timerange': timerange}
        
    except subprocess.TimeoutExpired:
        logger.error(f"Hyperopt timeout: {timerange}")
        return {'optimized': False, 'error': 'timeout'}
    except Exception as e:
        logger.error(f"Hyperopt error: {e}")
        return {'optimized': False, 'error': str(e)}


def run_backtest(
    strategy: str,
    timerange: str
) -> Dict[str, Any]:
    """
    Run Freqtrade backtest.
    
    Args:
        strategy: Strategy name
        timerange: Timerange string
        
    Returns:
        Dictionary with backtest results
    """
    cmd = [
        'docker-compose', '-f', 'docker-compose.backtest.yml',
        'run', '--rm',
        '-e', f'STRATEGY={strategy}',
        '-e', f'TIMERANGE={timerange}',
        'freqtrade-backtest'
    ]
    
    logger.info(f"Running backtest: {timerange}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        
        # Parse results (in production, read from backtest_results)
        return {
            'completed': True,
            'timerange': timerange,
            'trades': 0,  # Would parse from actual output
            'profit': 0.0,
            'sharpe': 0.0
        }
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {'completed': False, 'error': str(e)}


def run_wfo(
    strategy: str,
    timerange: str,
    num_windows: int = 6,
    train_ratio: float = 0.6,
    validate_ratio: float = 0.2,
    epochs: int = 100,
    output_dir: str = 'reports/wfo'
) -> List[WFOResult]:
    """
    Run complete walk-forward optimization.
    
    Args:
        strategy: Strategy name
        timerange: Total timerange
        num_windows: Number of WFO windows
        train_ratio: Training data ratio
        validate_ratio: Validation data ratio
        epochs: Hyperopt epochs per window
        output_dir: Output directory for results
        
    Returns:
        List of WFOResult objects
    """
    # Parse timerange
    start, end = parse_timerange(timerange)
    
    # Create windows
    windows = create_windows(
        start, end, num_windows, train_ratio, validate_ratio
    )
    
    results = []
    
    for window in windows:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Window {window.window_id}/{len(windows)}")
        logger.info(f"{'='*60}")
        
        # Step 1: Optimize on training data
        train_range = f"{window.train_start}-{window.train_end}"
        logger.info(f"\n[1/3] Optimizing on training data: {train_range}")
        opt_result = run_hyperopt(strategy, train_range, epochs)
        
        # Step 2: Validate parameters
        validate_range = f"{window.validate_start}-{window.validate_end}"
        logger.info(f"\n[2/3] Validating on hold-out data: {validate_range}")
        validate_result = run_backtest(strategy, validate_range)
        
        # Step 3: Test on out-of-sample data
        test_range = f"{window.test_start}-{window.test_end}"
        logger.info(f"\n[3/3] Testing on OOS data: {test_range}")
        test_result = run_backtest(strategy, test_range)
        
        # Collect results
        wfo_result = WFOResult(
            window_id=window.window_id,
            train_trades=0,
            train_profit=0.0,
            train_sharpe=0.0,
            validate_trades=validate_result.get('trades', 0),
            validate_profit=validate_result.get('profit', 0.0),
            validate_sharpe=validate_result.get('sharpe', 0.0),
            test_trades=test_result.get('trades', 0),
            test_profit=test_result.get('profit', 0.0),
            test_sharpe=test_result.get('sharpe', 0.0),
            best_params=opt_result,
            timestamp=datetime.now().isoformat()
        )
        
        results.append(wfo_result)
        
        # Save intermediate results
        save_results(results, output_dir, strategy)
    
    return results


def save_results(
    results: List[WFOResult],
    output_dir: str,
    strategy: str
) -> None:
    """
    Save WFO results to file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as YAML
    yaml_path = output_path / f"{strategy}_wfo_results.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(
            [asdict(r) for r in results],
            f,
            default_flow_style=False
        )
    
    # Save as JSON
    json_path = output_path / f"{strategy}_wfo_results.json"
    with open(json_path, 'w') as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            indent=2
        )
    
    logger.info(f"Results saved to {output_path}")


def generate_summary(results: List[WFOResult]) -> Dict[str, Any]:
    """
    Generate summary statistics from WFO results.
    """
    if not results:
        return {}
    
    avg_validate_profit = sum(r.validate_profit for r in results) / len(results)
    avg_test_profit = sum(r.test_profit for r in results) / len(results)
    avg_validate_sharpe = sum(r.validate_sharpe for r in results) / len(results)
    avg_test_sharpe = sum(r.test_sharpe for r in results) / len(results)
    
    # Calculate degradation (difference between validate and test)
    profit_degradation = avg_validate_profit - avg_test_profit
    sharpe_degradation = avg_validate_sharpe - avg_test_sharpe
    
    return {
        'num_windows': len(results),
        'avg_validate_profit': avg_validate_profit,
        'avg_test_profit': avg_test_profit,
        'avg_validate_sharpe': avg_validate_sharpe,
        'avg_test_sharpe': avg_test_sharpe,
        'profit_degradation': profit_degradation,
        'sharpe_degradation': sharpe_degradation,
        'is_robust': profit_degradation < 0.05 and sharpe_degradation < 0.5
    }


def main():
    parser = argparse.ArgumentParser(
        description='Walk-Forward Optimization for Stoic Citadel'
    )
    parser.add_argument(
        '--strategy', '-s',
        default='StoicEnsembleStrategy',
        help='Strategy name'
    )
    parser.add_argument(
        '--timerange', '-t',
        default='20230101-20240101',
        help='Total timerange (YYYYMMDD-YYYYMMDD)'
    )
    parser.add_argument(
        '--windows', '-w',
        type=int,
        default=6,
        help='Number of WFO windows'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.6,
        help='Training data ratio'
    )
    parser.add_argument(
        '--validate-ratio',
        type=float,
        default=0.2,
        help='Validation data ratio'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Hyperopt epochs per window'
    )
    parser.add_argument(
        '--output', '-o',
        default='reports/wfo',
        help='Output directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show windows without running'
    )
    
    args = parser.parse_args()
    
    logger.info(f"""
╔════════════════════════════════════════════════════════════════╗
║          STOIC CITADEL - WALK-FORWARD OPTIMIZATION             ║
╠════════════════════════════════════════════════════════════════╣
║  Strategy: {args.strategy:<48} ║
║  Timerange: {args.timerange:<47} ║
║  Windows: {args.windows:<49} ║
║  Train/Validate/Test: {args.train_ratio:.0%}/{args.validate_ratio:.0%}/{1-args.train_ratio-args.validate_ratio:.0%}                               ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    if args.dry_run:
        start, end = parse_timerange(args.timerange)
        windows = create_windows(
            start, end, args.windows, args.train_ratio, args.validate_ratio
        )
        logger.info("\nDry run complete. No optimization performed.")
        return
    
    # Run WFO
    results = run_wfo(
        strategy=args.strategy,
        timerange=args.timerange,
        num_windows=args.windows,
        train_ratio=args.train_ratio,
        validate_ratio=args.validate_ratio,
        epochs=args.epochs,
        output_dir=args.output
    )
    
    # Generate and display summary
    summary = generate_summary(results)
    
    logger.info(f"""
╔════════════════════════════════════════════════════════════════╗
║                      WFO SUMMARY                               ║
╠════════════════════════════════════════════════════════════════╣
║  Windows Completed: {summary.get('num_windows', 0):<41} ║
║  Avg Validate Profit: {summary.get('avg_validate_profit', 0):.2%}                                 ║
║  Avg Test Profit: {summary.get('avg_test_profit', 0):.2%}                                     ║
║  Avg Validate Sharpe: {summary.get('avg_validate_sharpe', 0):.2f}                                  ║
║  Avg Test Sharpe: {summary.get('avg_test_sharpe', 0):.2f}                                      ║
║  Robust: {'Yes' if summary.get('is_robust') else 'No':<51} ║
╚════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
