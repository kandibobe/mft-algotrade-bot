#!/usr/bin/env python3
"""
Stoic Citadel Management CLI
============================

Unified entry point for all operations.

Usage:
    python manage.py train --pairs BTC/USDT ETH/USDT
    python manage.py backtest
    python manage.py optimize
"""

import argparse
import sys
import logging
import subprocess
import os
import tempfile
import atexit
from pathlib import Path

# Configure structured logging
from src.utils.logger import setup_structured_logging, get_logger

setup_structured_logging(enable_console=True)
logger = get_logger("manage")

def setup_path():
    """Ensure src is in python path."""
    root = Path(__file__).parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

setup_path()

def generate_temp_config():
    """Generate a temporary config file for Freqtrade from Unified Config."""
    from src.config.manager import ConfigurationManager
    
    fd, path = tempfile.mkstemp(suffix=".json", prefix="freqtrade_config_")
    os.close(fd) # Close file descriptor, we just need path
    
    ConfigurationManager.export_freqtrade_config(path)
    logger.info(f"Generated temporary config at {path}")
    
    # Ensure cleanup on exit
    atexit.register(lambda: os.unlink(path) if os.path.exists(path) else None)
    
    return path

def train_command(args):
    """Execute model training."""
    from src.ml.training.train import train_model
    
    logger.info(f"Starting training for {args.pairs}")
    
    results = train_model(
        pairs=args.pairs,
        days=args.days,
        target=args.target,
        timeframe=args.timeframe,
        optimize=args.optimize,
        quick=args.quick,
        n_trials=args.trials
    )

    if args.backtest:
        logger.info("\n" + "="*70)
        logger.info("🔄 CLOSING THE LOOP: RUNNING BACKTEST")
        logger.info("="*70)
        
        # Check if any training was successful
        if any(r.get('success') for r in results.values()):
            # Reuse backtest command logic
            args.strategy = "StoicEnsembleStrategyV4" # Default or inferred
            args.timerange = None # Default
            backtest_command(args)
        else:
            logger.warning("❌ Training failed, skipping backtest.")

def optimize_command(args):
    """Execute nightly optimization."""
    from src.ops.optimization import NightlyOptimizer
    
    # Generate temp config for Freqtrade hyperopt
    config_path = generate_temp_config()
    
    # Patch NightlyOptimizer to use this config (needs update in optimization.py too)
    # For now, we assume optimization.py will be updated or we rely on it using the generated config.
    # Actually optimization.py likely hardcodes config path. We must update it.
    
    optimizer = NightlyOptimizer()
    optimizer.execute_nightly_cycle(
        strategy=args.strategy,
        pairs=args.pairs,
        epochs=args.epochs,
        ml_trials=args.trials
    )

def backtest_command(args):
    """Execute backtest."""
    config_path = generate_temp_config()
    
    cmd = [
        "freqtrade", "backtesting",
        "--strategy", args.strategy,
        "--timeframe", args.timeframe,
        "--config", config_path
    ]
    
    if args.timerange:
        cmd.extend(["--timerange", args.timerange])
        
    if args.pairs:
        cmd.extend(["--pairs"] + args.pairs)
        
    logger.info(f"Running backtest: {' '.join(cmd)}")
    subprocess.run(cmd)

def trade_command(args):
    """Execute trading mode (Live or Dry-Run)."""
    config_path = generate_temp_config()
    
    # Log Mode
    from src.config.manager import ConfigurationManager
    cfg = ConfigurationManager.get_config()
    if cfg.dry_run:
        logger.info("🟢 STARTING IN DRY-RUN MODE")
    else:
        logger.warning("🔴 STARTING IN LIVE TRADING MODE")
    
    cmd = [
        "freqtrade", "trade",
        "--strategy", args.strategy,
        "--config", config_path
    ]
    
    logger.info(f"Starting Freqtrade: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Stoic Citadel Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_parser.add_argument("--pairs", nargs="+", default=["BTC/USDT", "ETH/USDT"], help="Pairs to train")
    train_parser.add_argument("--days", type=int, help="Days of history to use")
    train_parser.add_argument("--target", help="Target variable override")
    train_parser.add_argument("--timeframe", default="5m", help="Timeframe")
    train_parser.add_argument("--quick", action="store_true", help="Quick mode (less data)")
    train_parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    train_parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    train_parser.add_argument("--backtest", action="store_true", help="Run backtest after training")
    
    # OPTIMIZE
    opt_parser = subparsers.add_parser("optimize", help="Run nightly optimization")
    opt_parser.add_argument("--strategy", default="StoicEnsembleStrategyV4", help="Strategy class name")
    opt_parser.add_argument("--pairs", nargs="+", default=["BTC/USDT", "ETH/USDT"], help="Pairs to optimize")
    opt_parser.add_argument("--epochs", type=int, default=100, help="Hyperopt epochs")
    opt_parser.add_argument("--trials", type=int, default=50, help="ML optimization trials")

    # BACKTEST
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--strategy", default="StoicEnsembleStrategyV4", help="Strategy class name")
    bt_parser.add_argument("--timeframe", default="5m", help="Timeframe")
    bt_parser.add_argument("--timerange", help="Timerange (e.g. 20240101-)")
    bt_parser.add_argument("--pairs", nargs="+", help="Override pairs")
    
    # TRADE
    trade_parser = subparsers.add_parser("trade", help="Run trading bot")
    trade_parser.add_argument("--strategy", default="StoicEnsembleStrategyV4", help="Strategy class name")
    trade_parser.add_argument("--dry-run", action="store_true", help="Force dry-run (override config)")
    trade_parser.add_argument("--live", action="store_true", help="Force live mode (override config)")

    # VERIFY
    verify_parser = subparsers.add_parser("verify", help="Verify mode consistency (Dry-Run Replay)")

    # VERIFY-DRIFT
    subparsers.add_parser("verify-drift", help="Verify drift between backtest and live")

    # CONVERT-DATA
    convert_parser = subparsers.add_parser("convert-data", help="Convert Freqtrade data to .feather")
    convert_parser.add_argument("--dir", default="user_data/data/binance", help="Data directory")

    args = parser.parse_args()
    
    # Initialize Configuration Manager
    try:
        from src.config.manager import ConfigurationManager
        # Initialize singleton
        config = ConfigurationManager.initialize()
        
        # Override from CLI args if present
        if hasattr(args, 'dry_run') and args.dry_run:
            config.dry_run = True
        if hasattr(args, 'live') and args.live:
            config.dry_run = False
            
    except Exception as e:
        logger.error(f"Configuration initialization failed: {e}")
        sys.exit(1)

    if args.command == "train":
        train_command(args)
    elif args.command == "optimize":
        optimize_command(args)
    elif args.command == "backtest":
        backtest_command(args)
    elif args.command == "trade":
        trade_command(args)
    elif args.command == "verify":
        verify_command(args)
    elif args.command == "verify-drift":
        verify_drift_command(args)
    elif args.command == "convert-data":
        convert_data_command(args)
    else:
        parser.print_help()

def verify_drift_command(args):
    """Execute reality check (drift analysis)."""
    from src.analysis.reality_check import run_reality_check
    run_reality_check()

def convert_data_command(args):
    """Execute data conversion."""
    from src.data.convert_freqtrade import convert_freqtrade_data
    convert_freqtrade_data(data_dir=args.dir)

def verify_command(args):
    """Execute consistency verification."""
    import asyncio
    
    async def _run_replay():
        from src.order_manager.smart_order_executor import SmartOrderExecutor
        from src.order_manager.smart_order import SmartOrder
        from src.websocket.aggregator import DataAggregator, AggregatedTicker
        from src.order_manager.order_types import OrderSide, OrderType
        
        logger.info("🔄 Starting Execution Replay (Dry-Run)...")
        
        # 1. Setup
        aggregator = DataAggregator()
        executor = SmartOrderExecutor(aggregator=aggregator, dry_run=True)
        await executor.start()
        
        # 2. Simulate a Buy Signal
        symbol = "BTC/USDT"
        logger.info(f"Simulating BUY signal for {symbol} at $100.0")
        
        order = SmartOrder(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=100.0
        )
        
        oid = await executor.submit_order(order)
        logger.info(f"Submitted Order {oid}")
        
        # 3. Feed Price Updates
        # Scenario: Price is 100.5, then drops to 99.9 (crossing limit)
        prices = [100.5, 100.3, 100.1, 99.9] 
        
        for p in prices:
            # Mock Ticker
            ticker = AggregatedTicker(
                symbol=symbol,
                best_bid=p - 0.05,
                best_ask=p + 0.05,
                best_bid_exchange="binance",
                best_ask_exchange="binance",
                spread=0.1,
                exchanges={},
                vwap=p,
                timestamp=0,
                total_volume_24h=1000,
                spread_pct=0.001
            )
            
            # Inject into executor via private method (direct injection for testing)
            await executor._process_ticker_update(ticker)
            
            # Allow event loop to process
            await asyncio.sleep(0.1)
            
            status = executor.get_order_status(oid)
            # Access internal order to check filled quantity
            internal_order = executor._active_orders.get(oid)
            filled = internal_order.filled_quantity if internal_order else "N/A"
            
            logger.info(f"  Ticker: {p} | Order Status: {status} | Filled: {filled}")
            
            if str(status) == "OrderStatus.FILLED":
                logger.info("  ✅ Order Filled!")
                break
        
        await executor.stop()
        logger.info("✅ Execution Replay Completed")

    asyncio.run(_run_replay())

if __name__ == "__main__":
    main()
