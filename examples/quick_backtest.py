#!/usr/bin/env python3
"""
Quick Start Example - Test Backtest Engine
===========================================

Простой пример для быстрого тестирования backtest с синтетическими данными.

Использование:
    python examples/quick_backtest.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest import BacktestEngine, BacktestConfig
from src.ml.training.labeling import TripleBarrierLabeler, TripleBarrierConfig

def generate_synthetic_data(n_candles=1000, trend='sideways'):
    """
    Generate synthetic OHLCV data for testing.

    Args:
        n_candles: Number of candles to generate
        trend: 'up', 'down', or 'sideways'

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Generate timestamps
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_candles),
        periods=n_candles,
        freq='1H'
    )

    # Generate prices with trend
    base_price = 50000.0
    prices = [base_price]

    for i in range(n_candles - 1):
        # Add trend
        if trend == 'up':
            drift = np.random.uniform(0.0001, 0.001)  # 0.01-0.1% up
        elif trend == 'down':
            drift = np.random.uniform(-0.001, -0.0001)  # 0.01-0.1% down
        else:  # sideways
            drift = np.random.uniform(-0.0005, 0.0005)  # -0.05% to +0.05%

        # Add volatility
        volatility = np.random.normal(0, 0.002)  # 0.2% std dev

        new_price = prices[-1] * (1 + drift + volatility)
        prices.append(new_price)

    prices = np.array(prices)

    # Generate OHLCV
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # High/Low around close with some variance
        high_var = abs(np.random.normal(0, 0.003))  # 0.3% variance
        low_var = abs(np.random.normal(0, 0.003))

        high = close * (1 + high_var)
        low = close * (1 - low_var)
        open_price = prices[i-1] if i > 0 else close

        volume = np.random.uniform(100, 1000)

        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    return df


def run_quick_backtest():
    """Run quick backtest with synthetic data."""

    print("🚀 Quick Backtest Example")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n📊 Generating synthetic data...")
    data = generate_synthetic_data(n_candles=1000, trend='sideways')
    print(f"   Generated {len(data)} candles")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")

    # 2. Configure backtest
    print("\n⚙️  Configuring backtest...")
    config = BacktestConfig({
        'initial_capital': 10000.0,
        'maker_fee': 0.001,  # 0.1% maker fee
        'taker_fee': 0.001,  # 0.1% taker fee
        'position_size_pct': 0.1,  # 10% per trade
        'take_profit': 0.02,  # 2% TP
        'stop_loss': 0.01,  # 1% SL
        'slippage_model': 'fixed',  # Use fixed slippage (no SlippageSimulator needed)
    })
    print(f"   Initial capital: ${config.initial_capital:,.2f}")
    print(f"   Maker fee: {config.maker_fee:.2%}")
    print(f"   Position size: {config.position_size_pct:.0%} per trade")

    # 3. Run backtest
    print("\n🔄 Running backtest...")
    engine = BacktestEngine(config)

    try:
        results = engine.run_backtest(data)

        # 4. Display results
        print("\n" + "=" * 60)
        print("📈 BACKTEST RESULTS")
        print("=" * 60)

        # Show available results
        print(f"\nAvailable metrics: {list(results.keys())}")

        # Display metrics if available
        if 'total_return' in results:
            print(f"\n💰 P&L:")
            print(f"   Total Return:     {results.get('total_return', 0):>10.2%}")
            print(f"   Final Balance:    ${results.get('final_balance', config.initial_capital):>12,.2f}")
            print(f"   Total PnL:        ${results.get('total_pnl', 0):>12,.2f}")

        if 'sharpe_ratio' in results:
            print(f"\n📊 Performance Metrics:")
            print(f"   Sharpe Ratio:     {results.get('sharpe_ratio', 0):>10.2f}")
            print(f"   Sortino Ratio:    {results.get('sortino_ratio', 0):>10.2f}")
            print(f"   Max Drawdown:     {results.get('max_drawdown', 0):>10.2%}")
            print(f"   Calmar Ratio:     {results.get('calmar_ratio', 0):>10.2f}")

        if 'total_trades' in results:
            print(f"\n📈 Trading Statistics:")
            print(f"   Total Trades:     {results.get('total_trades', 0):>10}")
            print(f"   Winning Trades:   {results.get('winning_trades', 0):>10}")
            print(f"   Losing Trades:    {results.get('losing_trades', 0):>10}")
            print(f"   Win Rate:         {results.get('win_rate', 0):>10.2%}")
            print(f"   Profit Factor:    {results.get('profit_factor', 0):>10.2f}")
            print(f"   Avg Win:          ${results.get('avg_win', 0):>12,.2f}")
            print(f"   Avg Loss:         ${results.get('avg_loss', 0):>12,.2f}")

        # Check number of trades
        num_trades = results.get('total_trades', len(results.get('trades', [])))

        if num_trades > 0:
            print(f"\n✅ Backtest completed successfully!")
            print(f"   Generated {num_trades} trades")
        else:
            print(f"\n⚠️  Warning: No trades generated")
            print(f"   This might indicate:")
            print(f"   - Strategy too conservative")
            print(f"   - Not enough data")
            print(f"   - Labels generated no signals")

        # 5. Save report (if matplotlib available)
        try:
            print(f"\n💾 Generating visual report...")
            report_path = engine.generate_report(results, data, save_path="reports/quick_backtest.png")
            print(f"   Report saved to: {report_path}")
        except Exception as e:
            print(f"   ⚠️  Could not generate visual report: {e}")

        return results

    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STOIC CITADEL - Quick Backtest Example")
    print("  Test your setup with synthetic data")
    print("=" * 60)

    results = run_quick_backtest()

    if results:
        print("\n" + "=" * 60)
        print("✅ Setup verified! You can now:")
        print("   1. Run backtest with real data")
        print("   2. Test paper trading")
        print("   3. Customize strategies")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("❌ Setup needs attention - check errors above")
        print("=" * 60 + "\n")
