import numpy as np
import pandas as pd
import pytest

from src.backtesting.wfo_engine import BacktestConfig, VectorizedBacktester


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    # Create a predictable price path
    # Flat then up then down
    prices = np.ones(100) * 100.0

    # Bar 10: Signal Buy (Executes at Bar 11)
    # Bar 15: Price spikes to 102 (TP check)
    prices[15] = 102.0

    # Bar 50: Signal Buy (Executes at Bar 51)
    # Bar 55: Price drops to 98 (SL check)
    prices[50:60] = 100.0
    prices[55] = 98.0

    data = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5, # High is slightly higher
        'low': prices - 0.5,  # Low is slightly lower
        'close': prices,
        'volume': 1000
    }, index=dates)

    # Adjust high/low for the spike/drop
    data.loc[dates[15], 'high'] = 102.5
    data.loc[dates[55], 'low'] = 97.5

    return data

def test_take_profit(sample_ohlcv):
    # TP = 1.5% (1.015). Entry 100. TP = 101.5.
    # Bar 15 High is 102.5. Should hit TP.

    config = BacktestConfig(
        initial_capital=10000,
        fee_rate=0.0, # Zero fees for easier math
        slippage_rate=0.0,
        take_profit=0.015,
        stop_loss=0.05,
        max_holding_bars=20
    )
    bt = VectorizedBacktester(config)

    signals = pd.Series(0, index=sample_ohlcv.index)
    signals.iloc[10] = 1 # Buy signal at close of Bar 10

    results = bt.run(signals, sample_ohlcv)
    trades = results['trades']

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade['exit_reason'] == 'take_profit'
    assert trade['entry_price'] == 100.0 # Open of Bar 11
    assert abs(trade['exit_price'] - 101.5) < 1e-6 # Exact TP price (assuming limits fill)
    assert trade['gross_pnl'] > 0

def test_stop_loss(sample_ohlcv):
    # SL = 1.5% (0.985). Entry 100. SL = 98.5.
    # Bar 55 Low is 97.5. Should hit SL.

    config = BacktestConfig(
        initial_capital=10000,
        fee_rate=0.0,
        slippage_rate=0.0,
        take_profit=0.05,
        stop_loss=0.015,
        max_holding_bars=20
    )
    bt = VectorizedBacktester(config)

    signals = pd.Series(0, index=sample_ohlcv.index)
    signals.iloc[50] = 1 # Buy at close of Bar 50. Entry at Bar 51 Open (100).

    results = bt.run(signals, sample_ohlcv)
    trades = results['trades']

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade['exit_reason'] == 'stop_loss'
    assert trade['entry_price'] == 100.0
    assert abs(trade['exit_price'] - 98.5) < 1e-6 # SL price
    assert trade['gross_pnl'] < 0

def test_fees_and_slippage(sample_ohlcv):
    config = BacktestConfig(
        initial_capital=10000,
        fee_rate=0.001,      # 0.1%
        slippage_rate=0.001, # 0.1%
        take_profit=0.10,    # Won't hit
        stop_loss=0.10,      # Won't hit
        max_holding_bars=5   # Time exit
    )
    bt = VectorizedBacktester(config)

    signals = pd.Series(0, index=sample_ohlcv.index)
    signals.iloc[10] = 1 # Buy Bar 10 -> Enter Bar 11 (100) -> Exit Bar 16 (102 close)
    # Wait, Time limit 5 bars.
    # Entered at 11.
    # 11, 12, 13, 14, 15 (5 bars).
    # Should exit at close of 16? Or 15?
    # Logic: if (i - entry_idx) >= max_holding_bars
    # 16 - 11 = 5. >= 5. Exit at 16.

    results = bt.run(signals, sample_ohlcv)
    trades = results['trades']

    assert len(trades) == 1
    trade = trades.iloc[0]

    # Entry Price = Open(11) * (1 + slippage) = 100 * 1.001 = 100.1
    assert abs(trade['entry_price'] - 100.1) < 1e-6

    # Exit Price (Time limit) = Close(16) * (1 - slippage) = 102.0 * 0.999 = 101.898
    # Close at 16 is 100.0 (Wait, price spike was only at 15)
    # prices[15] = 102.0.
    # prices[16] = 100.0.
    # So exit price = 100.0 * 0.999 = 99.9

    # Correction: Bar 15 is index 15.
    # Entry at 11.
    # i=16. 16-11=5. Exit. Close[16]=100.
    assert abs(trade['exit_price'] - 99.9) < 1e-6

    # Fees
    # Entry Fee = Qty * EntryPrice * FeeRate
    # Exit Fee = Qty * ExitPrice * FeeRate
    # Total Fee = Sum
    expected_fees = (trade['quantity'] * trade['entry_price'] * 0.001) + \
                    (trade['quantity'] * trade['exit_price'] * 0.001)

    assert abs(trade['fees'] - expected_fees) < 1e-6
