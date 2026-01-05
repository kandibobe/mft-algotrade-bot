"""
Vectorized Backtester
=====================

A high-performance, event-driven backtester for ML strategies.
Simulates realistic execution, fees, and slippage.

Key Features:
- Event-driven execution (Entry at Open of Next Bar)
- Realistic Fee & Slippage modeling (Deducted from every trade)
- Correct handling of Take Profit / Stop Loss / Time Exits
- Conservative execution assumptions (SL hits before TP in ambiguous cases)
- Fast iteration using Numpy
"""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np

try:
    import cudf
    from numba import jit
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 10000.0
    fee_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.001  # 0.1% (Increased for Chase-Limit conservatism)
    take_profit: float = 0.015  # 1.5% default
    stop_loss: float = 0.0075  # 0.75% default
    max_holding_bars: int = 24  # Max bars to hold
    position_size_pct: float = 0.99  # Use 99% of capital (leave dust)


@dataclass
class Trade:
    """Trade record."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    direction: int  # 1 for Long
    gross_pnl: float
    net_pnl: float
    fees: float
    exit_reason: str
    holding_bars: int


class VectorizedBacktester:
    """
    Backtester that iterates through signals and simulates trade lifecycles.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, signals: pd.Series, ohlcv: pd.DataFrame) -> dict[str, Any]:
        """
        Run backtest.

        Args:
            signals: Series of signals (1=Buy, 0=No Signal). Index must match ohlcv.
            ohlcv: DataFrame with open, high, low, close, volume.

        Returns:
            Dict with results (trades, equity_curve, metrics).
        """
        # Align signals and data
        common_idx = signals.index.intersection(ohlcv.index)
        if len(common_idx) == 0:
            raise ValueError("Signals and OHLCV have no overlapping timestamps")

        signals = signals.loc[common_idx]
        ohlcv = ohlcv.loc[common_idx]

        if CUDA_AVAILABLE:
            # Convert to cuDF for GPU acceleration
            signals_gpu = cudf.from_pandas(signals)
            ohlcv_gpu = cudf.from_pandas(ohlcv)
            
            # Prepare cuDF arrays for fast iteration
            opens = ohlcv_gpu["open"].to_numpy()
            highs = ohlcv_gpu["high"].to_numpy()
            lows = ohlcv_gpu["low"].to_numpy()
            closes = ohlcv_gpu["close"].to_numpy()
            times = ohlcv_gpu.index.to_numpy()
            sig_values = signals_gpu.to_numpy()
        else:
            # Prepare numpy arrays for fast iteration
            opens = ohlcv["open"].values
            highs = ohlcv["high"].values
            lows = ohlcv["low"].values
            closes = ohlcv["close"].values
            times = ohlcv.index
            sig_values = signals.values
        
        if CUDA_AVAILABLE:
            run_backtest_numba = self._create_numba_backtester()
            trades, equity_curve, final_capital = run_backtest_numba(
                opens, highs, lows, closes, sig_values,
                self.config.initial_capital, self.config.fee_rate, self.config.slippage_rate,
                self.config.take_profit, self.config.stop_loss, self.config.max_holding_bars,
                self.config.position_size_pct
            )
            trades = [Trade(pd.to_datetime(t[0]), pd.to_datetime(t[1]), t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10]) for t in trades]

        else:
            n_bars = len(closes)
            trades: list[Trade] = []
            equity_curve = [self.config.initial_capital]
            current_capital = self.config.initial_capital

            # State
            in_position = False
            entry_price = 0.0
            entry_idx = 0
            quantity = 0.0

            # Iterate through bars
            for i in range(n_bars - 1):
                # 1. Check Exit if in position
                if in_position:
                    tp_price = entry_price * (1 + self.config.take_profit)
                    sl_price = entry_price * (1 - self.config.stop_loss)

                    exit_price = 0.0
                    exit_reason = None

                    if lows[i] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        exit_price *= 1 - self.config.slippage_rate
                    elif highs[i] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        exit_price *= 1 - self.config.slippage_rate
                    elif (i - entry_idx) >= self.config.max_holding_bars:
                        exit_price = closes[i]
                        exit_reason = "time_limit"
                        exit_price *= 1 - self.config.slippage_rate

                    if exit_reason:
                        exit_fee = (quantity * exit_price) * self.config.fee_rate
                        gross_payout = quantity * exit_price
                        net_payout = gross_payout - exit_fee
                        current_capital += net_payout
                        gross_pnl = (exit_price - entry_price) * quantity
                        total_fees = (quantity * entry_price * self.config.fee_rate) + exit_fee
                        net_pnl = gross_pnl - total_fees
                        trades.append(
                            Trade(
                                entry_time=times[entry_idx],
                                exit_time=times[i],
                                entry_price=entry_price,
                                exit_price=exit_price,
                                quantity=quantity,
                                direction=1,
                                gross_pnl=gross_pnl,
                                net_pnl=net_pnl,
                                fees=total_fees,
                                exit_reason=exit_reason,
                                holding_bars=i - entry_idx,
                            )
                        )
                        in_position = False
                        quantity = 0.0

                if not in_position:
                    if sig_values[i] == 1:
                        if i + 1 < n_bars:
                            entry_idx = i + 1
                            raw_entry_price = opens[entry_idx]
                            entry_price = raw_entry_price * (1 + self.config.slippage_rate)
                            position_cost = current_capital * self.config.position_size_pct
                            quantity = position_cost / (entry_price * (1 + self.config.fee_rate))
                            cost_outflow = quantity * entry_price
                            entry_fee = cost_outflow * self.config.fee_rate
                            current_capital -= cost_outflow + entry_fee
                            in_position = True
                
                if in_position:
                    if entry_idx > i:
                        total_equity = current_capital + (quantity * entry_price)
                    else:
                        mtm_value = quantity * closes[i]
                        total_equity = current_capital + mtm_value
                else:
                    total_equity = current_capital
                equity_curve.append(total_equity)

            if in_position:
                i = n_bars - 1
                exit_price = closes[i] * (1 - self.config.slippage_rate)
                exit_fee = (quantity * exit_price) * self.config.fee_rate
                net_payout = (quantity * exit_price) - exit_fee
                current_capital += net_payout
                gross_pnl = (exit_price - entry_price) * quantity
                total_fees = (quantity * entry_price * self.config.fee_rate) + exit_fee
                net_pnl = gross_pnl - total_fees
                trades.append(
                    Trade(
                        entry_time=times[entry_idx],
                        exit_time=times[i],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=quantity,
                        direction=1,
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        fees=total_fees,
                        exit_reason="end_of_data",
                        holding_bars=i - entry_idx,
                    )
                )
                equity_curve[-1] = current_capital
            
            final_capital = current_capital
            
        return {
            "trades": pd.DataFrame(trades),
            "equity_curve": pd.Series(equity_curve, index=times[: len(equity_curve)]),
            "final_capital": final_capital,
            "total_return": (final_capital - self.config.initial_capital)
            / self.config.initial_capital,
        }

    def _create_numba_backtester(self):
        @jit(nopython=True)
        def run_backtest_numba(opens, highs, lows, closes, sig_values, initial_capital, fee_rate, slippage_rate, take_profit, stop_loss, max_holding_bars, position_size_pct):
            n_bars = len(closes)
            trades = []
            equity_curve = np.full(n_bars, initial_capital)
            current_capital = initial_capital

            in_position = False
            entry_price = 0.0
            entry_idx = 0
            quantity = 0.0

            for i in range(n_bars - 1):
                if in_position:
                    tp_price = entry_price * (1 + take_profit)
                    sl_price = entry_price * (1 - stop_loss)
                    exit_price = 0.0
                    exit_reason = ""

                    if lows[i] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        exit_price *= 1 - slippage_rate
                    elif highs[i] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        exit_price *= 1 - slippage_rate
                    elif (i - entry_idx) >= max_holding_bars:
                        exit_price = closes[i]
                        exit_reason = "time_limit"
                        exit_price *= 1 - slippage_rate

                    if exit_reason != "":
                        exit_fee = (quantity * exit_price) * fee_rate
                        net_payout = (quantity * exit_price) - exit_fee
                        current_capital += net_payout
                        gross_pnl = (exit_price - entry_price) * quantity
                        total_fees = (quantity * entry_price * fee_rate) + exit_fee
                        net_pnl = gross_pnl - total_fees
                        trades.append(
                            (
                                entry_idx, i, entry_price, exit_price, quantity, 1, gross_pnl, net_pnl, total_fees, exit_reason, i - entry_idx
                            )
                        )
                        in_position = False
                        quantity = 0.0

                if not in_position:
                    if sig_values[i] == 1:
                        if i + 1 < n_bars:
                            entry_idx = i + 1
                            raw_entry_price = opens[entry_idx]
                            entry_price = raw_entry_price * (1 + slippage_rate)
                            position_cost = current_capital * position_size_pct
                            quantity = position_cost / (entry_price * (1 + fee_rate))
                            cost_outflow = quantity * entry_price
                            entry_fee = cost_outflow * fee_rate
                            current_capital -= cost_outflow + entry_fee
                            in_position = True
                
                if in_position:
                    if entry_idx > i:
                        total_equity = current_capital + (quantity * entry_price)
                    else:
                        mtm_value = quantity * closes[i]
                        total_equity = current_capital + mtm_value
                else:
                    total_equity = current_capital
                equity_curve[i] = total_equity
            
            if in_position:
                i = n_bars - 1
                exit_price = closes[i] * (1 - slippage_rate)
                exit_fee = (quantity * exit_price) * fee_rate
                net_payout = (quantity * exit_price) - exit_fee
                current_capital += net_payout
                gross_pnl = (exit_price - entry_price) * quantity
                total_fees = (quantity * entry_price * fee_rate) + exit_fee
                net_pnl = gross_pnl - total_fees
                trades.append(
                    (
                        entry_idx, i, entry_price, exit_price, quantity, 1, gross_pnl, net_pnl, total_fees, "end_of_data", i - entry_idx
                    )
                )
                equity_curve[i] = current_capital
            
            return trades, equity_curve, current_capital
        return run_backtest_numba
