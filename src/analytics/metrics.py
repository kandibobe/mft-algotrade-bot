#!/usr/bin/env python3
"""
Portfolio Performance Metrics
==============================

Comprehensive financial metrics calculations for portfolio analysis.

Metrics included:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Risk metrics (VaR, CVaR, volatility)
- Trade statistics

Author: Stoic Citadel Team
License: MIT
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Returns
    total_return: float
    annualized_return: float
    monthly_return: float
    daily_return: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Drawdown
    max_drawdown: float
    current_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int
    
    # Volatility
    annualized_volatility: float
    daily_volatility: float
    downside_volatility: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade_duration_hours: float
    
    # Other
    recovery_factor: float
    expectancy: float
    kelly_criterion: float
    
    def to_dict(self):
        return self.__dict__


@dataclass
class RiskMetrics:
    """Risk-focused metrics."""
    # Value at Risk
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    
    # Exposure
    gross_exposure: float
    net_exposure: float
    leverage: float
    
    # Concentration
    max_position_size: float
    herfindahl_index: float  # Concentration measure
    
    # Correlation
    beta: float
    correlation_to_benchmark: float
    
    # Tail risk
    skewness: float
    kurtosis: float
    
    def to_dict(self):
        return self.__dict__


class MetricsCalculator:
    """
    Calculate portfolio performance and risk metrics.
    
    Usage:
        calc = MetricsCalculator()
        
        # From equity curve
        equity_values = [10000, 10050, 10100, 10080, 10200, ...]
        metrics = calc.calculate_performance(equity_values)
        
        # From trades
        trades = [Trade(...), Trade(...), ...]
        trade_metrics = calc.calculate_trade_metrics(trades)
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,  # 5% annual risk-free rate
        benchmark_returns: Optional[List[float]] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns or []
    
    def calculate_performance(
        self,
        equity_values: List[float],
        trading_days_per_year: int = 365
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics from equity curve."""
        if len(equity_values) < 2:
            return self._empty_performance_metrics()
        
        equity = np.array(equity_values)
        returns = np.diff(equity) / equity[:-1]
        
        # Basic returns
        total_return = (equity[-1] - equity[0]) / equity[0] * 100
        n_days = len(returns)
        annualized_return = ((1 + total_return / 100) ** (trading_days_per_year / n_days) - 1) * 100
        
        # Daily and monthly returns
        daily_return = np.mean(returns) * 100 if len(returns) > 0 else 0
        monthly_return = daily_return * 30  # Approximate
        
        # Volatility
        daily_vol = np.std(returns) * 100 if len(returns) > 1 else 0
        annualized_vol = daily_vol * np.sqrt(trading_days_per_year)
        
        # Downside volatility (for Sortino)
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns) * np.sqrt(trading_days_per_year) * 100 if len(negative_returns) > 0 else 0
        
        # Drawdown analysis
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_dd = np.max(drawdown)
        current_dd = drawdown[-1]
        avg_dd = np.mean(drawdown)
        
        # Max drawdown duration
        max_dd_duration = self._calculate_max_dd_duration(equity)
        
        # Risk-adjusted ratios
        daily_rf = self.risk_free_rate / trading_days_per_year
        excess_returns = returns - daily_rf
        
        sharpe = self._calculate_sharpe(returns, self.risk_free_rate, trading_days_per_year)
        sortino = self._calculate_sortino(returns, self.risk_free_rate, trading_days_per_year)
        calmar = annualized_return / max_dd if max_dd > 0 else 0
        
        # Information ratio (vs benchmark)
        info_ratio = self._calculate_information_ratio(returns)
        
        # Recovery factor
        recovery = total_return / max_dd if max_dd > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            monthly_return=monthly_return,
            daily_return=daily_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=max_dd_duration,
            annualized_volatility=annualized_vol,
            daily_volatility=daily_vol,
            downside_volatility=downside_vol,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            best_trade=0,
            worst_trade=0,
            avg_trade_duration_hours=0,
            recovery_factor=recovery,
            expectancy=0,
            kelly_criterion=0
        )
    
    def calculate_trade_metrics(
        self,
        pnls: List[float],
        durations_hours: Optional[List[float]] = None
    ) -> dict:
        """Calculate trade-level statistics."""
        if not pnls:
            return {}
        
        pnl_array = np.array(pnls)
        wins = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array < 0]
        
        total = len(pnls)
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / total * 100 if total > 0 else 0
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = np.mean(pnl_array) if len(pnl_array) > 0 else 0
        
        # Kelly criterion
        if avg_loss > 0 and avg_win > 0:
            win_prob = win_rate / 100
            kelly = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            kelly = max(0, min(kelly, 1))  # Clamp to 0-100%
        else:
            kelly = 0
        
        avg_duration = np.mean(durations_hours) if durations_hours else 0
        
        return {
            "total_trades": total,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "best_trade": np.max(pnl_array) if len(pnl_array) > 0 else 0,
            "worst_trade": np.min(pnl_array) if len(pnl_array) > 0 else 0,
            "avg_trade_duration_hours": avg_duration,
            "expectancy": expectancy,
            "kelly_criterion": kelly * 100  # As percentage
        }
    
    def calculate_risk_metrics(
        self,
        returns: List[float],
        positions_value: List[float],
        total_equity: List[float]
    ) -> RiskMetrics:
        """Calculate risk metrics."""
        ret_array = np.array(returns)
        
        # VaR calculations
        var_95 = np.percentile(ret_array, 5) if len(ret_array) > 20 else 0
        var_99 = np.percentile(ret_array, 1) if len(ret_array) > 100 else 0
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(ret_array[ret_array <= var_95]) if len(ret_array[ret_array <= var_95]) > 0 else var_95
        
        # Exposure
        pos_array = np.array(positions_value)
        eq_array = np.array(total_equity)
        
        gross_exposure = np.mean(pos_array / eq_array) * 100 if len(eq_array) > 0 and np.all(eq_array > 0) else 0
        net_exposure = gross_exposure  # Simplified for long-only
        leverage = gross_exposure / 100
        
        # Distribution metrics
        skewness = self._calculate_skewness(ret_array) if len(ret_array) > 3 else 0
        kurtosis = self._calculate_kurtosis(ret_array) if len(ret_array) > 4 else 0
        
        # Benchmark correlation
        if len(self.benchmark_returns) >= len(returns) and len(returns) > 10:
            bench = np.array(self.benchmark_returns[:len(returns)])
            correlation = np.corrcoef(ret_array, bench)[0, 1]
            
            # Beta calculation
            cov = np.cov(ret_array, bench)[0, 1]
            var_bench = np.var(bench)
            beta = cov / var_bench if var_bench > 0 else 1
        else:
            correlation = 0
            beta = 1
        
        return RiskMetrics(
            var_95=abs(var_95) * 100,
            var_99=abs(var_99) * 100,
            cvar_95=abs(cvar_95) * 100,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            leverage=leverage,
            max_position_size=0,  # Requires position data
            herfindahl_index=0,  # Requires position data
            beta=beta,
            correlation_to_benchmark=correlation,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float,
        trading_days: int
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        daily_rf = risk_free_rate / trading_days
        excess_returns = returns - daily_rf
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe * np.sqrt(trading_days)  # Annualize
    
    def _calculate_sortino(
        self,
        returns: np.ndarray,
        risk_free_rate: float,
        trading_days: int
    ) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        daily_rf = risk_free_rate / trading_days
        excess_returns = returns - daily_rf
        
        downside_returns = returns[returns < daily_rf]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / downside_std
        return sortino * np.sqrt(trading_days)
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate information ratio vs benchmark."""
        if not self.benchmark_returns or len(returns) < 2:
            return 0.0
        
        bench = np.array(self.benchmark_returns[:len(returns)])
        if len(bench) != len(returns):
            return 0.0
        
        active_returns = returns - bench
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(active_returns) / tracking_error * np.sqrt(365)
    
    def _calculate_max_dd_duration(self, equity: np.ndarray) -> int:
        """Calculate maximum drawdown duration in days."""
        peak = equity[0]
        in_drawdown = False
        current_duration = 0
        max_duration = 0
        
        for value in equity:
            if value >= peak:
                peak = value
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    current_duration = 0
                    in_drawdown = False
            else:
                in_drawdown = True
                current_duration += 1
        
        return max(max_duration, current_duration)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (np.sum((data - mean) ** 3) / n) / (std ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of distribution."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return (np.sum((data - mean) ** 4) / n) / (std ** 4) - 3
    
    def _empty_performance_metrics(self) -> PerformanceMetrics:
        """Return empty metrics object."""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, monthly_return=0, daily_return=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, information_ratio=0,
            max_drawdown=0, current_drawdown=0, avg_drawdown=0, max_drawdown_duration_days=0,
            annualized_volatility=0, daily_volatility=0, downside_volatility=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            profit_factor=0, avg_win=0, avg_loss=0, best_trade=0, worst_trade=0,
            avg_trade_duration_hours=0, recovery_factor=0, expectancy=0, kelly_criterion=0
        )
