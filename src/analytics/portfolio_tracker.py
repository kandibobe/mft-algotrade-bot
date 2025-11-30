#!/usr/bin/env python3
"""
Real-time Portfolio Tracker
============================

Tracks portfolio state, positions, and P&L in real-time.
Integrates with WebSocket data streams for mark-to-market updates.

Features:
- Real-time position tracking
- Mark-to-market P&L calculation
- Historical equity curve
- Risk metrics (exposure, concentration)
- Trade history with analytics

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def notional_value(self) -> float:
        """Position value at current price."""
        return self.quantity * self.current_price
    
    @property
    def entry_value(self) -> float:
        """Position value at entry."""
        return self.quantity * self.entry_price
    
    @property
    def hold_duration_seconds(self) -> float:
        """How long position has been held."""
        return time.time() - self.entry_time
    
    def update_price(self, price: float):
        """Update position with new price."""
        self.current_price = price
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
        self.unrealized_pnl_pct = (self.unrealized_pnl / self.entry_value) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "notional_value": self.notional_value,
            "entry_time": self.entry_time,
            "hold_duration_seconds": self.hold_duration_seconds,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit
        }


@dataclass
class Trade:
    """Completed trade record."""
    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    pnl: float
    pnl_pct: float
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hold_duration(self) -> float:
        return self.exit_time - self.entry_time
    
    @property
    def net_pnl(self) -> float:
        return self.pnl - self.fees


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: float
    total_equity: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    open_positions: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "total_equity": self.total_equity,
            "cash_balance": self.cash_balance,
            "positions_value": self.positions_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "open_positions": self.open_positions
        }


class PortfolioTracker:
    """
    Real-time portfolio tracking with P&L calculations.
    
    Usage:
        tracker = PortfolioTracker(initial_balance=10000)
        
        # Open position
        tracker.open_position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.1,
            price=95000
        )
        
        # Update prices (from WebSocket)
        tracker.update_price("BTC/USDT", 96000)
        
        # Get snapshot
        snapshot = tracker.get_snapshot()
        print(f"Equity: ${snapshot.total_equity:.2f}")
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001  # 0.1% per trade
    ):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.fee_rate = fee_rate
        
        # Positions
        self._positions: Dict[str, Position] = {}
        self._trade_history: List[Trade] = []
        
        # Performance tracking
        self._realized_pnl = 0.0
        self._total_fees = 0.0
        self._equity_history: List[PortfolioSnapshot] = []
        self._peak_equity = initial_balance
        
        # Callbacks
        self._snapshot_handlers: List[Callable] = []
        self._position_handlers: List[Callable] = []
        self._alert_handlers: List[Callable] = []
        
        # Trade counter
        self._trade_counter = 0
    
    # =========================================================================
    # Position Management
    # =========================================================================
    
    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Position]:
        """Open a new position."""
        # Calculate fees
        fee = quantity * price * self.fee_rate
        required_cash = quantity * price + fee
        
        if required_cash > self.cash_balance:
            logger.warning(f"Insufficient funds for {symbol} position")
            return None
        
        if symbol in self._positions:
            logger.warning(f"Position already exists for {symbol}")
            return None
        
        # Deduct cash
        self.cash_balance -= required_cash
        self._total_fees += fee
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=time.time(),
            current_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )
        self._positions[symbol] = position
        
        logger.info(f"Opened {side.value} position: {symbol} @ {price}")
        
        # Notify handlers
        for handler in self._position_handlers:
            handler("open", position)
        
        return position
    
    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "manual"
    ) -> Optional[Trade]:
        """Close an existing position."""
        if symbol not in self._positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        position = self._positions[symbol]
        position.update_price(price)
        
        # Calculate final P&L
        fee = position.quantity * price * self.fee_rate
        net_value = position.notional_value - fee
        pnl = position.unrealized_pnl - fee
        pnl_pct = (pnl / position.entry_value) * 100
        
        # Update cash and realized P&L
        self.cash_balance += net_value
        self._realized_pnl += position.unrealized_pnl
        self._total_fees += fee
        
        # Record trade
        self._trade_counter += 1
        trade = Trade(
            trade_id=f"T{self._trade_counter:06d}",
            symbol=symbol,
            side=position.side.value,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=price,
            entry_time=position.entry_time,
            exit_time=time.time(),
            pnl=position.unrealized_pnl,
            pnl_pct=pnl_pct,
            fees=fee * 2,  # Entry + exit fees
            metadata={"reason": reason, **position.metadata}
        )
        self._trade_history.append(trade)
        
        # Remove position
        del self._positions[symbol]
        
        logger.info(f"Closed {symbol} @ {price}, P&L: ${pnl:.2f} ({reason})")
        
        # Notify handlers
        for handler in self._position_handlers:
            handler("close", trade)
        
        return trade
    
    def update_price(self, symbol: str, price: float):
        """Update price for a symbol."""
        if symbol in self._positions:
            position = self._positions[symbol]
            position.update_price(price)
            
            # Check stop loss / take profit
            self._check_exit_triggers(position)
    
    def update_prices(self, prices: Dict[str, float]):
        """Batch update prices."""
        for symbol, price in prices.items():
            self.update_price(symbol, price)
    
    def _check_exit_triggers(self, position: Position):
        """Check if stop loss or take profit triggered."""
        # Stop loss check
        if position.stop_loss:
            if position.side == PositionSide.LONG and position.current_price <= position.stop_loss:
                self.close_position(position.symbol, position.current_price, "stop_loss")
                self._emit_alert("stop_loss", position)
                return
            elif position.side == PositionSide.SHORT and position.current_price >= position.stop_loss:
                self.close_position(position.symbol, position.current_price, "stop_loss")
                self._emit_alert("stop_loss", position)
                return
        
        # Take profit check
        if position.take_profit:
            if position.side == PositionSide.LONG and position.current_price >= position.take_profit:
                self.close_position(position.symbol, position.current_price, "take_profit")
                self._emit_alert("take_profit", position)
                return
            elif position.side == PositionSide.SHORT and position.current_price <= position.take_profit:
                self.close_position(position.symbol, position.current_price, "take_profit")
                self._emit_alert("take_profit", position)
                return
    
    def _emit_alert(self, alert_type: str, data: Any):
        """Emit alert to handlers."""
        for handler in self._alert_handlers:
            handler(alert_type, data)
    
    # =========================================================================
    # Portfolio State
    # =========================================================================
    
    def get_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        positions_value = sum(p.notional_value for p in self._positions.values())
        total_equity = self.cash_balance + positions_value
        
        snapshot = PortfolioSnapshot(
            timestamp=time.time(),
            total_equity=total_equity,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            total_pnl=self._realized_pnl + unrealized_pnl,
            open_positions=len(self._positions)
        )
        
        # Track equity history
        self._equity_history.append(snapshot)
        
        # Update peak
        if total_equity > self._peak_equity:
            self._peak_equity = total_equity
        
        # Notify handlers
        for handler in self._snapshot_handlers:
            handler(snapshot)
        
        return snapshot
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())
    
    def get_trade_history(self, limit: int = 100) -> List[Trade]:
        """Get recent trade history."""
        return self._trade_history[-limit:]
    
    def get_equity_history(self, limit: int = 1000) -> List[PortfolioSnapshot]:
        """Get equity curve history."""
        return self._equity_history[-limit:]
    
    # =========================================================================
    # Risk Metrics
    # =========================================================================
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        current_equity = self.get_snapshot().total_equity
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - current_equity) / self._peak_equity * 100
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from equity history."""
        if len(self._equity_history) < 2:
            return 0.0
        
        peak = self._equity_history[0].total_equity
        max_dd = 0.0
        
        for snapshot in self._equity_history:
            if snapshot.total_equity > peak:
                peak = snapshot.total_equity
            dd = (peak - snapshot.total_equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    @property
    def total_exposure(self) -> float:
        """Total position exposure as percentage of equity."""
        snapshot = self.get_snapshot()
        if snapshot.total_equity <= 0:
            return 0.0
        return (snapshot.positions_value / snapshot.total_equity) * 100
    
    def get_concentration(self) -> Dict[str, float]:
        """Get position concentration by symbol."""
        snapshot = self.get_snapshot()
        if snapshot.total_equity <= 0:
            return {}
        
        return {
            p.symbol: (p.notional_value / snapshot.total_equity) * 100
            for p in self._positions.values()
        }
    
    # =========================================================================
    # Performance Metrics
    # =========================================================================
    
    @property
    def total_return(self) -> float:
        """Total return percentage."""
        if self.initial_balance <= 0:
            return 0.0
        current = self.get_snapshot().total_equity
        return ((current - self.initial_balance) / self.initial_balance) * 100
    
    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if not self._trade_history:
            return 0.0
        winners = sum(1 for t in self._trade_history if t.pnl > 0)
        return (winners / len(self._trade_history)) * 100
    
    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss ratio."""
        gross_profit = sum(t.pnl for t in self._trade_history if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._trade_history if t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    @property
    def average_trade_pnl(self) -> float:
        """Average P&L per trade."""
        if not self._trade_history:
            return 0.0
        return sum(t.pnl for t in self._trade_history) / len(self._trade_history)
    
    @property
    def trade_count(self) -> int:
        """Total number of completed trades."""
        return len(self._trade_history)
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    def on_snapshot(self, handler: Callable):
        """Register snapshot update handler."""
        self._snapshot_handlers.append(handler)
        return handler
    
    def on_position(self, handler: Callable):
        """Register position change handler."""
        self._position_handlers.append(handler)
        return handler
    
    def on_alert(self, handler: Callable):
        """Register alert handler."""
        self._alert_handlers.append(handler)
        return handler
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state."""
        snapshot = self.get_snapshot()
        return {
            "snapshot": snapshot.to_dict(),
            "positions": [p.to_dict() for p in self._positions.values()],
            "metrics": {
                "total_return": self.total_return,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "max_drawdown": self.max_drawdown,
                "current_drawdown": self.current_drawdown,
                "total_exposure": self.total_exposure,
                "trade_count": self.trade_count,
                "average_trade_pnl": self.average_trade_pnl,
                "total_fees": self._total_fees
            },
            "concentration": self.get_concentration()
        }
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)
