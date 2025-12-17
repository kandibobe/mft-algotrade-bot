"""
Position Manager
================

Tracks and manages open positions with PnL calculation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import logging

from src.order_manager.order_types import Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """
    Represents an open trading position.

    Tracks entry, current state, and PnL.
    """

    # Identity
    position_id: str
    symbol: str
    side: PositionSide

    # Entry details
    entry_price: float
    entry_time: datetime
    quantity: float

    # Current state
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # PnL tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0

    # Cost tracking
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    total_commission: float = 0.0

    # Metadata
    strategy_name: Optional[str] = None
    entry_reason: Optional[str] = None
    tags: Dict = field(default_factory=dict)

    # Related orders
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None

    # Lifecycle
    is_open: bool = True
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

    def update_price(self, current_price: float):
        """
        Update current market price and recalculate PnL.

        Args:
            current_price: Current market price
        """
        self.current_price = current_price
        self._calculate_unrealized_pnl()

    def _calculate_unrealized_pnl(self):
        """Calculate unrealized PnL based on current price."""
        if self.side == PositionSide.LONG:
            price_diff = self.current_price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - self.current_price

        self.unrealized_pnl = price_diff * self.quantity - self.entry_commission
        self.unrealized_pnl_pct = (price_diff / self.entry_price) * 100

    def close(self, exit_price: float, exit_commission: float = 0.0, reason: str = "manual"):
        """
        Close the position.

        Args:
            exit_price: Exit price
            exit_commission: Commission on exit
            reason: Reason for closing
        """
        self.is_open = False
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_commission = exit_commission
        self.exit_reason = reason

        # Calculate realized PnL
        if self.side == PositionSide.LONG:
            price_diff = exit_price - self.entry_price
        else:
            price_diff = self.entry_price - exit_price

        self.total_commission = self.entry_commission + exit_commission
        self.realized_pnl = price_diff * self.quantity - self.total_commission

        # Calculate realized PnL percentage
        realized_pnl_pct = (price_diff / self.entry_price) * 100

        logger.info(
            f"Position {self.position_id} closed: {self.symbol} "
            f"Entry: {self.entry_price:.2f} → Exit: {exit_price:.2f} "
            f"PnL: {self.realized_pnl:.2f} ({realized_pnl_pct:.2f}%) "
            f"Reason: {reason}"
        )

    @property
    def duration_minutes(self) -> float:
        """Get position duration in minutes."""
        end_time = self.exit_time if self.exit_time else datetime.now()
        delta = end_time - self.entry_time
        return delta.total_seconds() / 60

    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable."""
        pnl = self.realized_pnl if not self.is_open else self.unrealized_pnl
        return pnl > 0

    @property
    def should_stop_loss(self) -> bool:
        """Check if stop-loss should trigger."""
        if self.stop_loss is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss

    @property
    def should_take_profit(self) -> bool:
        """Check if take-profit should trigger."""
        if self.take_profit is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit

    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "quantity": self.quantity,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "is_open": self.is_open,
            "duration_minutes": self.duration_minutes,
            "strategy_name": self.strategy_name,
            "entry_reason": self.entry_reason,
        }


class PositionManager:
    """
    Manages all open and closed positions.

    Features:
    - Track multiple positions per symbol
    - Real-time PnL calculation
    - Position limits enforcement
    - Stop-loss/take-profit monitoring
    """

    def __init__(self, max_positions: int = 5, max_position_per_symbol: int = 1):
        """
        Initialize position manager.

        Args:
            max_positions: Maximum total open positions
            max_position_per_symbol: Maximum positions per symbol
        """
        self.max_positions = max_positions
        self.max_position_per_symbol = max_position_per_symbol

        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.symbol_positions: Dict[str, List[str]] = {}  # symbol -> [position_ids]
        self.closed_positions: List[Position] = []

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_name: Optional[str] = None,
        entry_reason: Optional[str] = None,
        entry_commission: float = 0.0,
        entry_order_id: Optional[str] = None,
    ) -> Optional[Position]:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            entry_price: Entry price
            quantity: Position size
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            strategy_name: Strategy name
            entry_reason: Reason for entry
            entry_commission: Entry commission
            entry_order_id: Related entry order ID

        Returns:
            Position object if opened, None if rejected
        """
        # Check position limits
        if not self.can_open_position(symbol):
            logger.warning(
                f"Cannot open position: limits exceeded "
                f"(total: {len(self.positions)}/{self.max_positions}, "
                f"symbol: {len(self.symbol_positions.get(symbol, []))}/{self.max_position_per_symbol})"
            )
            return None

        # Create position
        position_id = f"pos_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        position = Position(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_commission=entry_commission,
            strategy_name=strategy_name,
            entry_reason=entry_reason,
            entry_order_id=entry_order_id,
        )

        # Store position
        self.positions[position_id] = position
        if symbol not in self.symbol_positions:
            self.symbol_positions[symbol] = []
        self.symbol_positions[symbol].append(position_id)

        logger.info(
            f"Position opened: {position_id} | {symbol} {side.value.upper()} "
            f"{quantity} @ {entry_price:.2f} | SL: {stop_loss} TP: {take_profit}"
        )

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_commission: float = 0.0,
        reason: str = "manual",
        exit_order_id: Optional[str] = None,
    ) -> bool:
        """
        Close an existing position.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_commission: Exit commission
            reason: Reason for closing
            exit_order_id: Related exit order ID

        Returns:
            True if closed successfully
        """
        position = self.positions.get(position_id)
        if not position:
            logger.warning(f"Position {position_id} not found")
            return False

        if not position.is_open:
            logger.warning(f"Position {position_id} already closed")
            return False

        # Close position
        position.close(exit_price, exit_commission, reason)
        position.exit_order_id = exit_order_id

        # Move to closed positions
        del self.positions[position_id]
        self.symbol_positions[position.symbol].remove(position_id)
        self.closed_positions.append(position)

        return True

    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions.

        Args:
            prices: Dict of symbol -> current price
        """
        for position in self.positions.values():
            if position.symbol in prices:
                position.update_price(prices[position.symbol])

    def check_stop_loss_take_profit(self) -> List[Position]:
        """
        Check all positions for stop-loss or take-profit triggers.

        Returns:
            List of positions that should be closed
        """
        to_close = []

        for position in self.positions.values():
            if position.should_stop_loss:
                logger.warning(
                    f"Stop-loss triggered for {position.position_id}: "
                    f"{position.symbol} @ {position.current_price:.2f}"
                )
                to_close.append(position)
            elif position.should_take_profit:
                logger.info(
                    f"Take-profit triggered for {position.position_id}: "
                    f"{position.symbol} @ {position.current_price:.2f}"
                )
                to_close.append(position)

        return to_close

    def can_open_position(self, symbol: str) -> bool:
        """
        Check if can open a new position.

        Args:
            symbol: Trading symbol

        Returns:
            True if can open position
        """
        # Check total positions
        if len(self.positions) >= self.max_positions:
            return False

        # Check symbol positions
        symbol_count = len(self.symbol_positions.get(symbol, []))
        if symbol_count >= self.max_position_per_symbol:
            return False

        return True

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        return self.positions.get(position_id)

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all open positions for a symbol."""
        position_ids = self.symbol_positions.get(symbol, [])
        return [self.positions[pid] for pid in position_ids if pid in self.positions]

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_total_realized_pnl(self) -> float:
        """Get total realized PnL from closed positions."""
        return sum(pos.realized_pnl for pos in self.closed_positions)

    def get_statistics(self) -> Dict:
        """Get position statistics."""
        open_positions = list(self.positions.values())

        return {
            "open_positions": len(open_positions),
            "closed_positions": len(self.closed_positions),
            "total_unrealized_pnl": self.get_total_unrealized_pnl(),
            "total_realized_pnl": self.get_total_realized_pnl(),
            "total_pnl": self.get_total_unrealized_pnl() + self.get_total_realized_pnl(),
            "winning_trades": sum(1 for p in self.closed_positions if p.is_profitable),
            "losing_trades": sum(1 for p in self.closed_positions if not p.is_profitable),
            "win_rate": (
                sum(1 for p in self.closed_positions if p.is_profitable) /
                max(1, len(self.closed_positions)) * 100
            ),
            "max_positions": self.max_positions,
            "positions_by_symbol": {
                symbol: len(pids) for symbol, pids in self.symbol_positions.items()
            },
        }
