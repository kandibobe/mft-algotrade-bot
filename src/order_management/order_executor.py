"""
Stoic Citadel - Order Executor
==============================

Order execution engine with:
- Order lifecycle management
- Multiple order types (market, limit, stop)
- Execution tracking
- Fill simulation for backtesting
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import threading
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"       # Not yet submitted
    SUBMITTED = "submitted"   # Sent to exchange
    PARTIAL = "partial"       # Partially filled
    FILLED = "filled"         # Fully executed
    CANCELLED = "cancelled"   # Cancelled
    REJECTED = "rejected"     # Rejected by exchange
    EXPIRED = "expired"       # Expired (time-in-force)


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    
    # Prices (depending on order type)
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_pct: Optional[float] = None
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    
    # Execution details
    fills: List[Dict] = field(default_factory=list)
    commission: float = 0.0
    slippage: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    notes: str = ""
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "slippage": self.slippage,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None
        }


class OrderExecutor:
    """
    Order execution engine.
    
    Handles order creation, submission, and lifecycle management.
    Can operate in live or simulation mode.
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% default
        simulation_mode: bool = True
    ):
        self.commission_rate = commission_rate
        self.simulation_mode = simulation_mode
        
        self._orders: Dict[str, Order] = {}
        self._open_orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        
        self._lock = threading.Lock()
        self._callbacks: Dict[str, List[Callable]] = {
            "on_fill": [],
            "on_cancel": [],
            "on_reject": []
        }
        
        # Exchange adapter (to be set for live trading)
        self._exchange_adapter = None
        
        logger.info(
            f"Order Executor initialized (simulation={simulation_mode})"
        )
    
    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        **kwargs
    ) -> Order:
        """Create a market order."""
        return self._create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            **kwargs
        )
    
    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        **kwargs
    ) -> Order:
        """Create a limit order."""
        return self._create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            **kwargs
        )
    
    def create_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        **kwargs
    ) -> Order:
        """Create a stop order."""
        return self._create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=stop_price,
            **kwargs
        )
    
    def create_stop_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: float,
        **kwargs
    ) -> Order:
        """Create a stop-limit order."""
        return self._create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LIMIT,
            quantity=quantity,
            stop_price=stop_price,
            limit_price=limit_price,
            **kwargs
        )
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for execution.
        
        Returns True if successfully submitted.
        """
        with self._lock:
            if order.status != OrderStatus.PENDING:
                logger.warning(f"Order {order.order_id} not in pending state")
                return False
            
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()
            self._open_orders[order.order_id] = order
            
            logger.info(
                f"Order submitted: {order.order_id} "
                f"{order.side.value} {order.quantity} {order.symbol}"
            )
            
            # In simulation mode, market orders execute immediately
            if self.simulation_mode and order.order_type == OrderType.MARKET:
                # Will be filled by simulate_fill
                pass
            
            return True
    
    def cancel_order(
        self,
        order_id: str,
        reason: str = ""
    ) -> bool:
        """
        Cancel an open order.
        """
        with self._lock:
            if order_id not in self._open_orders:
                logger.warning(f"Order {order_id} not found or not open")
                return False
            
            order = self._open_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            order.notes = reason
            
            self._order_history.append(order)
            self._notify_callbacks("on_cancel", order)
            
            logger.info(f"Order cancelled: {order_id} - {reason}")
            return True
    
    def simulate_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: Optional[float] = None,
        slippage: float = 0.0
    ) -> None:
        """
        Simulate order fill (for backtesting).
        
        Args:
            order_id: Order to fill
            fill_price: Execution price
            fill_quantity: Quantity filled (None = full fill)
            slippage: Slippage amount
        """
        with self._lock:
            if order_id not in self._open_orders:
                return
            
            order = self._open_orders[order_id]
            fill_qty = fill_quantity or order.remaining_quantity
            
            # Apply slippage
            if order.side == OrderSide.BUY:
                adjusted_price = fill_price + slippage
            else:
                adjusted_price = fill_price - slippage
            
            # Record fill
            fill = {
                "price": adjusted_price,
                "quantity": fill_qty,
                "timestamp": datetime.utcnow().isoformat(),
                "slippage": slippage
            }
            order.fills.append(fill)
            
            # Update order state
            total_value = (order.average_fill_price * order.filled_quantity + 
                          adjusted_price * fill_qty)
            order.filled_quantity += fill_qty
            order.average_fill_price = total_value / order.filled_quantity
            order.slippage += slippage * fill_qty
            
            # Calculate commission
            order.commission += adjusted_price * fill_qty * self.commission_rate
            
            # Check if fully filled
            if order.remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
                self._open_orders.pop(order_id)
                self._order_history.append(order)
            else:
                order.status = OrderStatus.PARTIAL
            
            self._notify_callbacks("on_fill", order)
            
            logger.info(
                f"Order filled: {order_id} @ {adjusted_price:.4f} "
                f"(slippage: {slippage:.4f})"
            )
    
    def check_stop_orders(
        self,
        symbol: str,
        current_price: float,
        high_price: Optional[float] = None,
        low_price: Optional[float] = None
    ) -> List[str]:
        """
        Check if any stop orders should trigger.
        
        Returns list of triggered order IDs.
        """
        triggered = []
        high = high_price or current_price
        low = low_price or current_price
        
        with self._lock:
            for order_id, order in list(self._open_orders.items()):
                if order.symbol != symbol:
                    continue
                
                if order.order_type not in [OrderType.STOP, OrderType.STOP_LIMIT]:
                    continue
                
                if order.stop_price is None:
                    continue
                
                # Check trigger conditions
                if order.side == OrderSide.SELL and low <= order.stop_price:
                    triggered.append(order_id)
                elif order.side == OrderSide.BUY and high >= order.stop_price:
                    triggered.append(order_id)
        
        return triggered
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        if order_id in self._orders:
            return self._orders[order_id]
        return None
    
    def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """Get all open orders."""
        orders = list(self._open_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history."""
        history = self._order_history[-limit:]
        if symbol:
            history = [o for o in history if o.symbol == symbol]
        return history
    
    def cancel_all_orders(
        self,
        symbol: Optional[str] = None
    ) -> int:
        """Cancel all open orders."""
        count = 0
        for order_id in list(self._open_orders.keys()):
            order = self._open_orders[order_id]
            if symbol is None or order.symbol == symbol:
                if self.cancel_order(order_id, "Cancel all"):
                    count += 1
        return count
    
    def register_callback(
        self,
        event: str,
        callback: Callable
    ) -> None:
        """Register callback for order events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def get_statistics(self) -> Dict:
        """Get order execution statistics."""
        filled_orders = [
            o for o in self._order_history
            if o.status == OrderStatus.FILLED
        ]
        
        total_commission = sum(o.commission for o in filled_orders)
        total_slippage = sum(o.slippage for o in filled_orders)
        total_volume = sum(
            o.average_fill_price * o.filled_quantity
            for o in filled_orders
        )
        
        return {
            "total_orders": len(self._orders),
            "open_orders": len(self._open_orders),
            "filled_orders": len(filled_orders),
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "total_volume": total_volume,
            "avg_commission_pct": (total_commission / total_volume * 100) if total_volume > 0 else 0,
            "avg_slippage_pct": (total_slippage / total_volume * 100) if total_volume > 0 else 0
        }
    
    def _create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        **kwargs
    ) -> Order:
        """Create a new order."""
        order_id = str(uuid.uuid4())[:12]
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=kwargs.get("limit_price"),
            stop_price=kwargs.get("stop_price"),
            trailing_pct=kwargs.get("trailing_pct"),
            strategy_id=kwargs.get("strategy_id"),
            signal_id=kwargs.get("signal_id"),
            notes=kwargs.get("notes", "")
        )
        
        self._orders[order_id] = order
        return order
    
    def _notify_callbacks(
        self,
        event: str,
        order: Order
    ) -> None:
        """Notify registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Callback error: {e}")
