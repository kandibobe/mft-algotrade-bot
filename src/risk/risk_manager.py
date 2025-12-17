"""
Stoic Citadel - Integrated Risk Manager
========================================

Central risk management coordination:
- Circuit breaker integration
- Position sizing
- Portfolio risk
- Real-time risk monitoring
"""

import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from .position_sizing import PositionSizer, PositionSizingConfig
from .correlation import CorrelationAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Portfolio metrics
    total_exposure: float = 0.0
    exposure_pct: float = 0.0
    open_positions: int = 0
    
    # PnL metrics
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Risk metrics
    current_drawdown_pct: float = 0.0
    var_95: float = 0.0
    sharpe_estimate: float = 0.0
    
    # Circuit breaker
    circuit_state: str = "closed"
    can_trade: bool = True
    position_multiplier: float = 1.0


class RiskManager:
    """
    Central risk management system.
    
    Coordinates all risk components and provides unified interface
    for trading decisions.
    """
    
    def __init__(
        self,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        sizing_config: Optional[PositionSizingConfig] = None,
        enable_notifications: bool = True
    ):
        # Initialize components
        self.circuit_breaker = CircuitBreaker(circuit_config)
        self.position_sizer = PositionSizer(sizing_config)
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # State
        self._account_balance: float = 0.0
        self._positions: Dict[str, Dict] = {}
        self._trade_history: List[Dict] = []
        self._metrics: RiskMetrics = RiskMetrics()
        self._lock = threading.Lock()
        
        # Notifications
        self._enable_notifications = enable_notifications
        self._notification_handlers: List[callable] = []
        
        # Register circuit breaker callback
        self.circuit_breaker.register_callback(self._on_circuit_state_change)
        
        logger.info("Risk Manager initialized")
    
    def initialize(
        self,
        account_balance: float,
        existing_positions: Optional[Dict[str, Dict]] = None
    ) -> None:
        """
        Initialize risk manager with account state.
        
        Args:
            account_balance: Current account balance
            existing_positions: Dict of symbol -> position details
        """
        with self._lock:
            self._account_balance = account_balance
            self._positions = existing_positions or {}
            
            # Initialize circuit breaker
            self.circuit_breaker.initialize_session(account_balance)
            
            # Update position sizer
            position_values = {
                s: p.get("value", 0) for s, p in self._positions.items()
            }
            self.position_sizer.update_positions(position_values)
            
            # Update metrics
            self._update_metrics()
            
        logger.info(
            f"Risk Manager initialized: Balance=${account_balance:,.2f}, "
            f"Positions={len(self._positions)}"
        )
    
    def evaluate_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        side: str = "long",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate whether a trade should be taken.
        
        Returns comprehensive trade evaluation including:
        - Whether trade is allowed
        - Recommended position size
        - Risk metrics
        - Warnings
        """
        result = {
            "allowed": False,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "position_size": 0.0,
            "position_value": 0.0,
            "risk_amount": 0.0,
            "warnings": [],
            "rejection_reason": None
        }
        
        # Check circuit breaker
        if not self.circuit_breaker.can_trade():
            result["rejection_reason"] = f"Circuit breaker: {self.circuit_breaker.state.value}"
            return result
        
        # Check if already in position
        if symbol in self._positions:
            result["rejection_reason"] = f"Already in position for {symbol}"
            return result
        
        # Calculate position size
        try:
            sizing_result = self.position_sizer.calculate_position_size(
                account_balance=self._account_balance,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                method="optimal",
                **kwargs
            )
        except Exception as e:
            result["rejection_reason"] = f"Position sizing error: {e}"
            return result
        
        # Apply circuit breaker multiplier
        multiplier = self.circuit_breaker.get_position_multiplier()
        sizing_result["position_size"] *= multiplier
        sizing_result["position_value"] *= multiplier
        
        if multiplier < 1.0:
            result["warnings"].append(
                f"Position reduced to {multiplier:.0%} due to circuit breaker recovery"
            )
        
        # Check portfolio risk
        allowed, reason = self.position_sizer.check_portfolio_risk(
            sizing_result,
            symbol,
            self._account_balance
        )
        
        if not allowed:
            result["rejection_reason"] = reason
            return result
        
        # Trade approved
        result["allowed"] = True
        result["position_size"] = sizing_result["position_size"]
        result["position_value"] = sizing_result["position_value"]
        result["risk_amount"] = sizing_result.get("risk_amount", 0)
        result["sizing_method"] = sizing_result.get("method")
        result["sizing_details"] = sizing_result
        
        return result
    
    def record_entry(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        stop_loss_price: float,
        **kwargs
    ) -> None:
        """Record position entry."""
        with self._lock:
            self._positions[symbol] = {
                "entry_price": entry_price,
                "size": position_size,
                "stop_loss": stop_loss_price,
                "value": entry_price * position_size,
                "entry_time": datetime.utcnow(),
                **kwargs
            }
            
            # Update position sizer
            position_values = {
                s: p.get("value", 0) for s, p in self._positions.items()
            }
            self.position_sizer.update_positions(position_values)
            
            self._update_metrics()
            
        logger.info(f"Entry recorded: {symbol} @ {entry_price} x {position_size}")
    
    def record_exit(
        self,
        symbol: str,
        exit_price: float,
        reason: str = ""
    ) -> Dict:
        """Record position exit and return trade result."""
        with self._lock:
            if symbol not in self._positions:
                logger.warning(f"No position found for {symbol}")
                return {}
            
            position = self._positions.pop(symbol)
            
            # Calculate P&L
            entry_price = position["entry_price"]
            size = position["size"]
            pnl = (exit_price - entry_price) * size
            pnl_pct = (exit_price - entry_price) / entry_price
            
            trade_result = {
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": size,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
                "entry_time": position.get("entry_time"),
                "exit_time": datetime.utcnow()
            }
            
            # Update trade history
            self._trade_history.append(trade_result)
            
            # Update circuit breaker
            self.circuit_breaker.record_trade(trade_result, pnl_pct)
            
            # Update balance
            self._account_balance += pnl
            self.circuit_breaker.update_balance(self._account_balance)
            
            # Update position sizer
            position_values = {
                s: p.get("value", 0) for s, p in self._positions.items()
            }
            self.position_sizer.update_positions(position_values)
            
            self._update_metrics()
            
            logger.info(
                f"Exit recorded: {symbol} @ {exit_price} | "
                f"PnL: ${pnl:,.2f} ({pnl_pct:+.2%}) | Reason: {reason}"
            )
            
            return trade_result
    
    def update_market_data(
        self,
        symbol: str,
        current_price: float,
        volatility: Optional[float] = None
    ) -> None:
        """Update with current market data."""
        # Update unrealized P&L
        if symbol in self._positions:
            position = self._positions[symbol]
            unrealized = (current_price - position["entry_price"]) * position["size"]
            position["unrealized_pnl"] = unrealized
            position["current_price"] = current_price
        
        # Check volatility circuit breaker
        if volatility is not None:
            self.circuit_breaker.check_volatility(volatility)
        
        self._update_metrics()
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return self._metrics
    
    def get_status(self) -> Dict:
        """Get comprehensive risk status."""
        return {
            "metrics": self._metrics.__dict__,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "positions": self._positions.copy(),
            "account_balance": self._account_balance,
            "trade_count_today": len([
                t for t in self._trade_history
                if t.get("exit_time", datetime.min).date() == datetime.utcnow().date()
            ])
        }
    
    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self.circuit_breaker.manual_stop()
        self._notify("EMERGENCY STOP TRIGGERED", "critical")
    
    def register_notification_handler(
        self,
        handler: callable
    ) -> None:
        """Register notification handler."""
        self._notification_handlers.append(handler)
    
    def _update_metrics(self) -> None:
        """Update risk metrics."""
        total_exposure = sum(p.get("value", 0) for p in self._positions.values())
        unrealized = sum(p.get("unrealized_pnl", 0) for p in self._positions.values())
        
        cb_status = self.circuit_breaker.get_status()
        
        self._metrics = RiskMetrics(
            timestamp=datetime.utcnow(),
            total_exposure=total_exposure,
            exposure_pct=total_exposure / self._account_balance if self._account_balance > 0 else 0,
            open_positions=len(self._positions),
            daily_pnl=self.circuit_breaker.session.current_balance - self.circuit_breaker.session.initial_balance,
            daily_pnl_pct=cb_status["daily_pnl_pct"],
            unrealized_pnl=unrealized,
            current_drawdown_pct=cb_status["drawdown_pct"],
            circuit_state=cb_status["state"],
            can_trade=cb_status["can_trade"],
            position_multiplier=cb_status["position_multiplier"]
        )
    
    def _on_circuit_state_change(self, status: Dict) -> None:
        """Handle circuit breaker state change."""
        state = status.get("state", "unknown")
        reason = status.get("trip_reason", "unknown")
        
        if state == "open":
            self._notify(
                f"🚨 Circuit Breaker TRIPPED: {reason}",
                "critical"
            )
        elif state == "half_open":
            self._notify(
                "⚠️ Circuit Breaker: Testing recovery",
                "warning"
            )
        elif state == "closed":
            self._notify(
                "✅ Circuit Breaker: Normal operation resumed",
                "info"
            )
    
    def _notify(
        self,
        message: str,
        level: str = "info"
    ) -> None:
        """Send notification."""
        if not self._enable_notifications:
            return
        
        for handler in self._notification_handlers:
            try:
                handler(message, level)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")
