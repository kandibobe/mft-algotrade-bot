"""Advanced risk management module."""
from src.risk.correlation import CorrelationManager, DrawdownMonitor
from src.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from src.risk.position_sizing import (
    PositionSizer,
    PositionSizingConfig,
    create_freqtrade_stake_function,
)

__all__ = [
    'CorrelationManager',
    'DrawdownMonitor',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState',
    'PositionSizer',
    'PositionSizingConfig',
    'create_freqtrade_stake_function',
]
