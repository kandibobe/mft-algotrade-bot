class ExecutionError(Exception):
    """Base class for execution errors."""
    pass

class InsufficientFundsError(ExecutionError):
    """Raised when account has insufficient funds."""
    pass

class RateLimitError(ExecutionError):
    """Raised when exchange rate limit is exceeded."""
    pass

class OrderValidationError(ExecutionError):
    """Raised when order parameters are invalid."""
    pass

class NetworkError(ExecutionError):
    """Raised when network connection fails."""
    pass

class ExchangeError(ExecutionError):
    """Generic exchange error."""
    pass
