"""Configuration module with Pydantic validation."""

from src.config.unified_config import (
    TradingConfig,
    ExchangeConfig,
    RiskConfig,
    MLConfig,
    StrategyConfig,
    PathConfig,
    SystemConfig,
    load_config,
)
from src.config.manager import ConfigurationManager

config = ConfigurationManager.get_config

__all__ = [
    "TradingConfig",
    "ExchangeConfig",
    "RiskConfig",
    "MLConfig",
    "StrategyConfig",
    "PathConfig",
    "SystemConfig",
    "load_config",
    "config",
    "ConfigurationManager",
]
