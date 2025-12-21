"""Configuration module with Pydantic validation."""

from src.config.config_manager import TradingConfig as TradingSettings
from src.config.config_manager import (
    config,
    load_config_from_yaml,
    reload_config,
    save_config_to_yaml,
    validate_config_for_live_trading,
)
from src.config.validated_config import (
    BacktestConfig,
    ExchangeConfig,
    MLConfig,
    RiskConfig,
    TradingConfig,
    load_config,
)

__all__ = [
    # From validated_config
    "TradingConfig",
    "ExchangeConfig",
    "RiskConfig",
    "MLConfig",
    "BacktestConfig",
    "load_config",
    # From config_manager
    "TradingSettings",
    "config",
    "reload_config",
    "load_config_from_yaml",
    "save_config_to_yaml",
    "validate_config_for_live_trading",
]
