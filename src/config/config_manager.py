"""
Configuration Management with Pydantic Settings.

This module provides type-safe configuration with validation, environment variable
support, and hot-reload capabilities.

Key Features:
1. Type safety with Pydantic BaseSettings
2. Environment variable support (via validation_alias)
3. Range validation using Field constraints
4. Custom validators for business logic
5. Hot-reload capability without restarting the bot

Usage:
    from src.config.config_manager import TradingConfig, config, reload_config

    # Access configuration values
    print(config.exchange_name)
    print(config.max_position_pct)

    # Reload configuration (e.g., after updating .env file)
    reload_config()
"""

from typing import Optional
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
import logging

logger = logging.getLogger(__name__)


class TradingConfig(BaseSettings):
    """Type-safe trading configuration with validation."""

    # Exchange
    exchange_name: str = "binance"
    api_key: str = Field(..., validation_alias="BINANCE_API_KEY")
    api_secret: str = Field(..., validation_alias="BINANCE_API_SECRET")

    # Risk Management
    max_position_pct: float = Field(0.10, ge=0.01, le=0.50)  # 1-50%
    max_daily_loss_pct: float = Field(0.05, ge=0.01, le=0.20)  # 1-20%

    # ML
    model_path: str = "models/production/ensemble_v1.pkl"
    prediction_threshold: float = Field(0.60, ge=0.5, le=0.95)

    # Monitoring
    prometheus_port: int = 8000
    log_level: str = "INFO"

    @field_validator("max_position_pct")
    @classmethod
    def validate_position_size(cls, v):
        """Additional business logic validation."""
        if v > 0.30:
            raise ValueError("Position size > 30% is too risky!")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


# Global config instance
config: TradingConfig = TradingConfig()


def reload_config() -> None:
    """
    Reload configuration without restart.

    This function reloads configuration from environment variables and .env file.
    Useful for feature flags or parameter updates during runtime.
    """
    global config
    config = TradingConfig()
    logger.info("Configuration reloaded successfully")


def load_config_from_yaml(path: str) -> TradingConfig:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        TradingConfig instance populated from YAML.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Convert YAML data to TradingConfig
    # Note: Environment variables will still take precedence unless overridden
    return TradingConfig(**data)


def save_config_to_yaml(config: TradingConfig, path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: TradingConfig instance.
        path: Path where YAML file will be saved.
    """
    # Convert to dict (exclude default values if desired)
    data = config.model_dump(exclude_defaults=False)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    logger.info(f"Configuration saved to {path}")


def validate_config_for_live_trading(config: TradingConfig) -> list[str]:
    """
    Validate configuration for live trading safety.

    Returns a list of warnings/issues found.
    """
    issues = []

    if config.max_position_pct > 0.25:
        issues.append(f"High position size ({config.max_position_pct:.0%})")

    if config.prediction_threshold < 0.65:
        issues.append(f"Low prediction threshold ({config.prediction_threshold})")

    if config.log_level not in ["INFO", "WARNING", "ERROR"]:
        issues.append(f"Unusual log level: {config.log_level}")

    return issues


# Example usage when module is run directly
if __name__ == "__main__":
    # Print current configuration
    print("Current Configuration:")
    print(f"Exchange: {config.exchange_name}")
    print(f"Max Position: {config.max_position_pct:.0%}")
    print(f"Prediction Threshold: {config.prediction_threshold}")
    print(f"Log Level: {config.log_level}")

    # Validate for live trading
    issues = validate_config_for_live_trading(config)
    if issues:
        print("\n⚠️  Live Trading Warnings:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration is safe for live trading")
