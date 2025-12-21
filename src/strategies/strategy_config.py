"""
Stoic Citadel - Strategy Configuration
=======================================

Centralized configuration management for strategies.
Supports YAML/JSON config files and environment variables.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration.

    All parameters can be overridden via:
    1. Config file (YAML/JSON)
    2. Environment variables (STOIC_<PARAM>)
    3. Direct instantiation
    """

    # === General ===
    name: str = "StoicStrategy"
    version: str = "1.0.0"
    timeframe: str = "5m"
    startup_candle_count: int = 200

    # === Risk Management ===
    risk_per_trade: float = 0.02
    max_positions: int = 3
    max_drawdown: float = 0.15
    stoploss: float = -0.05
    trailing_stop: bool = True
    trailing_stop_positive: float = 0.01
    trailing_stop_positive_offset: float = 0.015

    # === Entry Parameters ===
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    stoch_oversold: float = 30.0
    stoch_overbought: float = 80.0
    min_adx: float = 20.0
    min_volume_ratio: float = 0.8
    min_bb_width: float = 0.02
    max_bb_width: float = 0.20

    # === Exit Parameters ===
    exit_rsi_overbought: float = 75.0
    exit_stoch_overbought: float = 80.0

    # === ROI Table ===
    roi_0: float = 0.15  # 15% immediate
    roi_30: float = 0.08  # 8% after 30 min
    roi_60: float = 0.05  # 5% after 60 min
    roi_120: float = 0.03  # 3% after 120 min

    # === Regime Detection ===
    regime_aware: bool = True
    regime_adx_threshold: float = 25.0
    regime_aggressive_score: float = 70.0
    regime_defensive_score: float = 40.0

    # === Fees & Slippage (for backtest) ===
    fee: float = 0.001  # 0.1%
    slippage_entry: float = 0.0005  # 0.05%
    slippage_exit: float = 0.0005

    # === Protections ===
    protection_stoploss_guard: bool = True
    protection_stoploss_lookback: int = 60
    protection_stoploss_limit: int = 3
    protection_cooldown: int = 24

    @classmethod
    def from_file(cls, path: str) -> "StrategyConfig":
        """
        Load config from YAML or JSON file.

        Args:
            path: Path to config file

        Returns:
            StrategyConfig instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            if path.suffix in [".yml", ".yaml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        logger.info(f"Loaded config from {path}")
        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = "STOIC_") -> "StrategyConfig":
        """
        Load config from environment variables.

        Environment variables should be prefixed (default: STOIC_)
        Example: STOIC_RISK_PER_TRADE=0.02
        """
        config = cls()

        for field_name in config.__dataclass_fields__:
            env_name = f"{prefix}{field_name.upper()}"
            if env_name in os.environ:
                value = os.environ[env_name]
                field_type = config.__dataclass_fields__[field_name].type

                # Type conversion
                if field_type == bool:
                    value = value.lower() in ("true", "1", "yes")
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)

                setattr(config, field_name, value)
                logger.debug(f"Config from env: {field_name}={value}")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Save to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"Saved config to {path}")

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved config to {path}")

    def get_roi_table(self) -> Dict[str, float]:
        """Get ROI table for Freqtrade."""
        return {"0": self.roi_0, "30": self.roi_30, "60": self.roi_60, "120": self.roi_120}

    def get_protections(self) -> list:
        """Get protections config for Freqtrade."""
        protections = []

        if self.protection_stoploss_guard:
            protections.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": self.protection_stoploss_lookback,
                    "trade_limit": self.protection_stoploss_limit,
                    "stop_duration_candles": self.protection_cooldown,
                    "required_profit": 0.0,
                }
            )

        return protections

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid

        Raises:
            ValueError if invalid
        """
        errors = []

        if not 0 < self.risk_per_trade < 0.2:
            errors.append(f"risk_per_trade should be 0-20%, got {self.risk_per_trade}")

        if not 0 < self.max_positions <= 10:
            errors.append(f"max_positions should be 1-10, got {self.max_positions}")

        if not -0.5 < self.stoploss < 0:
            errors.append(f"stoploss should be -50% to 0, got {self.stoploss}")

        if self.rsi_oversold >= self.rsi_overbought:
            errors.append("rsi_oversold must be < rsi_overbought")

        if self.min_bb_width >= self.max_bb_width:
            errors.append("min_bb_width must be < max_bb_width")

        if errors:
            raise ValueError("Config validation errors: " + "; ".join(errors))

        return True
