"""
Validated Configuration
=======================

Pydantic-based configuration with validation.

Key Benefits:
1. Type safety - catches config errors before runtime
2. Validation - ensures values are within expected ranges
3. Documentation - self-documenting config fields
4. Serialization - easy JSON/YAML export

Usage:
    from src.config import TradingConfig

    # Fails fast if invalid
    config = TradingConfig(
        exchange="binance",
        pairs=["BTC/USDT"],
        stake_amount=100,
        max_open_trades=3,
    )

    # Load from file
    config = TradingConfig.from_yaml("config/trading.yaml")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field, field_validator, model_validator

    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseModel, Field, root_validator, validator

        PYDANTIC_V2 = False
    except ImportError:
        raise ImportError("Pydantic not installed. Run: pip install pydantic")

logger = logging.getLogger(__name__)


class ExchangeConfig(BaseModel):
    """Exchange connection configuration."""

    name: str = Field(default="binance", description="Exchange name (binance, bybit, etc.)")
    sandbox: bool = Field(
        default=True, description="Use testnet/sandbox mode (CRITICAL for testing!)"
    )
    api_key: Optional[str] = Field(default=None, description="API key (keep secret!)")
    api_secret: Optional[str] = Field(default=None, description="API secret (keep secret!)")
    rate_limit: bool = Field(default=True, description="Enable rate limiting to avoid bans")
    timeout_ms: int = Field(
        default=30000, ge=1000, le=120000, description="Request timeout in milliseconds"
    )

    if PYDANTIC_V2:

        @field_validator("name")
        @classmethod
        def validate_exchange(cls, v):
            supported = ["binance", "bybit", "okx", "kucoin", "gate"]
            if v.lower() not in supported:
                logger.warning(f"Exchange '{v}' may not be fully supported")
            return v.lower()

    else:

        @validator("name")
        def validate_exchange(cls, v):
            supported = ["binance", "bybit", "okx", "kucoin", "gate"]
            if v.lower() not in supported:
                logger.warning(f"Exchange '{v}' may not be fully supported")
            return v.lower()


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_pct: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="Maximum position size as fraction of portfolio (0.10 = 10%)",
    )
    max_portfolio_risk: float = Field(
        default=0.02, ge=0.001, le=0.10, description="Maximum portfolio risk per trade (0.02 = 2%)"
    )
    max_drawdown_pct: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50,
        description="Maximum drawdown before circuit breaker (0.20 = 20%)",
    )
    max_daily_loss_pct: float = Field(
        default=0.05, ge=0.01, le=0.20, description="Maximum daily loss before stopping (0.05 = 5%)"
    )
    stop_loss_pct: float = Field(
        default=0.02, ge=0.001, le=0.20, description="Default stop-loss percentage (0.02 = 2%)"
    )
    take_profit_pct: float = Field(
        default=0.04, ge=0.001, le=0.50, description="Default take-profit percentage (0.04 = 4%)"
    )
    use_atr_sizing: bool = Field(default=True, description="Use ATR-based position sizing")
    atr_multiplier: float = Field(
        default=2.0, ge=0.5, le=5.0, description="ATR multiplier for stop-loss distance"
    )

    if PYDANTIC_V2:

        @model_validator(mode="after")
        def validate_risk_reward(self):
            if self.take_profit_pct < self.stop_loss_pct:
                logger.warning(
                    f"Take profit ({self.take_profit_pct}) < Stop loss ({self.stop_loss_pct}). "
                    f"Consider adjusting for positive risk/reward ratio."
                )
            return self

    else:

        @root_validator
        def validate_risk_reward(cls, values):
            tp = values.get("take_profit_pct", 0.04)
            sl = values.get("stop_loss_pct", 0.02)
            if tp < sl:
                logger.warning(
                    f"Take profit ({tp}) < Stop loss ({sl}). "
                    f"Consider adjusting for positive risk/reward ratio."
                )
            return values


class MLConfig(BaseModel):
    """Machine Learning configuration."""

    model_type: Literal["lightgbm", "xgboost", "random_forest", "catboost"] = Field(
        default="lightgbm", description="ML model type"
    )
    feature_selection: bool = Field(
        default=True, description="Enable feature selection to prevent overfitting"
    )
    optimize_hyperparams: bool = Field(default=True, description="Run hyperparameter optimization")
    n_optuna_trials: int = Field(
        default=100, ge=10, le=1000, description="Number of Optuna optimization trials"
    )
    cv_folds: int = Field(default=5, ge=2, le=10, description="Cross-validation folds")
    min_samples_split: int = Field(
        default=1000, ge=100, description="Minimum samples for train/test split"
    )
    label_method: Literal["triple_barrier", "simple", "meta_label"] = Field(
        default="triple_barrier", description="Label generation method (triple_barrier recommended)"
    )
    max_holding_period: int = Field(
        default=24, ge=1, le=168, description="Maximum holding period in candles for triple barrier"
    )


class TradingConfig(BaseModel):
    """Main trading configuration."""

    # Basic settings
    dry_run: bool = Field(
        default=True, description="CRITICAL: Set to False only for live trading with real money!"
    )
    pairs: List[str] = Field(default=["BTC/USDT"], min_length=1, description="Trading pairs")
    timeframe: str = Field(default="1h", description="Main trading timeframe")
    stake_currency: str = Field(default="USDT", description="Quote currency for stake")
    stake_amount: float = Field(default=100.0, ge=1.0, description="Stake amount per trade")
    max_open_trades: int = Field(
        default=3, ge=1, le=20, description="Maximum concurrent open trades"
    )
    leverage: float = Field(
        default=1.0, ge=1.0, le=20.0, description="Leverage (1.0 = no leverage, be careful!)"
    )

    # Sub-configs
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    # Strategy settings
    strategy: str = Field(default="StoicCitadel", description="Strategy class name")
    use_regime_filter: bool = Field(
        default=True, description="Filter trades based on market regime"
    )
    use_smart_orders: bool = Field(
        default=True, description="Use smart limit orders for fee optimization"
    )

    if PYDANTIC_V2:

        @field_validator("pairs")
        @classmethod
        def validate_pairs(cls, v):
            for pair in v:
                if "/" not in pair:
                    raise ValueError(f"Invalid pair format: {pair}. Expected 'BASE/QUOTE'")
            return v

        @field_validator("timeframe")
        @classmethod
        def validate_timeframe(cls, v):
            valid = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
            if v not in valid:
                raise ValueError(f"Invalid timeframe: {v}. Valid: {valid}")
            return v

        @field_validator("leverage")
        @classmethod
        def validate_leverage(cls, v):
            if v > 5.0:
                logger.warning(f"Leverage {v}x is high! Consider lower leverage for safety.")
            return v

        @model_validator(mode="after")
        def validate_live_trading(self):
            if not self.dry_run and self.exchange.sandbox:
                logger.warning(
                    "dry_run=False but sandbox=True. " "This will trade on testnet, not mainnet."
                )
            if not self.dry_run and self.leverage > 1.0:
                logger.warning(
                    f"LIVE TRADING with {self.leverage}x leverage! "
                    f"Make sure you understand the risks."
                )
            return self

    else:

        @validator("pairs", each_item=True)
        def validate_pairs(cls, v):
            if "/" not in v:
                raise ValueError(f"Invalid pair format: {v}. Expected 'BASE/QUOTE'")
            return v

        @validator("timeframe")
        def validate_timeframe(cls, v):
            valid = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
            if v not in valid:
                raise ValueError(f"Invalid timeframe: {v}. Valid: {valid}")
            return v

        @validator("leverage")
        def validate_leverage(cls, v):
            if v > 5.0:
                logger.warning(f"Leverage {v}x is high! Consider lower leverage for safety.")
            return v

        @root_validator
        def validate_live_trading(cls, values):
            dry_run = values.get("dry_run", True)
            exchange = values.get("exchange", ExchangeConfig())
            leverage = values.get("leverage", 1.0)

            if not dry_run and exchange.sandbox:
                logger.warning(
                    "dry_run=False but sandbox=True. " "This will trade on testnet, not mainnet."
                )
            if not dry_run and leverage > 1.0:
                logger.warning(
                    f"LIVE TRADING with {leverage}x leverage! "
                    f"Make sure you understand the risks."
                )
            return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        if PYDANTIC_V2:
            return self.model_dump()
        else:
            return self.dict()

    def to_json(self, path: Optional[str] = None) -> str:
        """Export to JSON."""
        if PYDANTIC_V2:
            json_str = self.model_dump_json(indent=2)
        else:
            json_str = self.json(indent=2)

        if path:
            Path(path).write_text(json_str)
            logger.info(f"Config saved to {path}")

        return json_str

    @classmethod
    def from_json(cls, path: str) -> "TradingConfig":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "TradingConfig":
        """Load from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required. Run: pip install pyyaml")

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def validate_for_live_trading(self) -> List[str]:
        """
        Validate config is safe for live trading.

        Returns list of warnings/issues found.
        """
        issues = []

        if self.dry_run:
            issues.append("dry_run is True - this is paper trading mode")

        if self.exchange.sandbox:
            issues.append("sandbox mode enabled - trading on testnet")

        if not self.exchange.api_key:
            issues.append("No API key configured")

        if not self.exchange.api_secret:
            issues.append("No API secret configured")

        # Stricter validation for live trading
        if self.leverage > 5.0:
            issues.append(f"High leverage ({self.leverage}x) - consider reducing")
        elif self.leverage > 3.0 and not self.dry_run:
            issues.append(f"Moderate leverage ({self.leverage}x) - ensure risk management")

        # Position size limits for live trading
        if self.risk.max_position_pct > 0.25 and not self.dry_run:
            issues.append(f"Large position size ({self.risk.max_position_pct:.0%}) - max 25% recommended for live trading")
        elif self.risk.max_position_pct > 0.50:
            issues.append(f"Very large position size ({self.risk.max_position_pct:.0%}) - max 50% allowed")

        # Risk limits
        if self.risk.max_drawdown_pct > 0.30:
            issues.append(f"High max drawdown ({self.risk.max_drawdown_pct:.0%}) - max 30% recommended")
        if self.risk.max_daily_loss_pct > 0.10:
            issues.append(f"High daily loss limit ({self.risk.max_daily_loss_pct:.0%}) - max 10% recommended")

        # Strategy safety
        if not self.use_regime_filter:
            issues.append("Regime filter disabled - may trade in unfavorable conditions")

        if not self.use_smart_orders:
            issues.append("Smart orders disabled - paying higher fees")

        # Portfolio concentration
        if len(self.pairs) < 3 and not self.dry_run:
            issues.append(f"Low diversification ({len(self.pairs)} pairs) - consider more pairs")

        # Concurrent trades
        if self.max_open_trades > 5 and not self.dry_run:
            issues.append(f"High concurrent trades ({self.max_open_trades}) - increases risk")

        return issues


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    start_date: str = Field(description="Start date YYYYMMDD format")
    end_date: str = Field(description="End date YYYYMMDD format")
    starting_balance: float = Field(
        default=10000.0, ge=100.0, description="Starting balance for backtest"
    )
    fee_pct: float = Field(
        default=0.001, ge=0.0, le=0.01, description="Trading fee percentage (0.001 = 0.1%)"
    )
    slippage_pct: float = Field(
        default=0.0005, ge=0.0, le=0.01, description="Slippage percentage (0.0005 = 0.05%)"
    )
    use_walk_forward: bool = Field(default=True, description="Use walk-forward validation")
    walk_forward_windows: int = Field(
        default=5, ge=2, le=20, description="Number of walk-forward windows"
    )

    if PYDANTIC_V2:

        @field_validator("start_date", "end_date")
        @classmethod
        def validate_date_format(cls, v):
            if len(v) != 8 or not v.isdigit():
                raise ValueError(f"Date must be YYYYMMDD format, got: {v}")
            return v

        @model_validator(mode="after")
        def validate_date_range(self):
            if self.start_date >= self.end_date:
                raise ValueError("start_date must be before end_date")
            return self

    else:

        @validator("start_date", "end_date")
        def validate_date_format(cls, v):
            if len(v) != 8 or not v.isdigit():
                raise ValueError(f"Date must be YYYYMMDD format, got: {v}")
            return v

        @root_validator
        def validate_date_range(cls, values):
            start = values.get("start_date")
            end = values.get("end_date")
            if start and end and start >= end:
                raise ValueError("start_date must be before end_date")
            return values


# Convenience function
def load_config(path: str) -> TradingConfig:
    """
    Load configuration from file.

    Supports JSON and YAML formats (auto-detected by extension).
    """
    path_obj = Path(path)

    if path_obj.suffix in [".yaml", ".yml"]:
        return TradingConfig.from_yaml(path)
    else:
        return TradingConfig.from_json(path)
