
"""
Unified Configuration System with Pydantic Validation
=====================================================

This module provides a comprehensive, type-safe configuration system that:
1. Loads configuration from multiple sources (YAML, JSON, environment variables)
2. Validates all configuration values with Pydantic
3. Provides dot notation access (config.exchange.name instead of config['exchange']['name'])
4. Fails fast at startup if configuration is invalid

Key Features:
- Type safety with Pydantic BaseSettings
- Environment variable support with validation_alias
- Range validation using Field constraints
- Custom validators for business logic
- Hot-reload capability
- Support for YAML, JSON, and environment variables

Usage:
    from src.config.unified_config import TradingConfig, load_config

    # Load from YAML file
    config = load_config("config/strategy_config.yaml")

    # Access with dot notation (type-safe!)
    print(config.exchange.name)
    print(config.risk.max_position_pct)

    # Validate for live trading
    issues = config.validate_for_live_trading()
    if issues:
        print("Warnings:", issues)
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.utils.secret_manager import SecretManager

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    try:
        # Fallback for Pydantic v2 without pydantic-settings installed
        from pydantic.v1 import (
            BaseModel,
            BaseSettings,
            Field,
            root_validator,
        )
        from pydantic.v1 import (
            validator as field_validator,
        )

        # Shim for model_validator
        def model_validator(mode):
            def decorator(func):
                return root_validator(pre=(mode == "before"))(func)

            return decorator

        SettingsConfigDict = None
        ConfigDict = None
    except ImportError:
        # Fallback for Pydantic v1
        from pydantic import (
            BaseModel,
            BaseSettings,
            Field,
            root_validator,
        )
        from pydantic import (
            validator as field_validator,
        )

        # Shim for model_validator
        def model_validator(mode):
            def decorator(func):
                return root_validator(pre=(mode == "before"))(func)

            return decorator

        SettingsConfigDict = None
        ConfigDict = None

logger = logging.getLogger(__name__)


class ExchangeConfig(BaseModel):
    """Exchange connection configuration with validation."""

    name: str = Field(default="binance", description="Exchange name (binance, bybit, okx, etc.)")
    sandbox: bool = Field(
        default=True, description="Use testnet/sandbox mode (CRITICAL for testing!)"
    )
    api_key: str | None = Field(default=None, description="API key (keep secret!)")
    api_secret: str | None = Field(default=None, description="API secret (keep secret!)")
    rate_limit: bool = Field(default=True, description="Enable rate limiting to avoid bans")

    @model_validator(mode="after")
    def decrypt_secrets(self) -> "ExchangeConfig":
        """Automatically decrypt secrets if they are encrypted."""
        if self.api_key:
            self.api_key = SecretManager.get_secret(self.api_key)
        if self.api_secret:
            self.api_secret = SecretManager.get_secret(self.api_secret)
        return self

    timeout_ms: int = Field(
        default=30000, ge=1000, le=120000, description="Request timeout in milliseconds"
    )

    @field_validator("name")
    @classmethod
    def validate_exchange(cls, v: str) -> str:
        """Validate exchange name and convert to lowercase."""
        supported = ["binance", "bybit", "okx", "kucoin", "gate", "coinbase", "kraken"]
        if v.lower() not in supported:
            logger.warning(f"Exchange '{v}' may not be fully supported")
        return v.lower()


class RiskConfig(BaseModel):
    """Risk management configuration with validation."""

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

    # Liquidation Guard
    liquidation_buffer: float = Field(
        default=0.20, ge=0.05, le=0.50, description="Safety buffer distance to liquidation price"
    )
    max_safe_leverage: float = Field(
        default=3.0, ge=1.0, le=20.0, description="Max allowed leverage"
    )

    # Correlation Guard
    max_correlation: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Max allowed correlation between assets"
    )
    correlation_pairs_block: list[str] = Field(
        default=["BTC", "ETH"], description="Pairs to check for directional correlation blocking"
    )
    correlation_threshold_block: float = Field(
        default=0.80, ge=0.0, le=1.0, description="Correlation threshold for blocking same-side trades"
    )

    @model_validator(mode="after")
    def validate_risk_reward(self) -> "RiskConfig":
        """Validate that take profit is greater than stop loss for positive risk/reward."""
        if self.take_profit_pct is not None and self.stop_loss_pct is not None:
            if self.take_profit_pct < self.stop_loss_pct:
                logger.warning(
                    f"Take profit ({self.take_profit_pct}) < Stop loss ({self.stop_loss_pct}). "
                    f"Consider adjusting for positive risk/reward ratio."
                )
        return self


class StrategyConfig(BaseModel):
    """Strategy-specific configuration from YAML files."""

    # General
    name: str = Field(default="StoicEnsembleStrategy", description="Strategy name")
    version: str = Field(default="1.0.0", description="Strategy version")
    timeframe: str = Field(default="5m", description="Trading timeframe")
    startup_candle_count: int = Field(default=200, ge=50, le=1000, description="Startup candles")

    # Risk Management
    risk_per_trade: float = Field(default=0.02, ge=0.001, le=0.10, description="Risk per trade")
    max_positions: int = Field(default=3, ge=1, le=20, description="Max concurrent positions")
    max_drawdown: float = Field(default=0.15, ge=0.05, le=0.50, description="Max drawdown")
    stoploss: float = Field(default=-0.05, le=0.0, description="Hard stop loss")
    trailing_stop: bool = Field(default=True, description="Enable trailing stop")
    trailing_stop_positive: float = Field(default=0.01, ge=0.0, description="Trailing start")
    trailing_stop_positive_offset: float = Field(
        default=0.015, ge=0.0, description="Trailing offset"
    )

    # Entry Parameters
    min_adx: float = Field(default=20.0, ge=0.0, le=100.0, description="Minimum ADX")
    ml_confidence_base: float = Field(default=0.60, ge=0.1, le=0.95, description="Base ML confidence threshold")
    ml_confidence_max: float = Field(default=0.90, ge=0.1, le=0.95, description="Max ML confidence threshold in high volatility")
    hurst_trending_threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Hurst threshold for trending regime")
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=100.0, description="RSI oversold")
    rsi_overbought: float = Field(default=70.0, ge=0.0, le=100.0, description="RSI overbought")
    stoch_oversold: float = Field(default=30.0, ge=0.0, le=100.0, description="Stoch oversold")
    stoch_overbought: float = Field(default=80.0, ge=0.0, le=100.0, description="Stoch overbought")
    min_volume_ratio: float = Field(default=0.8, ge=0.0, le=5.0, description="Min volume ratio")
    min_bb_width: float = Field(default=0.02, ge=0.0, description="Min Bollinger Band width")
    max_bb_width: float = Field(default=0.20, ge=0.0, description="Max Bollinger Band width")

    # Exit Parameters
    exit_rsi_overbought: float = Field(default=75.0, ge=0.0, le=100.0, description="Exit RSI")
    exit_stoch_overbought: float = Field(default=80.0, ge=0.0, le=100.0, description="Exit Stoch")

    # ROI Table
    roi_0: float = Field(default=0.15, ge=0.0, description="Immediate ROI")
    roi_30: float = Field(default=0.08, ge=0.0, description="30-minute ROI")
    roi_60: float = Field(default=0.05, ge=0.0, description="60-minute ROI")
    roi_120: float = Field(default=0.03, ge=0.0, description="120-minute ROI")

    # Regime Detection
    regime_aware: bool = Field(default=True, description="Enable regime detection")
    regime_adx_threshold: float = Field(default=25.0, ge=0.0, le=100.0, description="Regime ADX")
    regime_aggressive_score: float = Field(
        default=70.0, ge=0.0, le=100.0, description="Aggressive score"
    )
    regime_defensive_score: float = Field(
        default=40.0, ge=0.0, le=100.0, description="Defensive score"
    )

    # Fees & Slippage
    fee: float = Field(default=0.001, ge=0.0, le=0.01, description="Trading fee")
    slippage_entry: float = Field(default=0.0005, ge=0.0, le=0.01, description="Entry slippage")
    slippage_exit: float = Field(default=0.0005, ge=0.0, le=0.01, description="Exit slippage")

    # Protections
    protection_stoploss_guard: bool = Field(default=True, description="Enable stop loss guard")
    max_slippage_pct: float = Field(default=0.5, ge=0.0, le=5.0, description="Max allowed slippage percentage")
    protection_stoploss_lookback: int = Field(default=60, ge=1, description="Stop loss lookback")
    protection_stoploss_limit: int = Field(default=3, ge=1, description="Stop loss limit")
    protection_cooldown: int = Field(default=24, ge=1, description="Cooldown period")

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe string."""
        valid = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
        if v not in valid:
            raise ValueError(f"Invalid timeframe: {v}. Valid: {valid}")
        return v


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
        default="triple_barrier", description="Label generation method"
    )
    max_holding_period: int = Field(
        default=24, ge=1, le=168, description="Maximum holding period in candles"
    )


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""

    target_variable: str = Field(default="target", description="Name of target variable")
    model_type: Literal["lightgbm", "xgboost", "random_forest", "catboost"] = Field(
        default="lightgbm", description="ML model type"
    )
    hyperopt_trials: int = Field(
        default=100, ge=1, le=1000, description="Number of hyperparameter optimization trials"
    )
    validation_split: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation data split ratio"
    )
    feature_set_version: str = Field(default="v1", description="Feature set version for tracking")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all cores)")
    early_stopping_rounds: int = Field(default=10, description="Early stopping rounds")


class TelegramConfig(BaseModel):
    """Telegram bot configuration."""

    token: str | None = Field(default=None, description="Telegram Bot Token")
    chat_id: str | None = Field(default=None, description="Telegram Chat ID")
    enabled: bool = Field(default=False, description="Enable Telegram notifications")


class PathConfig(BaseModel):
    """Centralized path management for portability."""

    user_data_dir: Path = Field(default=Path("user_data"), description="User data directory")
    data_dir: Path = Field(default=Path("user_data/data"), description="Feather/Parquet data directory")
    models_dir: Path = Field(default=Path("user_data/models"), description="ML models directory")
    logs_dir: Path = Field(default=Path("user_data/logs"), description="Logs directory")
    db_url: str = Field(default="sqlite:///user_data/stoic_citadel.db", description="Database connection URL")

    @model_validator(mode="before")
    def resolve_paths(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure all paths are resolved relative to the project root."""
        # This assumes the script is run from the project root.
        # A more robust solution might use a known file as an anchor.
        project_root = Path().absolute()

        # Resolve paths if they are provided, otherwise use defaults
        for key, value in values.items():
            if isinstance(value, str) and "dir" in key:
                values[key] = (project_root / value).resolve()

        # Handle db_url separately
        if "db_url" in values and values["db_url"].startswith("sqlite:///"):
            db_path_str = values["db_url"].replace("sqlite:///", "")
            db_path = (project_root / db_path_str).resolve()
            values["db_url"] = f"sqlite:///{db_path}"

        return values

class SystemConfig(BaseModel):
    """System-level configuration."""

    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    db_url: str = Field(default="sqlite:///user_data/stoic_citadel.db", description="Database URL")
    sentry_dsn: str | None = Field(default=None, description="Sentry DSN for error tracking")
    ws_watchdog_timeout: float = Field(default=5.0, ge=1.0, le=60.0, description="WebSocket watchdog timeout in seconds")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        supported = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in supported:
            raise ValueError(f"Invalid log level: {v}. Supported: {supported}")
        return v.upper()

class FeatureStoreConfig(BaseModel):
    """Feature Store configuration."""
    
    enabled: bool = Field(default=False, description="Enable Feature Store")
    use_redis: bool = Field(default=False, description="Use Redis for online store")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    config_path: str = Field(default="feature_repo", description="Path to feature repo")


class TradingConfig(BaseSettings):
    """Main trading configuration with environment variable support."""

    if SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=True,
        )
    else:

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"
            case_sensitive = True

    # Basic settings
    dry_run: bool = Field(
        default=True, description="CRITICAL: Set to False only for live trading with real money!"
    )
    pairs: list[str] = Field(default=["BTC/USDT"], min_length=1, description="Trading pairs")
    timeframe: str = Field(default="1h", description="Main trading timeframe")
    stake_currency: str = Field(default="USDT", description="Quote currency for stake")
    stake_amount: float = Field(default=100.0, ge=1.0, description="Stake amount per trade")
    max_open_trades: int = Field(
        default=3, ge=1, le=20, description="Maximum concurrent open trades"
    )
    leverage: float = Field(
        default=1.0, ge=1.0, le=20.0, description="Leverage (1.0 = no leverage)"
    )

    # Sub-configs
    exchange: ExchangeConfig = Field(default_factory=lambda: ExchangeConfig(rate_limit=True))
    additional_exchanges: list[ExchangeConfig] = Field(default_factory=list, description="Secondary exchanges for arbitrage")
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    strategy: StrategyConfig | str | None = Field(default=None, description="Strategy configuration")

    # Strategy settings
    strategy_name: str = Field(default="StoicCitadel", description="Strategy class name")
    use_regime_filter: bool = Field(
        default=True, description="Filter trades based on market regime"
    )
    use_smart_orders: bool = Field(
        default=True, description="Use smart limit orders for fee optimization"
    )

    # Environment variables for API credentials
    api_key: str | None = Field(default=None, validation_alias="BINANCE_API_KEY")
    api_secret: str | None = Field(default=None, validation_alias="BINANCE_API_SECRET")

    # Telegram environment variables
    telegram_token: str | None = Field(default=None, validation_alias="TELEGRAM_TOKEN")
    telegram_chat_id: str | None = Field(default=None, validation_alias="TELEGRAM_CHAT_ID")

    # Alternative Data Sources (Optional)
    fng_enabled: bool = Field(default=True, description="Enable Fear & Greed Index")
    coingecko_enabled: bool = Field(default=True, description="Enable CoinGecko Data")
    defillama_enabled: bool = Field(default=True, description="Enable DefiLlama Data")

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, v: list[str]) -> list[str]:
        """Validate trading pair format."""
        for pair in v:
            if "/" not in pair:
                raise ValueError(f"Invalid pair format: {pair}. Expected 'BASE/QUOTE'")
        return v

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe string."""
        valid = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
        if v not in valid:
            raise ValueError(f"Invalid timeframe: {v}. Valid: {valid}")
        return v

    @field_validator("leverage")
    @classmethod
    def validate_leverage(cls, v: float) -> float:
        """Warn about high leverage."""
        if v > 5.0:
            logger.warning(f"Leverage {v}x is high! Consider lower leverage for safety.")
        return v

    @model_validator(mode="after")
    def validate_live_trading(self) -> "TradingConfig":
        """Validate configuration for live trading safety."""
        # Handle case where strategy is a string (legacy Freqtrade config)
        if isinstance(self.strategy, str):
            self.strategy_name = self.strategy
            self.strategy = None

        # Sync environment variables to telegram config if provided
        if self.telegram:
            if self.telegram_token and not self.telegram.token:
                self.telegram.token = self.telegram_token
            if self.telegram_chat_id and not self.telegram.chat_id:
                self.telegram.chat_id = self.telegram_chat_id

            # Auto-enable if token and chat_id are present
            if self.telegram.token and self.telegram.chat_id and not self.telegram.enabled:
                self.telegram.enabled = True

        if not self.dry_run and self.exchange and self.exchange.sandbox:
            logger.warning(
                "dry_run=False but sandbox=True. This will trade on testnet, not mainnet."
            )
        if not self.dry_run and self.leverage > 1.0:
            logger.warning(
                f"LIVE TRADING with {self.leverage}x leverage! Make sure you understand the risks."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    def to_json(self, path: str | None = None) -> str:
        """Export to JSON."""
        json_str = self.model_dump_json(indent=2)
        if path:
            Path(path).write_text(json_str)
            logger.info(f"Config saved to {path}")
        return json_str

    def reload(self, config_path: str | None = None) -> None:
        """
        Reload configuration from file.
        Updates the current instance with new values.
        """
        if not config_path:
            logger.error("Reload failed: config_path is required")
            return

        # Use a small retry mechanism for file reading to avoid race conditions
        # when a file is being written while we try to read it.
        max_retries = 5
        last_exception = None
        for i in range(max_retries):
            try:
                new_config = load_config(config_path)
                # Update current fields with new values
                # In Pydantic V2, we use self.__class__.model_fields to avoid deprecation warning
                fields = getattr(self.__class__, "model_fields", getattr(self, "model_fields", []))
                for field in fields:
                    new_value = getattr(new_config, field)
                    setattr(self, field, new_value)
                logger.info(f"Configuration reloaded from {config_path}")
                return
            except Exception as e:
                last_exception = e
                time.sleep(0.1 * (i + 1))

        logger.error(f"Failed to reload configuration after {max_retries} attempts: {last_exception}")

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

    def validate_for_live_trading(self) -> list[str]:
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
            issues.append(
                f"Large position size ({self.risk.max_position_pct:.0%}) - max 25% recommended for live trading"
            )
        elif self.risk.max_position_pct > 0.50:
            issues.append(
                f"Very large position size ({self.risk.max_position_pct:.0%}) - max 50% allowed"
            )

        # Risk limits
        if self.risk.max_drawdown_pct > 0.30:
            issues.append(
                f"High max drawdown ({self.risk.max_drawdown_pct:.0%}) - max 30% recommended"
            )
        if self.risk.max_daily_loss_pct > 0.10:
            issues.append(
                f"High daily loss limit ({self.risk.max_daily_loss_pct:.0%}) - max 10% recommended"
            )

        # Stop loss validation
        if self.risk.stop_loss_pct > 0.10:
            issues.append(f"Large stop loss ({self.risk.stop_loss_pct:.0%}) - max 10% recommended")

        # Risk/reward ratio validation
        if self.risk.take_profit_pct / self.risk.stop_loss_pct < 1.5:
            issues.append(
                f"Low risk/reward ratio ({self.risk.take_profit_pct / self.risk.stop_loss_pct:.1f}:1) - aim for at least 1.5:1"
            )

        # Strategy validation
        if not self.strategy:
            issues.append("No strategy configuration loaded")
        elif self.strategy.risk_per_trade > 0.05:
            issues.append(
                f"High strategy risk per trade ({self.strategy.risk_per_trade:.0%}) - max 5% recommended"
            )

        return issues


# Global cache for the configuration instance
_CONFIG_CACHE: dict[str, Any] = {}
_CONFIG_LOCK = threading.Lock()

def load_config(config_path: str | None = None) -> TradingConfig:
    """
    Load configuration from file or environment.
    Uses a Singleton pattern with caching for high performance in MFT loops.

    Args:
        config_path: Path to YAML or JSON config file (optional)

    Returns:
        TradingConfig instance
    """
    cache_key = config_path or "DEFAULT_ENV"
    
    # <Î Fast path: check cache without lock first
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]
        
    with _CONFIG_LOCK:
        # Double-check cache inside lock
        if cache_key in _CONFIG_CACHE:
            return _CONFIG_CACHE[cache_key]
            
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            if path.suffix.lower() in [".yaml", ".yml"]:
                cfg = TradingConfig.from_yaml(config_path)
            elif path.suffix.lower() == ".json":
                cfg = TradingConfig.from_json(config_path)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        else:
            # Load from environment variables only
            cfg = TradingConfig()
            
        _CONFIG_CACHE[cache_key] = cfg
        return cfg


class ConfigWatcher:
    """
    Watches configuration files for changes and triggers reload.
    """

    def __init__(self, config: TradingConfig, config_path: str):
        self.config = config
        self.config_path = str(Path(config_path).absolute())
        self.callbacks: list[Callable[[TradingConfig], None]] = []
        self._observer = Observer()
        self._handler = self._create_handler()
        self._last_reload_time = 0
        self._reload_debounce = 1.0  # seconds

    def _create_handler(self) -> FileSystemEventHandler:
        outer_self = self

        class ReloadHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and str(Path(event.src_path).absolute()) == outer_self.config_path:
                    current_time = time.time()
                    if current_time - outer_self._last_reload_time > outer_self._reload_debounce:
                        logger.info(f"Detected change in {outer_self.config_path}, reloading...")
                        outer_self.config.reload(outer_self.config_path)
                        outer_self._last_reload_time = current_time
                        for callback in outer_self.callbacks:
                            try:
                                callback(outer_self.config)
                            except Exception as e:
                                logger.error(f"Error in config reload callback: {e}")

        return ReloadHandler()

    def add_callback(self, callback: Callable[[TradingConfig], None]) -> None:
        """Add a callback to be called when config is reloaded."""
        self.callbacks.append(callback)

    def start(self) -> None:
        """Start watching the config file."""
        config_dir = str(Path(self.config_path).parent)
        self._observer.schedule(self._handler, config_dir, recursive=False)
        self._observer.start()
        logger.info(f"Started watching configuration at {self.config_path}")

    def stop(self) -> None:
        """Stop watching the config file."""
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Stopped watching configuration")