"""
Configuration Manager
=====================

Single source of truth for application configuration.
Loads validated config and exposes it as a singleton.
"""

import json
import logging
import os

from src.config.unified_config import TradingConfig, load_config

logger = logging.getLogger(__name__)


class ConfigurationManager:
    _instance: TradingConfig | None = None
    _config_path: str | None = None

    @classmethod
    def initialize(cls, config_path: str | None = None) -> TradingConfig:
        """Initialize the configuration singleton."""
        if cls._instance is not None:
            logger.warning("Configuration already initialized, reloading...")

        # Priority: Argument > Environment Variable > Hardcoded Docker Path > Default
        path = config_path or os.getenv("STOIC_CONFIG_PATH")
        
        # Fallback for Docker deployment if no env var is set
        if not path and os.path.exists("/freqtrade/user_data/config/config_production.yaml"):
             path = "/freqtrade/user_data/config/config_production.yaml"

        try:
            cls._instance = load_config(path)

            cls._config_path = path

            # Run safety checks
            issues = cls._instance.validate_for_live_trading()
            for issue in issues:
                logger.warning(f"Config Warning: {issue}")

            logger.info(f"Configuration loaded successfully from {path or 'env'}")
            return cls._instance
        except Exception as e:
            logger.critical(f"Failed to load configuration: {e}")
            raise

    @classmethod
    def get_config(cls) -> TradingConfig:
        """Get the configuration instance."""
        if cls._instance is None:
            # Lazy initialization with default/env
            return cls.initialize()
        return cls._instance

    @classmethod
    def export_freqtrade_config(cls, output_path: str = "config.json") -> None:
        """
        Generate Freqtrade-compatible JSON config from internal config.
        This bridges the gap between our Pydantic config and Freqtrade.
        """
        if cls._instance is None:
            cls.initialize()

        cfg = cls._instance

        freqtrade_config = {
            "max_open_trades": cfg.max_open_trades,
            "stake_currency": cfg.stake_currency,
            "stake_amount": "unlimited" if cfg.stake_amount == -1 else cfg.stake_amount,
            "tradable_balance_ratio": 0.99,
            "fiat_display_currency": "USD",
            "dry_run": cfg.dry_run,
            "dry_run_wallet": 1000,
            "cancel_open_orders_on_exit": False,
            "trading_mode": "spot",
            "margin_mode": "isolated",
            "unidirectional_wills": True,
            "dataformat_ohlcv": "json",
            "exchange": {
                "name": cfg.exchange.name,
                "key": cfg.exchange.api_key or "",
                "secret": cfg.exchange.api_secret or "",
                "ccxt_config": {"enableRateLimit": True},
                "ccxt_async_config": {"enableRateLimit": True, "rateLimit": 200},
                "pair_whitelist": cfg.pairs,
                "pair_blacklist": [],
            },
            "timeframe": cfg.timeframe,
            "entry_pricing": {
                "price_side": "same",
                "use_order_book": True,
                "order_book_top": 1,
                "price_last_balance": 0.0,
                "check_depth_of_market": {"enabled_in_dry_run": False, "bids_to_ask_delta": 1},
            },
            "exit_pricing": {"price_side": "same", "use_order_book": True, "order_book_top": 1},
            "pairlists": [
                {"method": "StaticPairList"},
            ],
            "telegram": {"enabled": False, "token": "", "chat_id": ""},
            "api_server": {
                "enabled": True,
                "listen_ip_address": "0.0.0.0",
                "listen_port": 8080,
                "username": "admin",
                "password": os.environ.get("API_SERVER_PASSWORD", "change-this-password"),
                "verbosity": "info",
                "jwt_secret_key": os.environ.get("API_SERVER_JWT_SECRET", "change-this-secret-key"),
            },
            "bot_name": "StoicCitadel",
            "initial_state": "running",
            "force_entry_enable": True,
            "internals": {"process_throttle_secs": 5},
        }

        with open(output_path, "w") as f:
            json.dump(freqtrade_config, f, indent=4)

        logger.info(f"Generated Freqtrade config at {output_path}")


# Global accessor
config = ConfigurationManager.get_config
