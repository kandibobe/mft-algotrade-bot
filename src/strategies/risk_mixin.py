"""
Stoic Citadel - Risk Management Mixin
=====================================

Provides integrated risk management for Freqtrade strategies.
Uses the central RiskManager to enforce strict safety checks.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from freqtrade.persistence import Trade

from src.config.manager import ConfigurationManager
from src.ml.online_learner import OnlineLearner
from src.risk.circuit_breaker import CircuitBreakerConfig
from src.risk.liquidation import LiquidationConfig
from src.risk.position_sizing import PositionSizingConfig

# Import Stoic Risk Components
from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class StoicRiskMixin:
    """
    Mixin class to add professional risk management to any Freqtrade strategy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_manager: RiskManager | None = None
        self.online_learner: OnlineLearner | None = None
        self._last_balance_update = datetime.min
        self._correlation_update_interval = timedelta(hours=1)
        self._last_correlation_update = datetime.min
        self._last_learning_update = datetime.min
        self._processed_trade_ids = set()

    def __getstate__(self):
        """Custom pickling to avoid unpicklable objects."""
        state = self.__dict__.copy()
        # Risk manager and online learner are recreated or not needed in workers
        # Also handle any logger or other non-picklable objects
        # Standardize on keys that may be present in strategy or mixins
        for key in ('risk_manager', 'online_learner', 'feature_store', '_ml_adapters', 'dp'):
            if key in state:
                state[key] = None
        return state

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of each bot iteration.
        """
        if hasattr(super(), "bot_loop_start"):
            super().bot_loop_start(**kwargs)

        if self.risk_manager and self.config.get("runmode") in ("live", "dry_run"):
            try:
                btc_df = self.dp.get_pair_dataframe("BTC/USDT", "1h")
                if btc_df is not None and not btc_df.empty:
                    last_candle = btc_df.iloc[-1]
                    price_change = (last_candle["close"] - last_candle["open"]) / last_candle["open"]
                    self.risk_manager.circuit_breaker.check_market_crash("BTC/USDT", price_change)
            except Exception as e:
                logger.warning(f"Market crash check failed: {e}")

        if datetime.now() - self._last_learning_update > timedelta(hours=1):
            self._learn_from_closed_trades()
            self._last_learning_update = datetime.now()

    def _learn_from_closed_trades(self):
        try:
            trades = Trade.get_trades([Trade.is_open.is_(False)]).all()
            for trade in trades:
                if trade.id not in self._processed_trade_ids:
                    self._processed_trade_ids.add(trade.id)
        except Exception as e:
            logger.warning(f"Online learning update failed: {e}")

    def bot_start(self, **kwargs) -> None:
        """
        Initialize Risk Manager on bot startup.
        """
        try:
            config = ConfigurationManager.get_config()
            cb_config = CircuitBreakerConfig(
                max_drawdown_pct=config.risk.max_drawdown_pct,
                daily_loss_limit_pct=config.risk.max_daily_loss_pct,
            )
            size_config = PositionSizingConfig(
                max_position_pct=config.risk.max_position_pct,
                max_portfolio_risk_pct=config.risk.max_portfolio_risk,
                max_correlation_exposure=config.risk.max_correlation,
            )
            liq_config = LiquidationConfig(
                safety_buffer=config.risk.liquidation_buffer,
                max_safe_leverage=config.risk.max_safe_leverage,
            )
            self.risk_manager = RiskManager(
                circuit_config=cb_config, sizing_config=size_config, liquidation_config=liq_config
            )
            logger.info("✅ Stoic Risk Mixin Initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Risk Manager: {e}")

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Final gatekeeper check before trade entry.
        """
        if not self.risk_manager:
            logger.warning(f"Risk Manager not initialized for {pair}. Allowing trade but check initialization.")
            return True

        # Evaluate trade with central Risk Manager
        # We use a placeholder stop loss if not provided in kwargs for safety evaluation
        sl_pct = self.config.get("risk", {}).get("stop_loss_pct", 0.02)
        stop_loss = rate * (1 - sl_pct) if side == "long" else rate * (1 + sl_pct)

        risk_check = self.risk_manager.evaluate_trade(
            symbol=pair,
            entry_price=rate,
            stop_loss_price=kwargs.get("stop_loss", stop_loss),
            side=side,
            **kwargs
        )

        if not risk_check["allowed"]:
            logger.warning(f"❌ Trade REJECTED by Risk Manager for {pair}: {risk_check.get('rejection_reason', 'Unknown')}")
            return False

        logger.info(f"✅ Trade APPROVED by Risk Manager for {pair}")
        return True

    def custom_exit(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, **kwargs,
    ) -> str | None:
        if self.risk_manager and self.risk_manager.emergency_exit:
            return "emergency_exit"
        return None

    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float | None, max_stake: float,
        leverage: float, entry_tag: str | None, side: str, **kwargs,
    ) -> float:
        return proposed_stake
        return proposed_stake
