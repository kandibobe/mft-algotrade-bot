"""
Stoic Citadel - Base Strategy Logic
===================================

Provides a common foundation for all Stoic Citadel strategies.
Centralizes indicators, ML, and risk management.
"""

import logging
from datetime import datetime

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from pandas import DataFrame

from src.strategies.core_logic import StoicLogic
from src.strategies.hybrid_connector import HybridConnectorMixin
from src.strategies.ml_adapter import StrategyMLAdapter
from src.strategies.risk_mixin import StoicRiskMixin
from src.utils.logger import log as stoic_log

# Internal Imports
from src.utils.regime_detection import calculate_regime

logger = logging.getLogger(__name__)


class BaseStoicStrategy(HybridConnectorMixin, StoicRiskMixin, IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    startup_candle_count = 500

    # Common Parameters (Hyperoptable)
    buy_rsi = IntParameter(10, 60, default=30, space="buy")
    sell_rsi = IntParameter(60, 95, default=75, space="sell")
    regime_vol_threshold = DecimalParameter(0.5, 3.0, default=0.5, space="buy")
    regime_adx_threshold = IntParameter(10, 40, default=25, space="buy")
    regime_hurst_threshold = DecimalParameter(0.40, 0.60, default=0.55, space="buy")
    risk_per_trade = DecimalParameter(0.005, 0.02, default=0.01, space="sell")
    max_equity_drawdown = DecimalParameter(0.05, 0.20, default=0.10, space="sell")

    # Liquidity Filter Parameters
    min_volume_1h = DecimalParameter(
        100000.0, 1000000.0, default=500000.0, space="buy", optimize=True
    )
    max_spread_pct = DecimalParameter(0.01, 0.2, default=0.05, space="buy", optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._correlation_manager = None
        self._alt_data_fetcher = None
        self.last_alt_data = {}
        self._ml_adapters = {}

    def informative_pairs(self):
        return [("BTC/USDT:USDT", "1h"), ("ETH/USDT:USDT", "1h")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.columns = dataframe.columns.str.lower()
        try:
            # 1. Technical Indicators
            dataframe = StoicLogic.populate_indicators(dataframe)

            # 1b. Liquidity Metrics
            dataframe["rolling_volume_1h"] = (
                dataframe["volume"].rolling(window=12).sum()
            )  # 12 * 5m = 1h

            # 2. Regime Detection
            regime_df = calculate_regime(
                dataframe["high"],
                dataframe["low"],
                dataframe["close"],
                dataframe["volume"],
                vol_threshold=float(self.regime_vol_threshold.value),
                adx_threshold=float(self.regime_adx_threshold.value),
                hurst_threshold=float(self.regime_hurst_threshold.value),
            )
            dataframe["regime"] = regime_df["regime"]
            dataframe["hurst"] = regime_df["hurst"]
            dataframe["adx"] = regime_df["adx"]
            dataframe["vol_zscore"] = regime_df["vol_zscore"]

            # 3. Broad Market Trend (Informative BTC)
            if self.dp:
                inf_btc = self.dp.get_pair_dataframe("BTC/USDT:USDT", "1h")
                if not inf_btc.empty:
                    inf_btc["ema_200"] = StoicLogic.calculate_ema(inf_btc["close"], 200)
                    dataframe = merge_informative_pair(dataframe, inf_btc, self.timeframe, "1h")

            # 4. ML Predictions
            # Skip ML in hyperopt to avoid pickling issues
            if self.config.get("runmode") != "hyperopt":
                dataframe = self._calculate_ml_predictions(dataframe, metadata)
            else:
                dataframe["ml_prediction"] = 0.5

        except Exception:
            if "ml_prediction" not in dataframe.columns:
                dataframe["ml_prediction"] = 0.5
        return dataframe

    def _calculate_ml_predictions(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            pair = metadata["pair"]
            if pair not in self._ml_adapters:
                self._ml_adapters[pair] = StrategyMLAdapter(pair)
            dataframe["ml_prediction"] = self._ml_adapters[pair].get_predictions(dataframe)
        except Exception:
            dataframe["ml_prediction"] = 0.5
        return dataframe

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
        # 1. Hybrid Safety Check
        if not self.check_market_safety(pair, side):
            return False

        # 2. MFT Execution Diversion (Live/Dry Run Only)
        if self.config.get("runmode") in ("live", "dry_run"):
            # Double check if we have a connection
            if self._executor:
                try:
                    from src.order_manager.smart_order import ChaseLimitOrder

                    # Get the features that led to this decision
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    features = dataframe.iloc[-1].to_dict()

                    # Clean up for JSON serialization
                    for key, value in features.items():
                        if isinstance(value, (np.integer, np.int64)):
                            features[key] = int(value)
                        elif isinstance(value, (np.floating, np.float64)):
                            features[key] = float(value)
                        elif isinstance(value, (np.bool_)):
                            features[key] = bool(value)
                        elif pd.isna(value):
                            features[key] = None
                            
                    # Construct Smart Order
                    smart_order = ChaseLimitOrder(
                        symbol=pair,
                        side=side,
                        quantity=amount,
                        price=rate,
                        attribution_metadata={
                            "strategy": self.get_strategy_name(),
                            "tag": entry_tag,
                        },
                        signal_features=features,
                    )

                    # Submit to Async Executor
                    order_id = self.submit_smart_order(smart_order)

                    if order_id:
                        logger.info(f"⚡ MFT Order Submitted: {order_id} for {pair} {side}")
                        # Return False to prevent Freqtrade from placing a duplicate dumb order
                        return False
                    else:
                        logger.error("Failed to submit MFT order. Execution blocked.")
                        return False
                except Exception as e:
                    logger.critical(f"Error in MFT execution diversion: {e}", exc_info=True)
                    return False

        return super().confirm_trade_entry(
            pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side, **kwargs
        )

    def bot_start(self, **kwargs) -> None:
        runmode = self.config.get("runmode")
        if runmode in ("live", "dry_run"):
            from src.config.manager import ConfigurationManager

            ConfigurationManager.initialize()
            super().bot_start(**kwargs)
            config_obj = ConfigurationManager.get_config()
            pairs = config_obj.pairs if hasattr(config_obj, "pairs") else []
            self.initialize_hybrid_connector(pairs=pairs, risk_manager=self.risk_manager)
            stoic_log.info("base_strategy_initialized", runmode=runmode)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Placeholder - Override in Strategy."""
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Placeholder - Override in Strategy."""
        return dataframe
