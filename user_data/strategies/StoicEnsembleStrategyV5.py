"""
Stoic Citadel - Ensemble Strategy V5 (Regime-Adaptive)
======================================================

Features:
1. Regime-Adaptive Logic: Switches behavior based on Hurst Exponent and Volatility Rank.
2. Statistical Robustness: Uses Percentile Ranks instead of fixed thresholds.
3. Random Walk Filter: Blocks trades when market lacks structure.
4. Volatility Scaling: Position sizing inversely proportional to volatility rank.

Philosophy: "Don't fight the regime. Surf the waves, fade the chop, sit out the noise."

Author: Stoic Citadel Team
Version: 5.2.0 (Regime Refactor)
"""

import sys
from pathlib import Path
# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Increase recursion depth for Cloudpickle
sys.setrecursionlimit(2000)

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from functools import reduce

import pandas as pd
import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter, merge_informative_pair
from freqtrade.persistence import Trade
from freqtrade.exceptions import DependencyException

# Imports
from src.utils.regime_detection import calculate_regime, get_market_regime
from src.risk.correlation import CorrelationManager
# We still use V4's ML integration helper functions if we want to keep ML
from src.ml.model_loader import get_production_model
# Import Risk Mixin
from src.strategies.risk_mixin import StoicRiskMixin
# Import Core Logic
from src.strategies.core_logic import StoicLogic
# Import Hybrid Connector
from src.strategies.hybrid_connector import HybridConnectorMixin

USE_CUSTOM_MODULES = True

logger = logging.getLogger(__name__)

class StoicEnsembleStrategyV5(HybridConnectorMixin, StoicRiskMixin, IStrategy):
    INTERFACE_VERSION = 3

    # Hyperparameters
    # Buy Params
    buy_rsi = IntParameter(10, 60, default=30, space="buy") # Fallback
    buy_rsi_trend = IntParameter(30, 70, default=50, space="buy")     # Higher RSI allowed in trend (dip buying)
    buy_rsi_mean_rev = IntParameter(10, 40, default=25, space="buy")  # Lower RSI needed for mean rev
    
    buy_persistence = IntParameter(1, 15, default=3, space="buy")
    
    # ML
    ml_weight = DecimalParameter(0.1, 0.9, default=0.5, space="buy")
    entry_prob_threshold = DecimalParameter(0.50, 0.90, default=0.49, space="buy")

    # Sell Params
    sell_rsi = IntParameter(60, 95, default=75, space="sell")
    
    # Risk (Position Sizing & Stoploss)
    risk_per_trade = DecimalParameter(0.005, 0.02, default=0.01, space="sell") # Risk per trade (0.5% - 2%)
    stoploss_atr_mult = DecimalParameter(1.0, 5.0, default=2.5, space="sell")
    stoploss_zscore_mult = DecimalParameter(1.0, 4.0, default=2.0, space="sell")
    
    # Regime Hyperopt (Level 3 Feature)
    regime_vol_threshold = DecimalParameter(0.5, 3.0, default=0.5, space="buy")
    regime_adx_threshold = IntParameter(10, 40, default=25, space="buy")
    regime_hurst_threshold = DecimalParameter(0.40, 0.60, default=0.55, space="buy")
    
    # Equity Curve Protection (Level 3 Feature)
    max_equity_drawdown = DecimalParameter(0.05, 0.20, default=0.10, space="sell") 
    
    # FreqAI Thresholds (Level 3 Feature)
    freqai_roi_threshold = DecimalParameter(0.005, 0.05, default=0.01, space="buy")
    freqai_confidence_threshold = DecimalParameter(0.5, 1.0, default=0.8, space="buy") # High Confidence = Low DI

    # Exits
    exit_profit_time_limit_candles = IntParameter(24, 120, default=24, space="sell")
    exit_profit_time_limit_roi = DecimalParameter(0.005, 0.03, default=0.01, space="sell")
    trailing_atr_mult = DecimalParameter(1.5, 3.5, default=2.0, space="sell")
    
    # Trailing Stop (Hyperoptable via 'trailing' space)
    trailing_stop = False 
    
    # Timeframe
    timeframe = '5m'
    startup_candle_count = 500  # Needed for robust Volatility Rank & Hurst

    # ROI (Dynamic via Hyperopt)
    minimal_roi = {
        "0": 0.20,
        "30": 0.10,
        "60": 0.05,
        "120": 0.02
    }
    
    stoploss = -0.10  # Fallback, overridden by custom_stoploss
    trailing_stop = False # We use custom trailing
    
    # Order Types (Institutional Standards)
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    use_exit_signal = True
    process_only_new_candles = True

    @property
    def protections(self):
        """
        Protections list.
        """
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,  # 4 hours
                "trade_limit": 1,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.05    # 5% drawdown
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,  # 2 hours
                "trade_limit": 2,
                "stop_duration_candles": 12,
                "only_per_pair": True
            }
        ]

    def informative_pairs(self):
        """
        Define informative pairs (BTC/ETH) for broad market analysis.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [("BTC/USDT:USDT", "1h"), ("ETH/USDT:USDT", "1h")]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate indicators."""
        # Ensure columns are lowercase
        dataframe.columns = dataframe.columns.str.lower()
        
        try:
            # 1. Standard Indicators (ATR, RSI, BB, etc.)
            try:
                dataframe = StoicLogic.populate_indicators(dataframe)
            except KeyError as e:
                # Fallback if StoicLogic fails (e.g. during tests with partial data)
                logger.error(f"StoicLogic.populate_indicators failed: {e}. Columns: {dataframe.columns.tolist()}")
                pass

            # Add aliases for backward compatibility and tests
            if 'bb_lower' in dataframe.columns:
                dataframe['bb_lowerband'] = dataframe['bb_lower']
            if 'bb_upper' in dataframe.columns:
                dataframe['bb_upperband'] = dataframe['bb_upper']
            if 'bb_middle' in dataframe.columns:
                dataframe['bb_middleband'] = dataframe['bb_middle']
            
            # MACD aliases
            if 'macd_signal' in dataframe.columns:
                dataframe['macdsignal'] = dataframe['macd_signal']
            if 'macd_hist' in dataframe.columns:
                dataframe['macdhist'] = dataframe['macd_hist']
            
            # Ensure ATR exists (alias check)
            # StoicLogic usually adds 'atr', but just in case
            if 'atr' not in dataframe.columns:
                # Try to calculate basic ATR
                try:
                    import talib.abstract as ta
                    dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
                except Exception:
                    pass

            # Volume Mean for legacy tests
            if 'volume' in dataframe.columns and 'volume_mean' not in dataframe.columns:
                dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

            # Stochastic aliases (slowk/slowd)
            if 'stoch_k' in dataframe.columns:
                dataframe['slowk'] = dataframe['stoch_k']
            if 'stoch_d' in dataframe.columns:
                dataframe['slowd'] = dataframe['stoch_d']
            
            # Add basic Stochastic if missing (legacy support)
            if 'slowk' not in dataframe.columns and 'high' in dataframe.columns:
                try:
                    low_min = dataframe['low'].rolling(window=14).min()
                    high_max = dataframe['high'].rolling(window=14).max()
                    k = 100 * (dataframe['close'] - low_min) / (high_max - low_min)
                    dataframe['slowk'] = k.fillna(50)
                    dataframe['slowd'] = k.rolling(window=3).mean().fillna(50)
                except Exception as e:
                    logger.warning(f"Failed to calculate Stochastic: {e}")

            # Safe EMA calculation
            try:
                if 'close' in dataframe.columns:
                    dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
            except Exception as e:
                logger.warning(f"Failed to calculate EMA 100: {e}")
            
            # 2. Regime Metrics (Z-Score, Hurst, Enum)
            # Calculates over entire history with Hyperoptable Thresholds
            regime_df = calculate_regime(
                dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'],
                vol_threshold=float(self.regime_vol_threshold.value),
                adx_threshold=float(self.regime_adx_threshold.value),
                hurst_threshold=float(self.regime_hurst_threshold.value)
            )
            
            # Merge Regime Data
            dataframe['regime'] = regime_df['regime']
            dataframe['vol_zscore'] = regime_df['vol_zscore']
            dataframe['hurst'] = regime_df['hurst']
            dataframe['adx'] = regime_df['adx']
            
            # Dynamic Parameter Switching based on Regime
            # Trends (Pump/Grind) -> Use Trend RSI
            # Chop (Violent/Quiet) -> Use MeanRev RSI
            # We use .isin for vectorized check
            is_trend = dataframe['regime'].isin(['pump_dump', 'grind'])
            dataframe['target_buy_rsi'] = np.where(
                is_trend,
                self.buy_rsi_trend.value,
                self.buy_rsi_mean_rev.value
            )
            
            # 3. Informative Pairs (Market Trend)
            if self.dp:
                # BTC/USDT:USDT 1h
                inf_btc = self.dp.get_pair_dataframe("BTC/USDT:USDT", "1h")
                # Fix: Use StoicLogic wrapper or ta directly
                inf_btc['ema_200'] = StoicLogic.calculate_ema(inf_btc['close'], 200)
                # Market Trend: 1 if Price > EMA200, else 0
                inf_btc['market_trend'] = np.where(inf_btc['close'] > inf_btc['ema_200'], 1, 0)
                
                dataframe = merge_informative_pair(dataframe, inf_btc, self.timeframe, "1h", ffill=True)

            # 4. ML Predictions
            dataframe = self._calculate_ml_predictions(dataframe, metadata)

        except KeyError as e:
            logger.error(f"Missing column in populate_indicators: {e}")
            # Ensure critical columns exist to prevent crashes downstream
            if 'ml_prediction' not in dataframe.columns:
                dataframe['ml_prediction'] = 0.0
                
        except Exception as e:
            logger.exception(f"Unexpected error in populate_indicators: {e}")
            if 'ml_prediction' not in dataframe.columns:
                dataframe['ml_prediction'] = 0.0
            
        return dataframe

    def _calculate_ml_predictions(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate predictions using src.ml pipeline."""
        try:
            pair = metadata['pair']
            model, fe, feature_names = get_production_model(pair)
            
            if model and fe:
                # Feature Engineering
                df_fe = dataframe.copy()
                if 'date' in df_fe.columns and not isinstance(df_fe.index, pd.DatetimeIndex):
                    df_fe.set_index('date', inplace=True)
                
                # Transform (generate features + scale)
                # Note: This calls the refactored _apply_aggressive_cleaning which imputes NaNs
                X = fe.transform(df_fe)
                
                # Predict
                if hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)[:, 1] # Probability of Class 1 (Buy/Long)
                else:
                    preds = model.predict(X)
                
                # Align predictions
                pred_series = pd.Series(preds, index=X.index)
                
                if isinstance(dataframe.index, pd.DatetimeIndex):
                    aligned_preds = pred_series.reindex(dataframe.index, fill_value=0.5)
                else:
                    if 'date' in dataframe.columns:
                        temp_df = dataframe.set_index('date')
                        aligned_preds = pred_series.reindex(temp_df.index, fill_value=0.5)
                        aligned_preds = aligned_preds.values
                    else:
                        if len(preds) < len(dataframe):
                            padding = np.full(len(dataframe) - len(preds), 0.5)
                            aligned_preds = np.concatenate([padding, preds])
                        else:
                            aligned_preds = preds[-len(dataframe):]

                dataframe['ml_prediction'] = aligned_preds
                
            else:
                dataframe['ml_prediction'] = 0.0
                
        except Exception as e:
            logger.warning(f"ML Prediction failed: {e}")
            dataframe['ml_prediction'] = 0.0
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Regime-Conditional Entry Logic using Core Logic Layer.
        """
        # Ensure enter_long exists
        if 'enter_long' not in dataframe.columns:
            dataframe['enter_long'] = 0

        # We pass the Dynamic RSI column to Core Logic
        # Core Logic needs to handle Series if we pass it, or we rely on 'target_buy_rsi' being present
        # StoicLogic.populate_entry_exit_signals takes 'mean_rev_rsi' argument. 
        # If we pass the series, it uses it for the Mean Rev logic.
        
        target_rsi = dataframe['target_buy_rsi'] if 'target_buy_rsi' in dataframe.columns else int(self.buy_rsi.value)
        
        # Delegate to Core Logic which uses Regime Matrix
        df_signals = StoicLogic.populate_entry_exit_signals(
            dataframe, 
            buy_threshold=float(self.entry_prob_threshold.value),
            sell_rsi=int(self.sell_rsi.value),
            mean_rev_rsi=target_rsi,
            persistence_window=int(self.buy_persistence.value)
        )
        
        # Apply Market Trend Filter
        if 'market_trend_1h' in dataframe.columns:
            # Only allow entries if Broad Market (BTC) is Bullish
            df_signals['enter_long'] = df_signals['enter_long'] & (dataframe['market_trend_1h'] == 1)
        
        # --- FreqAI Logic ---
        # If FreqAI columns exist, filter entries
        if '&s_target_return_mean' in dataframe.columns:
            # 1. ROI Threshold
            # Predict Return > Threshold
            roi_mask = dataframe['&s_target_return_mean'] > self.freqai_roi_threshold.value
            
            # 2. Confidence / DI Threshold
            # DI (Dissimilarity Index) measures distance from training data.
            # Low DI = High Confidence. High DI = Unknown Territory (Risk).
            # We treat 'confidence' as (1 - DI).
            # If DI column exists (enabled in config)
            if 'DI_values' in dataframe.columns:
                # Max allowed DI = 1.0 - Confidence Threshold
                # e.g., Conf=0.8 -> Max DI = 0.2
                max_di = 1.0 - self.freqai_confidence_threshold.value
                confidence_mask = dataframe['DI_values'] < max_di
            else:
                confidence_mask = True # No DI, assume confident
                
            # Combine
            # Logic: If Strategy Signal exists AND FreqAI confirms
            # OR could be FreqAI generates signal independently. 
            # "filter out low-probability entries" implies Strategy Signal + FreqAI Confirmation.
            
            freqai_mask = roi_mask & confidence_mask

            # --- AUDIT TRAIL LOGGING (Live/Dry Only) ---
            if self.config.get('runmode') in ('live', 'dry_run'):
                try:
                    # Check the latest candle for rejection
                    last_idx = dataframe.index[-1]
                    # Note: We need to ensure we access the scalar value safely
                    base_signal = bool(df_signals['enter_long'].iloc[-1])
                    
                    if base_signal:
                        # Strategy says BUY, check FreqAI
                        is_freqai_approved = bool(freqai_mask.iloc[-1])
                        
                        if not is_freqai_approved:
                            pair = metadata['pair']
                            roi_pred = float(dataframe['&s_target_return_mean'].iloc[-1]) if '&s_target_return_mean' in dataframe.columns else 0.0
                            di_score = float(dataframe['DI_values'].iloc[-1]) if 'DI_values' in dataframe.columns else 0.0
                            
                            roi_thr = float(self.freqai_roi_threshold.value)
                            conf_thr = float(self.freqai_confidence_threshold.value)
                            max_di = 1.0 - conf_thr
                            
                            logger.info(
                                f"🛑 REJECTED {pair}: Base Strategy=GO. "
                                f"FreqAI Block: ROI={roi_pred:.4f} (Min {roi_thr:.4f}), "
                                f"DI={di_score:.4f} (Max {max_di:.4f})"
                            )
                except Exception as e:
                    logger.warning(f"Audit Trail Logging Error: {e}")
            # -------------------------------------------

            df_signals['enter_long'] = df_signals['enter_long'] & freqai_mask

        dataframe['enter_long'] = df_signals['enter_long']
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Regime-Conditional Exits using Core Logic Layer."""
        # Ensure exit_long exists
        if 'exit_long' not in dataframe.columns:
            dataframe['exit_long'] = 0

        # Delegate to Core Logic
        df_signals = StoicLogic.populate_entry_exit_signals(
            dataframe, 
            buy_threshold=float(self.entry_prob_threshold.value),
            sell_rsi=int(self.sell_rsi.value),
            mean_rev_rsi=int(self.buy_rsi.value),
            persistence_window=int(self.buy_persistence.value)
        )
        
        dataframe['exit_long'] = df_signals['exit_long']
            
        return dataframe
        
    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of each bot iteration.
        Checks for Emergency Exit conditions (Flash Crash).
        """
        # Emergency Exit: Check if BTC dropped > 20% in the last hour
        try:
            if self.config.get('runmode') in ('live', 'dry_run'):
                # Get BTC/USDT 1h data
                btc_df = self.dp.get_pair_dataframe("BTC/USDT", "1h")
                if not btc_df.empty and len(btc_df) > 0:
                    last_candle = btc_df.iloc[-1]
                    # Calculate Drop from Open
                    current_price = last_candle['close']
                    open_price = last_candle['open']
                    
                    drop_pct = (current_price - open_price) / open_price
                    
                    if drop_pct < -0.20:
                        msg = f"🚨 EMERGENCY EXIT: BTC Flash Crash detected! Drop: {drop_pct:.2%}. Stopping Bot."
                        logger.critical(msg)
                        # Attempt to send notification
                        if self.dp:
                            self.dp.send_msg(msg)
                        # Stop the bot
                        raise DependencyException(msg)
                        
        except Exception as e:
            if isinstance(e, DependencyException):
                raise
            logger.warning(f"Emergency Exit Check Failed: {e}")

        super().bot_loop_start(**kwargs)

    def bot_start(self, **kwargs) -> None:
        """Initialize Risk Manager, Config, and Hybrid Connector."""
        try:
            # Check if we are in backtest/hyperopt mode
            # Safe access to self.config
            if hasattr(self, 'config') and self.config:
                runmode = self.config.get('runmode')
            else:
                runmode = None

            if runmode in ('backtest', 'hyperopt', 'plot'):
                logger.info(f"Skipping heavy initialization in {runmode} mode.")
                return

            from src.config.manager import ConfigurationManager
            ConfigurationManager.initialize()
            
            # Initialize parent classes
            super().bot_start(**kwargs)
            
            # Initialize Hybrid Connector
            # We need to know which pairs to monitor. 
            # Freqtrade doesn't give us the pairlist easily here, so we use config
            config = ConfigurationManager.get_config()
            pairs = config.pairs if hasattr(config, 'pairs') else ["BTC/USDT", "ETH/USDT"]
            
            self.initialize_hybrid_connector(pairs=pairs)
            
            logger.info("StoicRiskMixin, Configuration, and HybridConnector initialized")
        except Exception as e:
            logger.critical(f"Failed to initialize strategy components: {e}")
            raise

    @property
    def correlation_manager(self):
        """
        Lazy property to initialize CorrelationManager only when needed
        and strictly forbidden in Hyperopt/Backtest to avoid pickling errors.
        """
        # Strictly block in Hyperopt
        if self.config.get('runmode') in ('hyperopt', 'backtest', 'plot'):
            return None
            
        if not hasattr(self, '_correlation_manager'):
            from src.risk.correlation import CorrelationManager
            self._correlation_manager = CorrelationManager(max_correlation=0.85)
            
        return self._correlation_manager

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, 
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str], 
                           side: str, **kwargs) -> bool:
        """
        Override to use Risk Mixin validation, Hybrid Connector, Correlation, and Spread checks.
        """
        # 1. Check Market Safety (Real-time MFT checks)
        if not self.check_market_safety(pair, side):
            return False

        # 2. Liquidity & Spread Protection
        try:
            ticker = self.dp.ticker(pair)
            if ticker:
                # Spread Check (> 0.5% skip)
                bid = ticker.get('bid', 0)
                ask = ticker.get('ask', 0)
                if ask > 0 and bid > 0:
                    spread_pct = (ask - bid) / ask
                    if spread_pct > 0.005:
                        logger.warning(f"Blocking {pair}: Spread {spread_pct:.2%} > 0.5%")
                        return False
                
                # Volume Check (optional, e.g., < 100k daily)
                # quoteVolume is usually 24h volume in quote currency
                if ticker.get('quoteVolume', 1e9) < 100000:
                    logger.warning(f"Blocking {pair}: Low Volume")
                    return False
        except Exception as e:
            logger.warning(f"Spread check failed for {pair}: {e}")
            
        # 3. Dynamic Correlation Filter
        # Use the property accessor which handles lazy loading and runmode checks
        cm = self.correlation_manager
        
        if cm is not None:
            try:
                # Skip in backtest/hyperopt as Trade.get_trades is not supported
                # Redundant check but keeps logic clear
                if self.config.get('runmode') in ('backtest', 'hyperopt'):
                    open_trades = []
                else:
                    open_trades = Trade.get_trades([Trade.is_open.is_(True)]).all()
                
                if open_trades:
                    # DEADLOCK FIX: Only check correlation if we have significant exposure or multiple trades
                    open_positions = [{'pair': t.pair, 'open_rate': t.open_rate, 'stake_amount': t.stake_amount} for t in open_trades]
                    
                    if open_positions:
                        all_pairs_data = {}
                        
                        # Get data for new pair
                        new_pair_data, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                        
                        if new_pair_data is not None and not new_pair_data.empty:
                            # Get data for existing positions
                            for trade in open_trades:
                                df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                                if df is not None and not df.empty:
                                    all_pairs_data[trade.pair] = df
                                else:
                                    logger.warning(f"Correlation: Could not get data for open trade {trade.pair}")
                            
                            # Only proceed if we have data to compare
                            if all_pairs_data:
                                # Check correlation
                                if not cm.check_entry_correlation(
                                    pair, new_pair_data, open_positions, all_pairs_data
                                ):
                                    return False
                        else:
                            logger.warning(f"Correlation: Could not get data for new pair {pair}. Skipping check.")

            except Exception as e:
                logger.warning(f"Correlation check failed: {e}")

        # --- Custom Telegram Notification (Market Regime & Confidence) ---
        try:
            # We are about to confirm entry. Let's fetch the info.
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                last_row = dataframe.iloc[-1]
                
                regime = last_row.get('regime', 'Unknown')
                hurst = last_row.get('hurst', 0.5)
                confidence = 1.0 - last_row.get('DI_values', 0.5) if 'DI_values' in dataframe.columns else 0.5
                
                msg = (
                    f"🚀 <b>OPEN LONG: {pair}</b>\n"
                    f"📊 Regime: {regime} (Hurst: {hurst:.2f})\n"
                    f"🤖 Model Confidence: {confidence:.2f}\n"
                    f"📈 Trend Strength: {last_row.get('adx', 0):.0f}"
                )
                # Send custom message
                if self.dp:
                    self.dp.send_msg(msg)
        except Exception as e:
            logger.warning(f"Custom Notification failed: {e}")


        # 4. Equity Curve Protection
        # Check if we are in deep drawdown -> Block new entries or strict filtering
        # We calculate drawdown of the last 10 CLOSED trades
        if self.config.get('runmode') not in ('backtest', 'hyperopt'):
            try:
                closed_trades = Trade.get_trades([Trade.is_open.is_(False)]).order_by(Trade.close_date.desc()).limit(10).all()
                if closed_trades:
                    # Simple "Drawdown" proxy: Sum of profits of last 10 trades
                    recent_pnl = sum([t.close_profit_abs for t in closed_trades])
                    current_balance = self.wallets.get_total_stake_amount()
                    
                    # If recent losses > max_equity_drawdown * balance
                    if recent_pnl < -(float(self.max_equity_drawdown.value) * current_balance):
                        logger.warning(f"Blocking entry due to recent drawdown: {recent_pnl:.2f}")
                        return False
            except Exception as e:
                logger.warning(f"Equity check failed: {e}")

        # 5. Check Risk Limits (Portfolio Risk)
        return super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, 
                                         current_time, entry_tag, side, **kwargs)

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], 
                           max_stake: float, leverage: float, entry_tag: Optional[str],
                           side: str, **kwargs) -> float:
        """
        Dynamic Position Sizing (Risk Parity / Inverse Volatility).
        Overrides Mixin to use Strategy Parameters.
        """
        try:
            # 1. Get Equity
            current_balance = self.wallets.get_total_stake_amount()
            
            # 2. Get Volatility (ATR)
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            
            atr_pct = last_candle.get('atr', current_rate * 0.02) / current_rate
            if pd.isna(atr_pct) or atr_pct == 0:
                atr_pct = 0.02

            # 3. Calculate Size
            # Risk Amount = Balance * risk_per_trade (e.g., 1000 * 0.01 = 10 USD)
            # Stop Distance % = ATR * StopMultiplier (e.g., 2% * 2.5 = 5%)
            # Position Size = Risk Amount / Stop Distance % = 10 / 0.05 = 200 USD
            
            risk_amt = current_balance * float(self.risk_per_trade.value)
            stop_mult = float(self.stoploss_atr_mult.value)
            stop_dist_pct = atr_pct * stop_mult
            
            if stop_dist_pct == 0:
                return proposed_stake
                
            safe_stake = risk_amt / stop_dist_pct
            
            # 4. Limits
            # Max 20% of account
            max_cap = current_balance * 0.20
            safe_stake = min(safe_stake, max_cap)
            
            # Freqtrade Limits
            if min_stake:
                safe_stake = max(safe_stake, min_stake)
            if max_stake:
                safe_stake = min(safe_stake, max_stake)
                
            logger.info(f"{pair} Sizing: Bal={current_balance:.0f} Risk={risk_amt:.1f} ATR={atr_pct:.1%} -> Stake={safe_stake:.1f}")
            
            return safe_stake
            
        except Exception as e:
            logger.error(f"Error in custom_stake_amount: {e}")
            return proposed_stake

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[str]:
        """
        Advanced Exits: Time Decay & Volatility Trailing.
        """
        # 1. Time-Based Decay
        # If trade is open for > X candles and Profit < Y% -> Exit
        current_candles = (current_time - trade.open_date_utc).total_seconds() / (5*60) # Assuming 5m candles
        if (current_candles > self.exit_profit_time_limit_candles.value) and (current_profit < self.exit_profit_time_limit_roi.value):
            return "time_decay_exit"
            
        # 2. Volatility Trailing Stop
        # Instead of fixed trailing, we check if price dropped by ATR * Mult from High
        # Freqtrade doesn't easily give "High since entry" in this callback without lookup
        # But we can approximate or leave this to `custom_stoploss` which updates the stop price.
        # Actually, `custom_stoploss` is better for trailing adjustment.
        # `custom_exit` is for immediate market exit.
        
        return super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Volatility-Adjusted Stop Loss using Z-Score from Regime AND Equity Curve Protection.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Base ATR stop
        atr = last_candle.get('atr', current_rate * 0.02)
        
        # Use hyperoptable multiplier
        base_mult = float(self.stoploss_atr_mult.value)
        
        # Adjust multiplier based on Regime Z-Score
        # If High Vol (Z > 1), widen stop to avoid noise
        vol_z = last_candle.get('vol_zscore', 0)
        z_mult_factor = float(self.stoploss_zscore_mult.value)
        
        if vol_z > 1.0:
            # Widen stop in high volatility
            base_mult *= z_mult_factor
        elif vol_z < -1.0:
            # Tighten stop in very low volatility (optional, or keep base)
            base_mult *= 0.8

        # --- Equity Curve Protection ---
        # If recent performance is bad, TIGHTEN STOPLOSS
        if self.config.get('runmode') not in ('backtest', 'hyperopt'):
            try:
                closed_trades = Trade.get_trades([Trade.is_open.is_(False)]).order_by(Trade.close_date.desc()).limit(10).all()
                if closed_trades:
                    recent_pnl = sum([t.close_profit_abs for t in closed_trades])
                    current_balance = self.wallets.get_total_stake_amount()
                    # If Drawdown > 50% of max allowed, start tightening
                    if recent_pnl < -(float(self.max_equity_drawdown.value) * 0.5 * current_balance):
                        base_mult *= 0.75 # Tighten by 25%
            except Exception:
                pass
            
        stop_dist = atr * base_mult
        stop_pct = stop_dist / current_rate
        
        return -stop_pct

    # --- FreqAI Integration (Level 3) ---

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs):
        """
        FreqAI: Feature Engineering (Lightweight XGBoost features).
        """
        # 1. Volatility Ratios (ATR Short / ATR Long)
        dataframe['atr_14'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_100'] = ta.ATR(dataframe, timeperiod=100)
        dataframe['vol_ratio'] = dataframe['atr_14'] / dataframe['atr_100']
        
        # 2. Distance from EMA normalized by ATR
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['dist_ema_atr'] = (dataframe['close'] - dataframe['ema_20']) / dataframe['atr_14']
        
        # 3. RSI for context
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs):
        """
        Add features to basic dataframe (without period expansion).
        """
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs):
        """
        Standard feature engineering.
        """
        # Ensure 'date' is set
        if 'date' in dataframe.columns:
            dataframe['date'] = pd.to_datetime(dataframe['date'])
            dataframe = dataframe.set_index('date')
            
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs):
        """
        Set FreqAI targets: Predict 'Target Price Change' next 20 candles.
        """
        # Label: (Close[t+20] - Close[t]) / Close[t]
        dataframe['&s_target_return'] = (dataframe['close'].shift(-20) - dataframe['close']) / dataframe['close']
        return dataframe
