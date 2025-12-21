#!/usr/bin/env python3
"""
Walk-Forward Analysis for ML Trading Strategy
=============================================

Implements the user's requirements:
1. Load 1 year of data (or available data)
2. Split data into sliding windows (Train: 3 months, Test: 1 month)
3. Loop through windows:
   - Train XGBoost/LightGBM on TRAIN chunk
   - Save model temporarily
   - Run Backtest (simulate trades) on TEST chunk using this model
   - Record metrics (PnL, Sharpe, Win Rate)
4. Aggregate results: Show total cumulative PnL over all TEST chunks
5. Visualization: Save a plot `wfo_results.png` showing equity curve

This version addresses issues with the existing implementation:
- Works with limited data (adjusts window sizes automatically)
- Handles class imbalance with appropriate techniques
- Uses simpler feature engineering to avoid validation issues
- Implements proper walk-forward validation
"""

import argparse
import logging
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Try to import ML libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class WalkForwardAnalysis:
    """
    Walk-Forward Analysis for ML trading strategies.
    
    Implements sliding window validation with ML models.
    """
    
    def __init__(
        self,
        data_path: str = "user_data/data/binance",
        models_dir: str = "user_data/models/walk_forward",
        results_dir: str = "user_data/walk_forward_results"
    ):
        """
        Initialize walk-forward analysis.
        
        Args:
            data_path: Path to data directory
            models_dir: Directory to save trained models
            results_dir: Directory to save results
        """
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.window_results = []
        self.cumulative_pnl = []
        self.equity_curve = []
        self.all_predictions = []
        
        # Feature scaler
        self.scaler = StandardScaler()
        
    def load_data(self, pair: str, timeframe: str = "5m") -> pd.DataFrame:
        """
        Load OHLCV data for specified pair and timeframe.
        
        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "5m", "1h")
            
        Returns:
            DataFrame with OHLCV data
        """
        pair_filename = pair.replace("/", "_")
        
        # Try different file formats
        possible_paths = [
            self.data_path / f"{pair_filename}-{timeframe}.feather",
            self.data_path / f"{pair_filename}-{timeframe}.parquet",
            self.data_path / f"{pair_filename}-{timeframe}.csv",
        ]
        
        df = None
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading data from {path}")
                if path.suffix == ".feather":
                    df = pd.read_feather(path)
                elif path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                break
        
        if df is None:
            raise FileNotFoundError(f"No data found for {pair} {timeframe}")
        
        # Ensure datetime index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ensure required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Sort by time
        df = df.sort_index()
        
        logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        logger.info(f"Total days: {(df.index[-1] - df.index[0]).days}")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML model.
        
        Simple feature engineering to avoid validation issues.
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        features['volume_change'] = df['volume'].pct_change()
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        # Moving averages
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['ema_12'] = df['close'].ewm(span=12).mean()
        features['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Price vs moving averages
        features['price_vs_sma20'] = df['close'] / features['sma_20'] - 1
        features['price_vs_sma50'] = df['close'] / features['sma_50'] - 1
        features['macd'] = features['ema_12'] - features['ema_26']
        
        # Volatility
        features['volatility_20'] = df['close'].rolling(20).std()
        features['atr'] = self._calculate_atr(df)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def create_labels(self, df: pd.DataFrame, 
                     take_profit: float = 0.008,  # 0.8%
                     stop_loss: float = 0.004,    # 0.4%
                     max_periods: int = 48) -> pd.Series:
        """
        Create binary labels using Triple Barrier method.
        
        Label = 1 if price hits take profit before stop loss within max_periods
        Label = 0 otherwise
        """
        labels = pd.Series(0, index=df.index, dtype=int)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        for i in range(len(df) - max_periods):
            entry_price = close[i]
            upper_barrier = entry_price * (1 + take_profit)
            lower_barrier = entry_price * (1 - stop_loss)
            
            # Check forward periods
            for j in range(1, max_periods + 1):
                if i + j >= len(df):
                    break
                
                # Check if both barriers hit in same candle
                upper_hit = high[i + j] >= upper_barrier
                lower_hit = low[i + j] <= lower_barrier
                
                if upper_hit and lower_hit:
                    # Ambiguous - use close price
                    if close[i + j] >= entry_price:
                        labels.iloc[i] = 1
                    break
                
                if upper_hit:
                    labels.iloc[i] = 1
                    break
                    
                if lower_hit:
                    labels.iloc[i] = 0
                    break
        
        # Fill remaining with NaN
        labels.iloc[-max_periods:] = np.nan
        
        logger.info(f"Positive labels: {labels.sum()} out of {labels.count()} ({labels.mean():.2%})")
        
        return labels
    
    def create_windows(
        self, 
        df: pd.DataFrame, 
        train_days: int = 90,  # 3 months
        test_days: int = 30,   # 1 month
        step_days: int = 30    # 1 month step
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create sliding windows for walk-forward validation.
        
        Adjusts window sizes based on available data.
        """
        windows = []
        
        # Calculate approximate candles per day (for 5m data)
        if len(df) > 0:
            time_diff = df.index[-1] - df.index[0]
            days_total = time_diff.days + time_diff.seconds / (24 * 3600)
            candles_per_day = len(df) / max(1, days_total)
        else:
            candles_per_day = 288  # Default for 5m data
        
        train_candles = int(train_days * candles_per_day)
        test_candles = int(test_days * candles_per_day)
        step_candles = int(step_days * candles_per_day)
        
        n = len(df)
        
        # Adjust if not enough data
        if train_candles + test_candles > n:
            # Reduce window sizes proportionally
            scale_factor = n / (train_candles + test_candles)
            train_candles = int(train_candles * scale_factor * 0.7)
            test_candles = int(test_candles * scale_factor * 0.3)
            step_candles = min(train_candles, step_candles)
            logger.warning(f"Adjusting window sizes due to limited data: train={train_candles}, test={test_candles}")
        
        start_idx = 0
        
        while start_idx + train_candles + test_candles <= n:
            train_end = start_idx + train_candles
            test_end = train_end + test_candles
            
            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            
            windows.append((train_df, test_df))
            
            # Slide window
            start_idx += step_candles
        
        logger.info(f"Created {len(windows)} sliding windows")
        logger.info(f"Train size: ~{train_days} days ({train_candles} candles)")
        logger.info(f"Test size: ~{test_days} days ({test_candles} candles)")
        
        return windows
    
    def train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        model_type: str = "xgboost",
        use_class_weight: bool = True
    ) -> Any:
        """
        Train ML model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: "xgboost" or "lightgbm"
            use_class_weight: Whether to use class weights for imbalance
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        logger.info(f"Class distribution: {y_train.value_counts().to_dict()}")
        
        # Calculate class weights if needed
        class_weight = None
        if use_class_weight and len(y_train.unique()) > 1:
            class_counts = y_train.value_counts()
            total = len(y_train)
            class_weight = {
                0: total / (2 * class_counts.get(0, 1)),
                1: total / (2 * class_counts.get(1, 1))
            }
            logger.info(f"Class weights: {class_weight}")
        
        if model_type == "xgboost" and XGB_AVAILABLE:
            # XGBoost model
            scale_pos_weight = class_weight[1] / class_weight[0] if class_weight else 1
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif model_type == "lightgbm" and LGB_AVAILABLE:
            # LightGBM model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight=class_weight if class_weight else 'balanced'
            )
        else:
            # Fallback to Random Forest
            from sklearn.ensemble import RandomForestClassifier
            logger.warning(f"{model_type} not available, using Random Forest")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight=class_weight if class_weight else 'balanced',
                n_jobs=-1
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        return model
    
    def simulate_trading(
        self,
        test_df: pd.DataFrame,
        features: pd.DataFrame,
        model: Any,
        probability_threshold: float = 0.55
    ) -> Dict[str, Any]:
        """
        Simulate trading on test data.
        
        Args:
            test_df: Test window OHLCV data
            features: Features for test data
            model: Trained model
            probability_threshold: Threshold for trade signals
            
        Returns:
            Dictionary with trading metrics
        """
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(features)[:, 1]
            trade_signals = (y_pred_proba > probability_threshold).astype(int)
        else:
            y_pred = model.predict(features)
            trade_signals = y_pred
        
        # Calculate returns
        returns = test_df['close'].pct_change().shift(-1)  # Next period return
        
        # Align indices
        common_idx = returns.index.intersection(features.index)
        if len(common_idx) == 0:
            return {
                "total_pnl": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "n_trades": 0,
                "avg_trade_return": 0.0,
                "cumulative_returns": []
            }
        
        returns = returns.loc[common_idx]
        trade_signals = pd.Series(trade_signals, index=features.index).loc[common_idx]
        
        # Strategy returns: long when signal=1
        strategy_returns = returns * trade_signals
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {
                "total_pnl": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "n_trades": 0,
                "avg_trade_return": 0.0,
                "cumulative_returns": []
            }
        
        # Calculate metrics
        n_trades = trade_signals.sum()
        
        # Winning and losing trades
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        
        win_rate = len(winning_trades) / max(1, len(winning_trades) + len(losing_trades))
        
        # Profit factor
        gross_profit = winning_trades.sum()
        gross_loss = abs(losing_trades.sum())
        profit_factor = gross_profit / max(0.0001, gross_loss)
        
        # Total PnL and cumulative returns
        total_pnl = (1 + strategy_returns).prod() - 1
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Sharpe ratio (annualized)
        if strategy_returns.std() > 0:
            # Assuming 5m data: 288 candles per day, 252 trading days
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252 * 288)
        else:
            sharpe = 0.0
        
        # Max drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # Average trade return
        avg_trade_return = strategy_returns.mean()
        
        return {
            "total_pnl": total_pnl,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "avg_trade_return": avg_trade_return,
            "cumulative_returns": cumulative_returns.tolist()
        }
    
    def process_window(
        self,
        window_id: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_type: str = "xgboost",
        probability_threshold: float = 0.55
    ) -> Dict[str, Any]:
        """
        Process a single window: train model and test on out-of-sample data.
        
        Args:
            window_id: Window identifier
            train_df: Training data
            test_df: Testing data
            model_type: Model type
            probability_threshold: Threshold for trade signals
            
        Returns:
            Dictionary with window results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Window {window_id + 1}")
        logger.info(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} candles)")
        logger.info(f"Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} candles)")
        
        try:
            # 1. Create features and labels
            logger.info("Creating features and labels...")
            train_features = self.create_features(train_df)
            test_features = self.create_features(test_df)
            
            train_labels = self.create_labels(train_df)
            test_labels = self.create_labels(test_df)
            
            # Align features and labels (remove NaN labels)
            train_mask = train_labels.notna()
            test_mask = test_labels.notna()
            
            X_train = train_features[train_mask]
            y_train = train_labels[train_mask]
            
            X_test = test_features[test_mask]
            y_test = test_labels[test_mask]
            
            if len(X_train) < 100 or len(X_test) < 20:
                logger.warning(f"Window {window_id}: Insufficient data after cleaning")
                return {
                    "window_id": window_id,
                    "success": False,
                    "error": "Insufficient data"
                }
            
            # 2. Scale features
            logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 3. Train model
            model = self.train_model(
                X_train_scaled, 
                y_train, 
                model_type=model_type,
                use_class_weight=True
            )
            
            # 4. Evaluate model
            logger.info("Evaluating model...")
            y_pred = model.predict(X_test_scaled)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = None
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # 5. Simulate trading
            logger.info("Simulating trading...")
            trading_results = self.simulate_trading(
                test_df.loc[X_test.index],
                pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns),
                model,
                probability_threshold
            )
            
            # 6. Save model
            model_path = self.models_dir / f"window_{window_id}_{model_type}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            # 7. Compile results
            result = {
                "window_id": window_id,
                "success": True,
                "train_start": train_df.index[0].isoformat(),
                "train_end": train_df.index[-1].isoformat(),
                "test_start": test_df.index[0].isoformat(),
                "test_end": test_df.index[-1].isoformat(),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "model_accuracy": accuracy,
                "model_f1": f1,
                "model_path": str(model_path),
                **trading_results
            }
            
            logger.info(f"Window {window_id} results:")
            logger.info(f"  Test Accuracy: {accuracy:.2%}")
            logger.info(f"  Test F1: {f1:.3f}")
            logger.info(f"  Test PnL: {trading_results['total_pnl']:.2%}")
            logger.info(f"  Sharpe: {trading_results['sharpe']:.2f}")
            logger.info(f"  Win Rate: {trading_results['win_rate']:.2%}")
            logger.info(f"  Profit Factor: {trading_results['profit_factor']:.2f}")
            logger.info(f"  Trades: {trading_results['n_trades']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Window {window_id} failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "window_id": window_id,
                "success": False,
                "error": str(e)
            }
    
    def run(
        self,
        pair: str,
        timeframe: str = "5m",
        train_days: int = 90,
        test_days: int = 30,
        step_days: int = 30,
        model_type: str = "xgboost",
        probability_threshold: float = 0.55
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward analysis.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe
            train_days: Training window size in days
            test_days: Testing window size in days
            step_days: Step size in days
            model_type: Model type ("xgboost" or "lightgbm")
            probability_threshold: Threshold for trade signals
            
        Returns:
            Dictionary with aggregated results
        """
        logger.info(f"\n{'='*70}")
        logger.info("WALK-FORWARD ANALYSIS")
        logger.info(f"{'='*70}")
        logger.info(f"Pair: {pair}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Train window: {train_days} days (~{train_days/30:.1f} months)")
        logger.info(f"Test window: {test_days} days (~{test_days/30:.1f} months)")
        logger.info(f"Step size: {step_days} days")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Probability threshold: {probability_threshold}")
        logger.info(f"{'='*70}")
        
        # 1. Load data
        logger.info("Loading data...")
        df = self.load_data(pair, timeframe)
        
        # 2. Create windows
        logger.info("Creating sliding windows...")
        windows = self.create_windows(df, train_days, test_days, step_days)
        
        if not windows:
            raise ValueError("No windows created - insufficient data")
        
        # 3. Process each window
        self.window_results = []
        self.cumulative_pnl = []
        self.equity_curve = []
        self.all_predictions = []
        
        cumulative_equity = 1.0
        
        for window_id, (train_df, test_df) in enumerate(windows):
            result = self.process_window(
                window_id, train_df, test_df, model_type, probability_threshold
            )
            self.window_results.append(result)
            
            if result.get("success", False):
                # Update cumulative PnL
                window_pnl = result.get("total_pnl", 0)
                cumulative_equity *= (1 + window_pnl)
                self.cumulative_pnl.append(cumulative_equity - 1)
                self.equity_curve.append(cumulative_equity)
                
                logger.info(f"Window {window_id + 1} cumulative PnL: {self.cumulative_pnl[-1]:.2%}")
        
        # 4. Aggregate results
        aggregated = self._aggregate_results()
        
        # 5. Generate visualization
        self._generate_visualization(aggregated)
        
        # 6. Save results
        self._save_results(aggregated)
        
        logger.info(f"\n{'='*70}")
        logger.info("WALK-FORWARD ANALYSIS COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total cumulative PnL: {aggregated['total_cumulative_pnl']:.2%}")
        logger.info(f"Average window PnL: {aggregated['avg_window_pnl']:.2%}")
        logger.info(f"Average Sharpe: {aggregated['avg_sharpe']:.2f}")
        logger.info(f"Average Profit Factor: {aggregated['avg_profit_factor']:.2f}")
        logger.info(f"Win Rate: {aggregated['avg_win_rate']:.2%}")
        logger.info(f"Successful Windows: {aggregated['n_successful']}/{aggregated['n_windows']}")
        
        return aggregated
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all windows."""
        successful_results = [r for r in self.window_results if r.get("success", False)]
        
        if not successful_results:
            return {
                "total_cumulative_pnl": 0.0,
                "avg_window_pnl": 0.0,
                "avg_sharpe": 0.0,
                "avg_profit_factor": 0.0,
                "avg_win_rate": 0.0,
                "n_windows": len(self.window_results),
                "n_successful": 0,
                "window_results": self.window_results
            }
        
        # Calculate aggregated metrics
        total_cumulative_pnl = self.cumulative_pnl[-1] if self.cumulative_pnl else 0.0
        avg_window_pnl = np.mean([r.get("total_pnl", 0) for r in successful_results])
        avg_sharpe = np.mean([r.get("sharpe", 0) for r in successful_results])
        avg_profit_factor = np.mean([r.get("profit_factor", 0) for r in successful_results])
        avg_win_rate = np.mean([r.get("win_rate", 0) for r in successful_results])
        
        return {
            "total_cumulative_pnl": total_cumulative_pnl,
            "avg_window_pnl": avg_window_pnl,
            "avg_sharpe": avg_sharpe,
            "avg_profit_factor": avg_profit_factor,
            "avg_win_rate": avg_win_rate,
            "n_windows": len(self.window_results),
            "n_successful": len(successful_results),
            "window_results": self.window_results,
            "equity_curve": self.equity_curve
        }
    
    def _generate_visualization(self, results: Dict[str, Any]) -> None:
        """Generate and save visualization of results."""
        if not self.equity_curve:
            logger.warning("No equity curve data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity curve
        ax1 = axes[0, 0]
        ax1.plot(self.equity_curve, label="Equity Curve", linewidth=2, color='blue')
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label="Starting Equity")
        ax1.set_title("Equity Curve (Cumulative PnL)")
        ax1.set_xlabel("Window")
        ax1.set_ylabel("Equity (Multiple)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Window PnL distribution
        ax2 = axes[0, 1]
        window_pnls = [r.get("total_pnl", 0) for r in self.window_results if r.get("success", False)]
        if window_pnls:
            ax2.hist(window_pnls, bins=20, edgecolor='black', alpha=0.7, color='green')
            ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_title("Window PnL Distribution")
            ax2.set_xlabel("PnL per Window")
            ax2.set_ylabel("Frequency")
            ax2.grid(True, alpha=0.3)
        
        # 3. Sharpe ratio by window
        ax3 = axes[1, 0]
        sharpe_ratios = [r.get("sharpe", 0) for r in self.window_results if r.get("success", False)]
        if sharpe_ratios:
            window_ids = [i for i, r in enumerate(self.window_results) if r.get("success", False)]
            ax3.bar(window_ids, sharpe_ratios, alpha=0.7, color='orange')
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax3.set_title("Sharpe Ratio by Window")
            ax3.set_xlabel("Window ID")
            ax3.set_ylabel("Sharpe Ratio")
            ax3.grid(True, alpha=0.3)
        
        # 4. Profit factor by window
        ax4 = axes[1, 1]
        profit_factors = [r.get("profit_factor", 0) for r in self.window_results if r.get("success", False)]
        if profit_factors:
            window_ids = [i for i, r in enumerate(self.window_results) if r.get("success", False)]
            ax4.bar(window_ids, profit_factors, alpha=0.7, color='purple')
            ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label="Break-even")
            ax4.set_title("Profit Factor by Window")
            ax4.set_xlabel("Window ID")
            ax4.set_ylabel("Profit Factor")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f"wfo_results_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_path}")
        
        # Also save as wfo_results.png (latest)
        latest_path = self.results_dir / "wfo_results.png"
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Latest visualization saved to {latest_path}")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"wfo_results_{timestamp}.json"
        
        # Convert non-serializable objects
        serializable_results = results.copy()
        serializable_results["window_results"] = [
            {k: v for k, v in r.items() if not isinstance(v, (np.ndarray, np.generic))}
            for r in results["window_results"]
        ]
        
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Also save as wfo_results.json (latest)
        latest_file = self.results_dir / "wfo_results.json"
        with open(latest_file, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
        logger.info(f"Latest results saved to {latest_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Analysis for ML Trading Strategies"
    )
    parser.add_argument(
        "--pair",
        default="BTC/USDT",
        help="Trading pair (e.g., BTC/USDT, ETH/USDT)"
    )
    parser.add_argument(
        "--timeframe",
        default="5m",
        help="Timeframe (e.g., 5m, 15m, 1h, 4h)"
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=90,
        help="Training window size in days (default: 90 = 3 months)"
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Testing window size in days (default: 30 = 1 month)"
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=30,
        help="Step size in days (default: 30 = 1 month)"
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "lightgbm", "randomforest"],
        help="ML model type (default: xgboost)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Probability threshold for trade signals (default: 0.55)"
    )
    parser.add_argument(
        "--output-dir",
        default="user_data/walk_forward_results",
        help="Directory to save results (default: user_data/walk_forward_results)"
    )
    
    args = parser.parse_args()
    
    # Check if required ML libraries are available
    if args.model == "xgboost" and not XGB_AVAILABLE:
        logger.warning("XGBoost not available. Install with: pip install xgboost")
        logger.warning("Falling back to Random Forest")
        args.model = "randomforest"
    
    if args.model == "lightgbm" and not LGB_AVAILABLE:
        logger.warning("LightGBM not available. Install with: pip install lightgbm")
        logger.warning("Falling back to Random Forest")
        args.model = "randomforest"
    
    # Run walk-forward analysis
    try:
        wfa = WalkForwardAnalysis(results_dir=args.output_dir)
        
        results = wfa.run(
            pair=args.pair,
            timeframe=args.timeframe,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            model_type=args.model,
            probability_threshold=args.threshold
        )
        
        # Print summary
        print("\n" + "="*70)
        print("WALK-FORWARD ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total Cumulative PnL: {results['total_cumulative_pnl']:.2%}")
        print(f"Average Window PnL: {results['avg_window_pnl']:.2%}")
        print(f"Average Sharpe Ratio: {results['avg_sharpe']:.2f}")
        print(f"Average Profit Factor: {results['avg_profit_factor']:.2f}")
        print(f"Average Win Rate: {results['avg_win_rate']:.2%}")
        print(f"Successful Windows: {results['n_successful']}/{results['n_windows']}")
        print("="*70)
        
        # Performance assessment
        if results['n_successful'] > 0:
            if results['avg_profit_factor'] > 1.5:
                print("✅ EXCELLENT: Profit Factor > 1.5")
            elif results['avg_profit_factor'] > 1.2:
                print("👍 GOOD: Profit Factor > 1.2")
            elif results['avg_profit_factor'] > 1.0:
                print("⚠️  MARGINAL: Profit Factor > 1.0 but <= 1.2")
            else:
                print("❌ POOR: Profit Factor <= 1.0")
            
            if results['avg_sharpe'] > 1.0:
                print("✅ EXCELLENT: Sharpe Ratio > 1.0")
            elif results['avg_sharpe'] > 0.5:
                print("👍 GOOD: Sharpe Ratio > 0.5")
            elif results['avg_sharpe'] > 0.0:
                print("⚠️  MARGINAL: Sharpe Ratio > 0.0 but <= 0.5")
            else:
                print("❌ POOR: Sharpe Ratio <= 0.0")
        else:
            print("❌ NO SUCCESSFUL WINDOWS: Analysis failed for all windows")
        
        print("\nResults saved to:")
        print(f"  - {args.output_dir}/wfo_results.json")
        print(f"  - {args.output_dir}/wfo_results.png")
        print(f"  - {args.output_dir}/wfo_results_*.json (timestamped)")
        print(f"  - {args.output_dir}/wfo_results_*.png (timestamped)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
