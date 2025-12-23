#!/usr/bin/env python3
"""
Walk-Forward Backtest Engine
=============================

Professional walk-forward validation pipeline for ML trading strategies.
Implements sliding window validation to prevent overfitting and provide
realistic performance estimates.

Key Features:
- Triple Barrier labeling with user-defined parameters
- XGBoost/LightGBM model training
- Out-of-sample backtesting on each test window
- Cumulative PnL tracking across all windows
- Visualization of equity curve

Usage:
    python scripts/walk_forward_backtest.py --pair BTC/USDT --timeframe 5m

    # Custom parameters
    python scripts/walk_forward_backtest.py --pair ETH/USDT --train-months 3 --test-months 1 --model xgboost
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.training.feature_engineering import FeatureEngineer, FeatureConfig
from src.ml.training.labeling import TripleBarrierLabeler, TripleBarrierConfig
from src.ml.training.model_trainer import ModelTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-Forward Backtest Engine for ML trading strategies.
    
    This implements the gold standard for strategy validation:
    1. Split data into sliding windows (train + test)
    2. Train model on train window
    3. Test on subsequent (unseen) test window
    4. Slide window forward and repeat
    5. Aggregate out-of-sample results
    
    This prevents look-ahead bias and provides realistic performance estimates.
    """
    
    def __init__(
        self,
        data_path: str = "user_data/data/binance",
        models_dir: str = "user_data/models/walk_forward",
        results_dir: str = "user_data/walk_forward_results"
    ):
        """
        Initialize walk-forward backtest engine.
        
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
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(FeatureConfig(
            scale_features=True,
            scaling_method="standard",
            include_time_features=True
        ))
        
        # Triple Barrier configuration (matches user requirements)
        self.labeler_config = TripleBarrierConfig(
            take_profit=0.015,  # 1.5%
            stop_loss=0.0075,   # 0.75%
            max_holding_period=24,  # 24 candles
            fee_adjustment=0.001,  # 0.1% fees
            include_hold_class=False  # Binary classification
        )
        self.labeler = TripleBarrierLabeler(self.labeler_config)
        
        # Results storage
        self.window_results = []
        self.cumulative_pnl = []
        self.equity_curve = []
        
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
            self.data_path / f"{pair_filename}-{timeframe}.json",
            self.data_path / f"{pair_filename}-{timeframe}.csv",
        ]
        
        df = None
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading data from {path}")
                if path.suffix == ".feather":
                    df = pd.read_feather(path)
                elif path.suffix == ".json":
                    with open(path, "r") as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                else:
                    df = pd.read_csv(path)
                break
        
        if df is None:
            raise FileNotFoundError(f"No data found for {pair} {timeframe}")
        
        # Ensure datetime index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
        elif "date" in df.columns:
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
        return df
    
    def create_windows(
        self, 
        df: pd.DataFrame, 
        train_months: int = 3,
        test_months: int = 1,
        step_months: int = 1,
        embargo_days: int = 7
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create sliding windows for purged walk-forward validation (De Prado Methodology).
        
        Implements embargo period to prevent data leakage from overlapping Triple Barrier events.
        
        Args:
            df: Full dataset
            train_months: Number of months for training
            test_months: Number of months for testing
            step_months: Step size to slide window (months)
            embargo_days: Number of days to leave as gap between train and test (default 7 days)
            
        Returns:
            List of (train_df, test_df) tuples with embargo period
        """
        windows = []
        
        # Convert months to approximate number of candles
        # Assuming 30 days per month, 288 candles per day for 5m timeframe
        if "5m" in str(self.data_path):
            candles_per_day = 288
        elif "1h" in str(self.data_path):
            candles_per_day = 24
        else:
            candles_per_day = 24  # Default
        
        train_candles = int(train_months * 30 * candles_per_day)
        test_candles = int(test_months * 30 * candles_per_day)
        step_candles = int(step_months * 30 * candles_per_day)
        embargo_candles = int(embargo_days * candles_per_day)
        
        n = len(df)
        start_idx = 0
        
        while start_idx + train_candles + embargo_candles + test_candles <= n:
            train_end = start_idx + train_candles
            test_start = train_end + embargo_candles  # Add embargo gap
            test_end = test_start + test_candles
            
            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            windows.append((train_df, test_df))
            
            # Slide window
            start_idx += step_candles
        
        logger.info(f"Created {len(windows)} sliding windows with {embargo_days}-day embargo")
        logger.info(f"Train size: {train_candles} candles (~{train_months} months)")
        logger.info(f"Test size: {test_candles} candles (~{test_months} months)")
        logger.info(f"Embargo: {embargo_candles} candles (~{embargo_days} days)")
        
        return windows
    
    def create_combinatorial_purged_windows(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        embargo_days: int = 7
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create Combinatorial Purged Cross-Validation (CPCV) windows (De Prado Methodology).
        
        CPCV creates multiple train/test splits with embargo periods to prevent
        data leakage from overlapping Triple Barrier events.
        
        Args:
            df: Full dataset
            n_splits: Number of splits for CPCV
            embargo_days: Number of days to leave as gap between train and test
            
        Returns:
            List of (train_df, test_df) tuples
        """
        windows = []
        
        # Convert embargo days to candles
        if "5m" in str(self.data_path):
            candles_per_day = 288
        elif "1h" in str(self.data_path):
            candles_per_day = 24
        else:
            candles_per_day = 24
        
        embargo_candles = int(embargo_days * candles_per_day)
        
        # Calculate split sizes
        n = len(df)
        split_size = n // (n_splits + 1)  # +1 for embargo periods
        
        for i in range(n_splits):
            # Test window
            test_start = i * split_size
            test_end = (i + 1) * split_size
            
            # Train window: everything before test_start minus embargo
            train_end = test_start - embargo_candles
            if train_end <= 0:
                continue  # Skip if not enough data for training
                
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            windows.append((train_df, test_df))
        
        logger.info(f"Created {len(windows)} CPCV windows with {embargo_days}-day embargo")
        
        return windows
    
    def train_and_test_window(
        self,
        window_id: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_type: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Train model on train window and test on test window.
        
        Args:
            window_id: Window identifier
            train_df: Training data
            test_df: Testing data
            model_type: Model type ("xgboost" or "lightgbm")
            
        Returns:
            Dictionary with window results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Window {window_id + 1}")
        logger.info(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} candles)")
        logger.info(f"Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} candles)")
        
        try:
            # 1. Label data
            logger.info("Labeling data with Triple Barrier...")
            train_labels = self.labeler.label(train_df)
            test_labels = self.labeler.label(test_df)
            
            # Convert labels from -1/1 to 0/1 for binary classification
            # -1 -> 0 (negative class), 1 -> 1 (positive class)
            train_labels = train_labels.replace({-1: 0, 1: 1})
            test_labels = test_labels.replace({-1: 0, 1: 1})
            
            # 2. Feature engineering
            logger.info("Engineering features...")
            train_features = self.feature_engineer.fit_transform(train_df)
            test_features = self.feature_engineer.transform(test_df)
            
            # Align features and labels
            feature_cols = self.feature_engineer.get_feature_names()
            
            X_train = train_features[feature_cols].dropna()
            y_train = train_labels.loc[X_train.index]
            
            X_test = test_features[feature_cols].dropna()
            y_test = test_labels.loc[X_test.index]
            
            # Remove NaN labels
            train_mask = y_train.notna()
            test_mask = y_test.notna()
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
            
            if len(X_train) < 100 or len(X_test) < 20:
                logger.warning(f"Window {window_id}: Insufficient data after cleaning")
                return {
                    "window_id": window_id,
                    "success": False,
                    "error": "Insufficient data"
                }
            
            # 3. Train model
            logger.info(f"Training {model_type} model...")
            trainer_config = TrainingConfig(
                model_type=model_type,
                optimize_hyperparams=False,  # Disable hyperparameter optimization to avoid early stopping issues
                n_trials=50,  # Reduced for speed
                use_time_series_split=True,
                n_splits=3,
                save_model=False,  # We'll save manually
                feature_selection=False,  # Disable feature selection to avoid feature mismatch
                early_stopping_rounds=0  # Disable early stopping for XGBoost to avoid validation dataset requirement
            )
            
            trainer = ModelTrainer(trainer_config)
            model, train_metrics = trainer.train(X_train, y_train)
            
            # 4. Test model
            logger.info("Testing model on out-of-sample data...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # 5. Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(
                test_df.loc[y_test.index], y_test, y_pred, y_pred_proba
            )
            
            # 6. Save model
            model_path = self.models_dir / f"window_{window_id}_{model_type}.pkl"
            import pickle
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
                "train_accuracy": train_metrics.get("accuracy", 0),
                "train_f1": train_metrics.get("f1", 0),
                "test_accuracy": self._calculate_accuracy(y_test, y_pred),
                "test_f1": self._calculate_f1(y_test, y_pred),
                "model_path": str(model_path),
                **trading_metrics
            }
            
            logger.info(f"Window {window_id} results:")
            logger.info(f"  Test Accuracy: {result['test_accuracy']:.2%}")
            logger.info(f"  Test Sharpe: {result['sharpe']:.2f}")
            logger.info(f"  Test PnL: {result['total_pnl']:.2%}")
            logger.info(f"  Win Rate: {result['win_rate']:.2%}")
            logger.info(f"  Profit Factor: {result['profit_factor']:.2f}")
            
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
    
    def _calculate_trading_metrics(
        self,
        test_df: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate trading performance metrics.
        
        Args:
            test_df: Test window OHLCV data
            y_true: True labels (1 = BUY, 0 = IGNORE)
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with trading metrics
        """
        # Use probability threshold for trading decisions
        # Only enter when confidence > 0.65 (as per user requirement)
        if y_pred_proba is not None:
            trade_signals = (y_pred_proba > 0.65).astype(int)
        else:
            trade_signals = y_pred
        
        # Calculate returns
        returns = test_df["close"].pct_change().shift(-1)  # Next period return
        
        # Align indices
        common_idx = returns.index.intersection(y_true.index)
        returns = returns.loc[common_idx]
        trade_signals = pd.Series(trade_signals, index=y_true.index).loc[common_idx]
        
        # Strategy returns: long when signal=1
        strategy_returns = returns * trade_signals
        
        # Remove NaN
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {
                "total_pnl": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "n_trades": 0,
                "avg_trade_return": 0.0
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
        
        # Total PnL
        total_pnl = (1 + strategy_returns).prod() - 1
        
        # Sharpe ratio (annualized)
        if strategy_returns.std() > 0:
            # Assuming 5m data: 288 candles per day, 252 trading days
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252 * 288)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Average trade return
        avg_trade_return = strategy_returns.mean()
        
        return {
            "total_pnl": total_pnl,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "avg_trade_return": avg_trade_return
        }
    
    def _calculate_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    
    def _calculate_f1(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    def run(
        self,
        pair: str,
        timeframe: str = "5m",
        train_months: int = 3,
        test_months: int = 1,
        step_months: int = 1,
        model_type: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward backtest.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe
            train_months: Training window size in months
            test_months: Testing window size in months
            step_months: Step size in months
            model_type: Model type ("xgboost" or "lightgbm")
            
        Returns:
            Dictionary with aggregated results
        """
        logger.info(f"\n{'='*70}")
        logger.info("WALK-FORWARD BACKTEST ENGINE")
        logger.info(f"{'='*70}")
        logger.info(f"Pair: {pair}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Train window: {train_months} months")
        logger.info(f"Test window: {test_months} months")
        logger.info(f"Step size: {step_months} months")
        logger.info(f"Model type: {model_type}")
        logger.info(f"{'='*70}")
        
        # 1. Load data
        logger.info("Loading data...")
        df = self.load_data(pair, timeframe)
        
        # 2. Create windows
        logger.info("Creating sliding windows...")
        windows = self.create_windows(df, train_months, test_months, step_months)
        
        if not windows:
            raise ValueError("No windows created - insufficient data")
        
        # 3. Process each window
        self.window_results = []
        self.cumulative_pnl = []
        self.equity_curve = []
        
        cumulative_equity = 1.0
        
        for window_id, (train_df, test_df) in enumerate(windows):
            result = self.train_and_test_window(window_id, train_df, test_df, model_type)
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
        logger.info("WALK-FORWARD BACKTEST COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total cumulative PnL: {aggregated['total_cumulative_pnl']:.2%}")
        logger.info(f"Average window PnL: {aggregated['avg_window_pnl']:.2%}")
        logger.info(f"Average Sharpe: {aggregated['avg_sharpe']:.2f}")
        logger.info(f"Average Profit Factor: {aggregated['avg_profit_factor']:.2f}")
        logger.info(f"Win Rate: {aggregated['avg_win_rate']:.2%}")
        
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
                "n_windows": 0,
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
        ax1.plot(self.equity_curve, label="Equity Curve", linewidth=2)
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
            ax2.hist(window_pnls, bins=20, edgecolor='black', alpha=0.7)
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
            ax3.bar(window_ids, sharpe_ratios, alpha=0.7)
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
            ax4.bar(window_ids, profit_factors, alpha=0.7)
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Walk-Forward Backtest Engine for ML Trading Strategies"
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
        "--train-months",
        type=int,
        default=3,
        help="Training window size in months"
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=1,
        help="Testing window size in months"
    )
    parser.add_argument(
        "--step-months",
        type=int,
        default=1,
        help="Step size in months"
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="Model type"
    )
    parser.add_argument(
        "--data-path",
        default="user_data/data/binance",
        help="Path to data directory"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run backtest
        backtest = WalkForwardBacktest(data_path=args.data_path)
        results = backtest.run(
            pair=args.pair,
            timeframe=args.timeframe,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            model_type=args.model
        )
        
        # Print summary
        print("\n" + "="*70)
        print("WALK-FORWARD BACKTEST SUMMARY")
        print("="*70)
        print(f"Total Cumulative PnL: {results['total_cumulative_pnl']:.2%}")
        print(f"Average Window PnL: {results['avg_window_pnl']:.2%}")
        print(f"Average Sharpe Ratio: {results['avg_sharpe']:.2f}")
        print(f"Average Profit Factor: {results['avg_profit_factor']:.2f}")
        print(f"Average Win Rate: {results['avg_win_rate']:.2%}")
        print(f"Successful Windows: {results['n_successful']}/{results['n_windows']}")
        print("="*70)
        
        # Interpretation
        if results['avg_profit_factor'] > 1.1:
            print("✅ STRATEGY PASSES: Profit Factor > 1.1 (likely profitable)")
        else:
            print("⚠️  STRATEGY MARGINAL: Profit Factor <= 1.1 (needs improvement)")
        
        if results['avg_sharpe'] > 0.5:
            print("✅ Good risk-adjusted returns (Sharpe > 0.5)")
        elif results['avg_sharpe'] > 0:
            print("⚠️  Marginal risk-adjusted returns")
        else:
            print("❌ Poor risk-adjusted returns")
        
    except Exception as e:
        logger.error(f"Walk-forward backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
