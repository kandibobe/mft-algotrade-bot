"""
ML Training Pipeline Module
===========================

Core logic for training ML models.
Delegates training and optimization to ModelOptimizer.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

from src.ml.training.optimizer import ModelOptimizer
from src.config.manager import ConfigurationManager
from src.ml.feature_store import TradingFeatureStore, create_feature_store

logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """Full ML Training Pipeline."""

    def __init__(
        self,
        data_dir: str = None,
        models_dir: str = None,
        quick_mode: bool = False,
        use_feature_store: bool = False
    ):
        self.config = ConfigurationManager.get_config()
        exchange_name = self.config.exchange.name

        self.data_dir = Path(data_dir or f"user_data/data/{exchange_name}")
        self.models_dir = Path(models_dir or "user_data/models")
        self.quick_mode = quick_mode
        self.use_feature_store = use_feature_store

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Optimizer
        self.optimizer = ModelOptimizer(self.config, models_dir=str(self.models_dir))

        # Initialize Feature Store if enabled
        if self.use_feature_store:
            logger.info("Initializing Feature Store...")
            try:
                self.feature_store = create_feature_store(
                    use_redis=self.config.feature_store.use_redis
                )
                self.feature_store.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize Feature Store: {e}")
                self.use_feature_store = False # Fallback to file-based loading

    def load_data(self, pairs: List[str], timeframe: str = "5m", days: int = None) -> Dict[str, pd.DataFrame]:
        """
        Load data from Feature Store or local files.
        
        Args:
            pairs: List of pairs to load
            timeframe: Candle timeframe
            days: Number of days of history to load (None = all)
        """
        logger.info("="*70)
        logger.info("📥 LOADING DATA")
        logger.info("="*70)

        if self.use_feature_store:
            logger.info("🔌 Using Feature Store to load offline features...")
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days) if days else datetime(2020, 1, 1) # Default start date if days not specified
                
                # Note: Feature store might return a single dataframe for all pairs.
                # We need to split it by pair.
                full_df = self.feature_store.get_offline_features(
                    symbols=pairs,
                    start_date=start_date,
                    end_date=end_date,
                )

                if full_df.empty:
                    logger.warning("Feature Store returned no data.")
                    return {}

                # The feature store returns a single dataframe with a 'symbol_id' column
                # We need to split this into a dictionary of dataframes, keyed by pair
                data = {
                    pair: group.drop(columns=['symbol_id']).set_index('timestamp')
                    for pair, group in full_df.groupby('symbol_id')
                }
                
                for pair, df in data.items():
                     logger.info(f"  ✅ Loaded {len(df):,} features for {pair}")
                     logger.info(f"  📅 Range: {df.index[0]} to {df.index[-1]}")

                return data

            except Exception as e:
                logger.error(f"Feature Store failed: {e}. Falling back to file loading.")
                # Fallback to file-based loading if feature store fails
        
        # Original file-based loading logic
        data = {}
        for pair in pairs:
            pair_filename = pair.replace('/', '_')
            feather_filename = f"{pair_filename}-{timeframe}.feather"
            json_filename = f"{pair_filename}-{timeframe}.json"
            
            feather_path = self.data_dir / feather_filename
            json_path = self.data_dir / json_filename

            if feather_path.exists():
                filepath = feather_path
                filetype = 'feather'
            elif json_path.exists():
                filepath = json_path
                filetype = 'json'
            else:
                logger.warning(f"⚠️  {pair}: File not found - {feather_filename} or {json_filename}")
                continue

            logger.info(f"📊 Loading {pair} from {filetype}...")

            if filetype == 'feather':
                df = pd.read_feather(filepath)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    df.set_index(df.columns[0], inplace=True)
                    df.index = pd.to_datetime(df.index, unit='ms')
            else:
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                df = pd.DataFrame(
                    json_data,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)

            logger.info(f"  ✅ Loaded {len(df):,} candles")
            logger.info(f"  📅 Range: {df.index[0]} to {df.index[-1]}")

            # Data Validation
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if df[required_cols].isnull().any().any():
                nan_count = df[required_cols].isnull().sum().sum()
                logger.warning(f"  ⚠️  {pair}: Found {nan_count} NaNs in OHLCV data. Dropping...")
                df = df.dropna(subset=required_cols)
                
            if len(df) < 100:
                logger.warning(f"  ⚠️  {pair}: Insufficient data ({len(df)} rows). Skipping.")
                continue

            if self.quick_mode:
                df = df.tail(1000)
                logger.info(f"  ⚡ Quick mode: Using last 1000 candles")
            elif days:
                start_date = df.index[-1] - pd.Timedelta(days=days)
                df = df[df.index >= start_date]
                logger.info(f"  📅 Filtered to last {days} days: {len(df)} candles")

            data[pair] = df

        if not data:
            raise ValueError(f"No data found in {self.data_dir}")

        logger.info("="*70)
        return data

    def train_and_get_model(self, df: pd.DataFrame, pair: str, optimize: bool = False) -> Any:
        """
        Train a model for a single pair and return the model object.

        Args:
            df: DataFrame with the training data.
            pair: The trading pair.
            optimize: Whether to run hyperparameter optimization.

        Returns:
            The trained model object, or None if training failed.
        """
        logger.info("\n" + "="*70)
        logger.info(f"🎯 TRAINING: {pair}")
        logger.info("="*70)

        try:
            # Delegate to ModelOptimizer
            result = self.optimizer.train(df, pair, optimize=optimize)
            
            if result.get("success"):
                return result.get("model")
            else:
                logger.error(f"Training failed for {pair}: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"\n❌ Error training {pair}: {e}", exc_info=True)
            return None

    def run(
        self,
        pairs: List[str],
        timeframe: str = "5m",
        optimize: bool = False,
        n_trials: int = 100,
        days: int = None
    ) -> Dict[str, Any]:
        """Run full training pipeline."""
        logger.info("="*70)
        logger.info("🚀 ML MODEL TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"\nConfiguration:")
        logger.info(f"  Pairs:      {', '.join(pairs)}")
        logger.info(f"  Timeframe:  {timeframe}")
        logger.info(f"  Optimize:   {optimize}")
        logger.info(f"  Quick mode: {self.quick_mode}")
        logger.info(f"  Use Feature Store: {self.use_feature_store}")
        if days:
            logger.info(f"  Days:       {days}")

        data = self.load_data(pairs, timeframe, days=days)
        results = {}

        # Update config with CLI args
        if optimize:
            self.config.training.hyperopt_trials = n_trials

        for pair, df in data.items():
            model = self.train_and_get_model(df, pair, optimize=optimize)
            if model:
                # For now, we're not doing anything with the model in the `run` command,
                # but we could save it or log its metrics.
                # The result format from `train` is what we need to replicate.
                # This part needs to be adjusted based on the return value of `train_and_get_model`.
                # For now, we will just mark it as success.
                results[pair] = {'success': True}
            else:
                results[pair] = {'success': False, 'error': "Training failed"}

        logger.info("\n" + "="*70)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("="*70)

        for pair, result in results.items():
            if result['success']:
                logger.info(f"\n✅ {pair}")
            else:
                logger.info(f"\n❌ {pair}")
                logger.info(f"   Error: {result['error']}")

        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING COMPLETED!")
        logger.info("="*70)
        
        return results
