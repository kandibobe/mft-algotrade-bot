#!/usr/bin/env python3
"""
ML Model Training Script
=========================

Обучает ML модели на исторических данных.

Features:
- Автоматическая загрузка данных из JSON
- Feature engineering с валидацией
- Triple Barrier labeling
- Обучение Random Forest, XGBoost, LightGBM
- Сохранение моделей и метрик

Usage:
    # Train on all available data
    python scripts/train_models.py

    # Train specific pairs
    python scripts/train_models.py --pairs BTC/USDT ETH/USDT

    # Quick training (for testing)
    python scripts/train_models.py --quick

    # With hyperparameter optimization
    python scripts/train_models.py --optimize --trials 50
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.training.feature_engineering import FeatureEngineer
from src.ml.training.labeling import TripleBarrierLabeler, LabelingConfig
from src.ml.training.model_trainer import ModelTrainer, TrainingConfig
from src.ml.training.model_registry import ModelRegistry


class MLTrainingPipeline:
    """Полный пайплайн обучения ML моделей."""

    def __init__(
        self,
        data_dir: str = "user_data/data/binance",
        models_dir: str = "user_data/models",
        quick_mode: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.quick_mode = quick_mode

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.labeler = TripleBarrierLabeler(config=LabelingConfig())
        self.registry = ModelRegistry()

    def load_data(self, pairs: List[str], timeframe: str = "5m") -> Dict[str, pd.DataFrame]:
        """
        Загрузить данные из JSON файлов.

        Args:
            pairs: Список торговых пар
            timeframe: Таймфрейм

        Returns:
            Dict с данными для каждой пары
        """
        print("\n" + "="*70)
        print("📥 LOADING DATA")
        print("="*70)

        data = {}

        for pair in pairs:
            pair_filename = pair.replace('/', '_')
            filename = f"{pair_filename}-{timeframe}.json"
            filepath = self.data_dir / filename

            if not filepath.exists():
                print(f"⚠️  {pair}: File not found - {filepath}")
                continue

            print(f"\n📊 Loading {pair}...")

            # Load JSON
            with open(filepath, 'r') as f:
                json_data = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(
                json_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            print(f"  ✅ Loaded {len(df):,} candles")
            print(f"  📅 Range: {df.index[0]} to {df.index[-1]}")

            # In quick mode, use only recent data
            if self.quick_mode:
                df = df.tail(1000)
                print(f"  ⚡ Quick mode: Using last 1000 candles")

            data[pair] = df

        if not data:
            raise ValueError(f"No data found in {self.data_dir}")

        print("\n" + "="*70)
        return data

    def prepare_features_and_labels(
        self,
        df: pd.DataFrame,
        pair: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Подготовить features и labels.

        Args:
            df: Данные OHLCV
            pair: Торговая пара

        Returns:
            (X, y) - features и labels
        """
        print(f"\n🔧 Feature Engineering for {pair}...")

        # Generate features
        features = self.feature_engineer.generate_features(
            df,
            mode='train',  # Auto-fix issues
            symbol=pair
        )

        print(f"  ✅ Generated {len(features.columns)} features")

        # Validate features
        is_valid, issues = self.feature_engineer.validate_features(
            features,
            fix_issues=True,
            raise_on_error=False
        )

        if issues['nan_columns']:
            print(f"  ⚠️  Fixed NaN in {len(issues['nan_columns'])} columns")
        if issues['inf_columns']:
            print(f"  ⚠️  Fixed Inf in {len(issues['inf_columns'])} columns")

        # Generate labels
        print(f"\n🏷️  Labeling for {pair}...")

        labels = self.labeler.label_data(df)

        print(f"  ✅ Generated {len(labels)} labels")
        print(f"  📊 Label distribution:")

        label_counts = labels.value_counts()
        for label_val, count in label_counts.items():
            label_name = {1: 'LONG', 0: 'NEUTRAL', -1: 'SHORT'}.get(label_val, 'UNKNOWN')
            pct = (count / len(labels)) * 100
            print(f"     {label_name:8s}: {count:6,} ({pct:5.1f}%)")

        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.loc[common_index]
        y = labels.loc[common_index]

        print(f"\n  ✅ Final dataset: {len(X):,} samples, {len(X.columns)} features")

        return X, y

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pair: str,
        optimize: bool = False,
        n_trials: int = 100
    ) -> tuple[Any, Dict[str, float]]:
        """
        Обучить модель.

        Args:
            X: Features
            y: Labels
            pair: Торговая пара
            optimize: Оптимизировать гиперпараметры
            n_trials: Количество trials для оптимизации

        Returns:
            (model, metrics)
        """
        print(f"\n🤖 Training Model for {pair}...")

        # Split data (time-aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"  📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Configure trainer
        config = TrainingConfig(
            model_type="random_forest",  # Change to xgboost or lightgbm if installed
            optimize_hyperparams=optimize,
            n_trials=n_trials if optimize else 10,
            use_time_series_split=True,
            n_splits=3 if self.quick_mode else 5,
            save_model=True,
            models_dir=str(self.models_dir)
        )

        # Train
        trainer = ModelTrainer(config)
        model, metrics = trainer.train(X_train, y_train, X_test, y_test)

        print(f"\n  ✅ Model trained successfully!")
        print(f"  📊 Test Metrics:")
        for metric_name, value in metrics.items():
            print(f"     {metric_name:15s}: {value:.4f}")

        # Save model
        model_name = f"{pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.models_dir / f"{model_name}.pkl"

        self.registry.save_model(
            model=model,
            name=model_name,
            metadata={
                'pair': pair,
                'features': X.columns.tolist(),
                'metrics': metrics,
                'training_date': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
            }
        )

        print(f"  💾 Model saved: {model_path}")

        return model, metrics

    def run(
        self,
        pairs: List[str],
        timeframe: str = "5m",
        optimize: bool = False,
        n_trials: int = 100
    ):
        """
        Запустить полный пайплайн обучения.

        Args:
            pairs: Список торговых пар
            timeframe: Таймфрейм
            optimize: Оптимизировать гиперпараметры
            n_trials: Количество trials
        """
        print("\n" + "="*70)
        print("🚀 ML MODEL TRAINING PIPELINE")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Pairs:      {', '.join(pairs)}")
        print(f"  Timeframe:  {timeframe}")
        print(f"  Optimize:   {optimize}")
        print(f"  Quick mode: {self.quick_mode}")

        # Load data
        data = self.load_data(pairs, timeframe)

        # Train models for each pair
        results = {}

        for pair, df in data.items():
            print("\n" + "="*70)
            print(f"🎯 TRAINING: {pair}")
            print("="*70)

            try:
                # Prepare features and labels
                X, y = self.prepare_features_and_labels(df, pair)

                # Train model
                model, metrics = self.train_model(X, y, pair, optimize, n_trials)

                results[pair] = {
                    'success': True,
                    'metrics': metrics
                }

            except Exception as e:
                print(f"\n❌ Error training {pair}: {e}")
                import traceback
                traceback.print_exc()

                results[pair] = {
                    'success': False,
                    'error': str(e)
                }

        # Summary
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY")
        print("="*70)

        for pair, result in results.items():
            if result['success']:
                metrics = result['metrics']
                print(f"\n✅ {pair}")
                print(f"   Accuracy:  {metrics.get('accuracy', 0):.4f}")
                print(f"   F1 Score:  {metrics.get('f1', 0):.4f}")
            else:
                print(f"\n❌ {pair}")
                print(f"   Error: {result['error']}")

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"\nModels saved to: {self.models_dir}")
        print("\nNext steps:")
        print("  1. Run backtest: python scripts/run_backtest.py --profile full")
        print("  2. View models:  ls user_data/models/")
        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models on historical data"
    )

    parser.add_argument(
        '--pairs',
        type=str,
        nargs='+',
        default=['BTC/USDT', 'ETH/USDT'],
        help='Trading pairs to train on'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='5m',
        help='Timeframe'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='user_data/data/binance',
        help='Data directory'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='user_data/models',
        help='Models output directory'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (use only recent 1000 candles)'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable hyperparameter optimization'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of optimization trials'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = MLTrainingPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        quick_mode=args.quick
    )

    # Run training
    pipeline.run(
        pairs=args.pairs,
        timeframe=args.timeframe,
        optimize=args.optimize,
        n_trials=args.trials
    )


if __name__ == "__main__":
    main()
