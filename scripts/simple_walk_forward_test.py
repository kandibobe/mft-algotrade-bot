#!/usr/bin/env python3
"""
Simple Walk-Forward Test for ML Trading Strategy
Упрощенная версия для тестирования на малом количестве данных.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleWalkForwardTest:
    """Упрощенный walk-forward тест для малого количества данных."""
    
    def __init__(self, data_path: str = "user_data/data/binance"):
        self.data_path = Path(data_path)
        self.results_dir = Path("user_data/walk_forward_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, pair: str, timeframe: str = "5m") -> pd.DataFrame:
        """Загрузить данные."""
        pair_filename = pair.replace("/", "_")
        path = self.data_path / f"{pair_filename}-{timeframe}.feather"
        
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        
        df = pd.read_feather(path)
        
        # Ensure datetime index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        
        logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
    
    def create_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создать простые фичи для ML."""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['returns_log'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        features['volume_change'] = df['volume'].pct_change()
        features['volume_sma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma']
        
        # Simple indicators
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['price_vs_sma20'] = df['close'] / features['sma_20'] - 1
        features['price_vs_sma50'] = df['close'] / features['sma_50'] - 1
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def create_triple_barrier_labels(self, df: pd.DataFrame, 
                                     take_profit: float = 0.015,
                                     stop_loss: float = 0.0075,
                                     max_periods: int = 24) -> pd.Series:
        """
        Создать метки Triple Barrier (бинарная классификация).
        
        Label = 1 если цена достигает Take Profit до Stop Loss в течение max_periods
        Label = 0 в противном случае
        """
        labels = pd.Series(0, index=df.index, dtype=int)
        
        for i in range(len(df) - max_periods):
            start_price = df['close'].iloc[i]
            
            # Проверить следующие max_periods свечей
            for j in range(1, max_periods + 1):
                if i + j >= len(df):
                    break
                    
                current_price = df['close'].iloc[i + j]
                return_pct = (current_price - start_price) / start_price
                
                # Проверить Take Profit
                if return_pct >= take_profit:
                    labels.iloc[i] = 1  # Успешная сделка
                    break
                
                # Проверить Stop Loss
                if return_pct <= -stop_loss:
                    labels.iloc[i] = 0  # Неуспешная сделка
                    break
        
        logger.info(f"Triple Barrier labels: {labels.sum()} positive out of {len(labels)}")
        return labels
    
    def run_test(self, pair: str = "BTC/USDT", timeframe: str = "5m",
                 train_size: float = 0.7) -> Dict[str, Any]:
        """Запустить упрощенный тест."""
        logger.info(f"\n{'='*70}")
        logger.info(f"SIMPLE WALK-FORWARD TEST: {pair} {timeframe}")
        logger.info(f"{'='*70}")
        
        # 1. Загрузить данные
        df = self.load_data(pair, timeframe)
        
        if len(df) < 100:
            raise ValueError(f"Not enough data: {len(df)} candles. Need at least 100.")
        
        # 2. Создать фичи и метки
        logger.info("Creating features and labels...")
        features = self.create_simple_features(df)
        labels = self.create_triple_barrier_labels(df)
        
        # Удалить строки с NaN
        valid_idx = features.index.intersection(labels.index)
        features = features.loc[valid_idx]
        labels = labels.loc[valid_idx]
        
        # 3. Разделить на train/test
        split_idx = int(len(features) * train_size)
        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = labels.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(f"Positive labels in train: {y_train.sum()}/{len(y_train)} ({y_train.mean():.1%})")
        logger.info(f"Positive labels in test: {y_test.sum()}/{len(y_test)} ({y_test.mean():.1%})")
        
        # 4. Обучить модель
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 5. Протестировать модель
        logger.info("Testing model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 6. Рассчитать метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 7. Симуляция торговли
        trading_results = self.simulate_trading(df.loc[X_test.index], y_test, y_pred_proba)
        
        # 8. Собрать результаты
        results = {
            "pair": pair,
            "timeframe": timeframe,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "model_accuracy": accuracy,
            "model_f1": f1,
            "feature_importance": dict(zip(features.columns, model.feature_importances_)),
            **trading_results
        }
        
        # 9. Визуализация
        self.create_visualization(results, df.loc[X_test.index], y_pred_proba)
        
        # 10. Сохранить результаты
        self.save_results(results)
        
        return results
    
    def simulate_trading(self, test_df: pd.DataFrame, 
                        y_true: pd.Series, 
                        y_pred_proba: pd.Series) -> Dict[str, Any]:
        """Симуляция торговли на тестовых данных."""
        # Использовать порог 0.65 для входа (как в стратегии)
        trade_signals = (y_pred_proba > 0.65).astype(int)
        
        # Рассчитать доходность
        returns = test_df['close'].pct_change().shift(-1)
        
        # Выровнять индексы
        common_idx = returns.index.intersection(y_true.index)
        returns = returns.loc[common_idx]
        trade_signals = pd.Series(trade_signals, index=y_true.index).loc[common_idx]
        
        # Доходность стратегии
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
                "avg_trade_return": 0.0
            }
        
        # Рассчитать метрики
        n_trades = trade_signals.sum()
        
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        
        win_rate = len(winning_trades) / max(1, len(winning_trades) + len(losing_trades))
        
        gross_profit = winning_trades.sum()
        gross_loss = abs(losing_trades.sum())
        profit_factor = gross_profit / max(0.0001, gross_loss)
        
        total_pnl = (1 + strategy_returns).prod() - 1
        
        # Sharpe ratio (annualized)
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252 * 288)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
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
    
    def create_visualization(self, results: Dict[str, Any], 
                            test_df: pd.DataFrame,
                            y_pred_proba: pd.Series) -> None:
        """Создать визуализацию результатов."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Цена и сигналы
        ax1 = axes[0, 0]
        ax1.plot(test_df.index, test_df['close'], label='Price', alpha=0.7)
        
        # Показать сигналы покупки
        buy_signals = y_pred_proba > 0.65
        if buy_signals.any():
            buy_points = test_df.index[buy_signals]
            buy_prices = test_df['close'][buy_signals]
            ax1.scatter(buy_points, buy_prices, color='green', 
                       label='Buy Signal', alpha=0.6, s=50)
        
        ax1.set_title(f"Price and Buy Signals - {results['pair']}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Важность фич
        ax2 = axes[0, 1]
        feature_importance = results['feature_importance']
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # Отсортировать
        sorted_idx = np.argsort(importances)[-10:]  # Топ 10
        ax2.barh(np.array(features)[sorted_idx], np.array(importances)[sorted_idx])
        ax2.set_title("Top 10 Feature Importance")
        ax2.set_xlabel("Importance")
        
        # 3. Распределение вероятностей
        ax3 = axes[1, 0]
        ax3.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0.65, color='red', linestyle='--', label='Threshold (0.65)')
        ax3.set_title("Prediction Probability Distribution")
        ax3.set_xlabel("Probability")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Метрики
        ax4 = axes[1, 1]
        metrics = {
            'Accuracy': results['model_accuracy'],
            'F1 Score': results['model_f1'],
            'Win Rate': results['win_rate'],
            'Profit Factor': results['profit_factor'],
            'Sharpe': results['sharpe']
        }
        
        ax4.bar(range(len(metrics)), list(metrics.values()))
        ax4.set_xticks(range(len(metrics)))
        ax4.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax4.set_title("Performance Metrics")
        ax4.set_ylabel("Value")
        
        # Добавить значения на столбцы
        for i, v in enumerate(metrics.values()):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        # Сохранить
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f"simple_test_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_path}")
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Сохранить результаты."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"simple_test_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description="Simple Walk-Forward Test for ML Trading Strategy"
    )
    parser.add_argument(
        "--pair",
        default="BTC/USDT",
        help="Trading pair (e.g., BTC/USDT, ETH/USDT)"
    )
    parser.add_argument(
        "--timeframe",
        default="5m",
        help="Timeframe (e.g., 5m, 15m, 1h)"
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Training set size ratio (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    try:
        test = SimpleWalkForwardTest()
        results = test.run_test(
            pair=args.pair,
            timeframe=args.timeframe,
            train_size=args.train_size
        )
        
        # Вывести результаты
        print("\n" + "="*70)
        print("SIMPLE WALK-FORWARD TEST RESULTS")
        print("="*70)
        print(f"Pair: {results['pair']}")
        print(f"Timeframe: {results['timeframe']}")
        print(f"Train samples: {results['train_samples']}")
        print(f"Test samples: {results['test_samples']}")
        print(f"Model Accuracy: {results['model_accuracy']:.2%}")
        print(f"Model F1 Score: {results['model_f1']:.3f}")
        print(f"Total PnL: {results['total_pnl']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe']:.2f}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Number of Trades: {results['n_trades']}")
        print("="*70)
        
        # Интерпретация
        if results['profit_factor'] > 1.1:
            print("✅ STRATEGY PASSES: Profit Factor > 1.1 (likely profitable)")
        else:
            print("⚠️  STRATEGY MARGINAL: Profit Factor <= 1.1 (needs improvement)")
        
        if results['sharpe'] > 0.5:
            print("✅ Good risk-adjusted returns (Sharpe > 0.5)")
        elif results['sharpe'] > 0:
            print("⚠️  Marginal risk-adjusted returns")
        else:
            print("❌ Poor risk-adjusted returns")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main
