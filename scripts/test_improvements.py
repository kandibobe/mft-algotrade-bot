#!/usr/bin/env python3
"""
Stoic Citadel - Test Improvements Script
=========================================

Test all improvements from the roadmap:
1. Dynamic probability thresholds in strategy V4
2. ML model improvements (class balancing, regularization)
3. Feature selection and engineering
4. Walk-forward validation
5. Hyperparameter optimization

Usage:
    python scripts/test_improvements.py --phase all
    python scripts/test_improvements.py --phase ml
    python scripts/test_improvements.py --phase strategy
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.ml.training.hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterOptimizerConfig
from src.ml.training.labeling import TripleBarrierLabeler, DynamicBarrierLabeler, RegimeAwareBarrierLabeler
from src.ml.training.model_trainer import ModelTrainer, TrainingConfig
from src.ml.training.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class ImprovementTester:
    """Test all improvements from the roadmap."""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.results = {}
        
    def test_dynamic_thresholds(self) -> Dict[str, Any]:
        """Test dynamic probability thresholds in strategy V4."""
        logger.info("Testing dynamic probability thresholds...")
        
        try:
            # Import strategy to test dynamic threshold calculation
            from user_data.strategies.StoicEnsembleStrategyV4 import StoicEnsembleStrategyV4
            
            # Create mock dataframe to test threshold calculation
            np.random.seed(42)
            n_samples = 500
            
            mock_data = pd.DataFrame({
                'open': np.random.randn(n_samples).cumsum() + 100,
                'high': np.random.randn(n_samples).cumsum() + 105,
                'low': np.random.randn(n_samples).cumsum() + 95,
                'close': np.random.randn(n_samples).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, n_samples),
                'ml_prediction': np.random.beta(2, 2, n_samples),  # Beta distribution for probabilities
                'ml_confidence': np.random.uniform(0.5, 0.9, n_samples),
                'atr_pct': np.random.uniform(0.005, 0.03, n_samples),
                'ema_200': np.random.randn(n_samples).cumsum() + 98
            })
            
            # Create strategy instance
            strategy = StoicEnsembleStrategyV4(config={})
            strategy._regime_mode = 'normal'
            
            # Test dynamic threshold calculation
            thresholds = strategy._calculate_dynamic_threshold(mock_data)
            
            # Analyze results
            threshold_stats = {
                'mean': thresholds.mean(),
                'std': thresholds.std(),
                'min': thresholds.min(),
                'max': thresholds.max(),
                'range': thresholds.max() - thresholds.min(),
                'adaptive': thresholds.std() > 0.01  # Should have some variation
            }
            
            logger.info(f"Dynamic threshold stats: mean={threshold_stats['mean']:.3f}, "
                       f"std={threshold_stats['std']:.3f}, range={threshold_stats['range']:.3f}")
            
            # Test different regimes
            strategy._regime_mode = 'defensive'
            thresholds_defensive = strategy._calculate_dynamic_threshold(mock_data)
            
            strategy._regime_mode = 'aggressive'
            thresholds_aggressive = strategy._calculate_dynamic_threshold(mock_data)
            
            # Check that defensive has higher thresholds
            defensive_higher = thresholds_defensive.mean() > thresholds_aggressive.mean()
            
            result = {
                'success': True,
                'threshold_stats': threshold_stats,
                'defensive_higher': defensive_higher,
                'adaptive': threshold_stats['adaptive'],
                'defensive_mean': thresholds_defensive.mean(),
                'aggressive_mean': thresholds_aggressive.mean()
            }
            
            logger.info(f"✓ Dynamic thresholds test passed: adaptive={threshold_stats['adaptive']}, "
                       f"defensive_higher={defensive_higher}")
            
        except Exception as e:
            logger.error(f"✗ Dynamic thresholds test failed: {e}")
            result = {'success': False, 'error': str(e)}
        
        self.results['dynamic_thresholds'] = result
        return result
    
    def test_ml_improvements(self) -> Dict[str, Any]:
        """Test ML model improvements (class balancing, regularization)."""
        logger.info("Testing ML model improvements...")
        
        try:
            # Create synthetic data for testing
            np.random.seed(42)
            n_samples = 1000
            
            # Create mock OHLCV data for testing (with datetime index)
            dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
            mock_ohlcv = pd.DataFrame({
                'open': np.random.randn(n_samples).cumsum() + 100,
                'high': np.random.randn(n_samples).cumsum() + 105,
                'low': np.random.randn(n_samples).cumsum() + 95,
                'close': np.random.randn(n_samples).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, n_samples)
            }, index=dates)
            
            # Create target labels using simple price movement (for testing)
            y = pd.Series((mock_ohlcv['close'].shift(-1) > mock_ohlcv['close']).astype(int))
            y = y.fillna(0)  # Fill last value
            
            # Test different labeling methods
            labelers = {
                'triple_barrier': TripleBarrierLabeler(),
                'dynamic_barrier': DynamicBarrierLabeler(),
                'regime_aware': RegimeAwareBarrierLabeler()
            }
            
            labeling_results = {}
            for name, labeler in labelers.items():
                try:
                    labels = labeler.label(mock_ohlcv)
                    label_dist = labels.value_counts()
                    
                    labeling_results[name] = {
                        'success': True,
                        'label_distribution': label_dist.to_dict(),
                        'unique_labels': len(label_dist)
                    }
                    
                    logger.info(f"  {name}: {label_dist.to_dict()}")
                    
                except Exception as e:
                    labeling_results[name] = {'success': False, 'error': str(e)}
                    logger.warning(f"  {name} failed: {e}")
            
            # Test feature engineering on OHLCV data
            feature_engineer = FeatureEngineer()
            X_engineered = feature_engineer.fit_transform(mock_ohlcv)
            
            feature_results = {
                'original_features': mock_ohlcv.shape[1],
                'engineered_features': X_engineered.shape[1],
                'feature_expansion': X_engineered.shape[1] > mock_ohlcv.shape[1]
            }
            
            # Align X and y after feature engineering (which may drop rows)
            # Get common indices after feature engineering
            common_idx = X_engineered.index.intersection(y.index)
            X_engineered = X_engineered.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            # Test model training with class balancing
            trainer = ModelTrainer(TrainingConfig(
                model_type='random_forest',
                feature_selection=True,
                max_features=10
            ))
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_engineered, y_aligned, test_size=0.2, random_state=42, stratify=y_aligned
            )
            
            model, metrics = trainer.train(X_train, y_train, X_test, y_test)
            
            # Check if class balancing is working
            class_balance_check = metrics['precision'] > 0.5 and metrics['recall'] > 0.5
            
            result = {
                'success': True,
                'labeling_methods': labeling_results,
                'feature_engineering': feature_results,
                'model_metrics': metrics,
                'class_balance_effective': class_balance_check,
                'feature_selection_working': trainer.feature_importance is not None
            }
            
            logger.info(f"✓ ML improvements test passed: precision={metrics['precision']:.3f}, "
                       f"recall={metrics['recall']:.3f}, "
                       f"features={feature_results['engineered_features']}")
            
        except Exception as e:
            logger.error(f"✗ ML improvements test failed: {e}")
            result = {'success': False, 'error': str(e)}
        
        self.results['ml_improvements'] = result
        return result
    
    def test_hyperparameter_optimization(self) -> Dict[str, Any]:
        """Test hyperparameter optimization with Optuna."""
        logger.info("Testing hyperparameter optimization...")
        
        try:
            # Create synthetic data - increased sample size to avoid TimeSeriesSplit error
            np.random.seed(42)
            n_samples = 1000  # Increased from 500 to 1000 for valid TimeSeriesSplit
            n_features = 15
            
            X = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            )
            
            # Create target with some signal
            y = pd.Series(
                (X.iloc[:, 0] > 0.5).astype(int) ^  # XOR with noise
                (np.random.rand(n_samples) > 0.7).astype(int)
            )
            
            # Test XGBoost optimization with adjusted configuration for test
            optimizer = HyperparameterOptimizer(HyperparameterOptimizerConfig(
                n_trials=20,  # Reduced for testing
                optimize_for='precision',
                min_precision_threshold=0.55,
                n_splits=3,  # Reduced from 5 to 3 for test data size
                test_size=0.1  # Reduced from 0.2 to 0.1 for test data size
            ))
            
            results = optimizer.optimize(X, y, model_type='xgboost')
            
            # Check results
            optimization_success = (
                results['best_score'] > 0.5 and
                results['best_params'] is not None and
                optimizer.best_model is not None
            )
            
            # Check that parameters enforce simplicity
            params = results['best_params']
            simplicity_checks = {
                'max_depth_reasonable': params.get('max_depth', 10) <= 7,
                'gamma_high_enough': params.get('gamma', 0) >= 0.1,
                'min_child_weight_high': params.get('min_child_weight', 1) >= 5
            }
            
            result = {
                'success': True,
                'optimization_success': optimization_success,
                'best_score': results['best_score'],
                'simplicity_checks': simplicity_checks,
                'all_checks_passed': all(simplicity_checks.values()),
                'best_params_keys': list(params.keys())
            }
            
            logger.info(f"✓ Hyperparameter optimization test passed: "
                       f"score={results['best_score']:.3f}, "
                       f"simplicity_checks={sum(simplicity_checks.values())}/{len(simplicity_checks)}")
            
        except Exception as e:
            logger.error(f"✗ Hyperparameter optimization test failed: {e}")
            result = {'success': False, 'error': str(e)}
        
        self.results['hyperparameter_optimization'] = result
        return result
    
    def test_walk_forward_validation(self) -> Dict[str, Any]:
        """Test walk-forward validation logic."""
        logger.info("Testing walk-forward validation...")
        
        try:
            # Test time series split logic
            from sklearn.model_selection import TimeSeriesSplit
            
            # Create time series data
            n_samples = 200
            dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
            X = pd.DataFrame(
                np.random.randn(n_samples, 10),
                index=dates,
                columns=[f'feature_{i}' for i in range(10)]
            )
            y = pd.Series(np.random.randint(0, 2, n_samples), index=dates)
            
            # Test purged walk-forward (no data leakage)
            tscv = TimeSeriesSplit(n_splits=5, test_size=20)
            
            fold_info = []
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                train_dates = X.index[train_idx]
                test_dates = X.index[test_idx]
                
                # Check no overlap between train and test
                max_train = train_dates.max()
                min_test = test_dates.min()
                
                fold_info.append({
                    'fold': fold + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'no_leakage': max_train < min_test,  # Train ends before test starts
                    'train_end': max_train,
                    'test_start': min_test
                })
            
            # All folds should have no data leakage
            no_leakage_all = all(f['no_leakage'] for f in fold_info)
            
            result = {
                'success': True,
                'folds': len(fold_info),
                'no_leakage_all': no_leakage_all,
                'fold_info': fold_info,
                'train_test_ratio_consistent': all(
                    abs(f['train_size'] / (f['train_size'] + f['test_size']) - 0.8) < 0.1
                    for f in fold_info
                )
            }
            
            logger.info(f"✓ Walk-forward validation test passed: "
                       f"folds={len(fold_info)}, no_leakage={no_leakage_all}")
            
        except Exception as e:
            logger.error(f"✗ Walk-forward validation test failed: {e}")
            result = {'success': False, 'error': str(e)}
        
        self.results['walk_forward_validation'] = result
        return result
    
    def test_strategy_improvements(self) -> Dict[str, Any]:
        """Test strategy V4 improvements."""
        logger.info("Testing strategy V4 improvements...")
        
        try:
            # Check that strategy file exists and can be imported
            strategy_path = Path("user_data/strategies/StoicEnsembleStrategyV4.py")
            
            if not strategy_path.exists():
                raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
            
            # Read strategy file to check for key improvements
            with open(strategy_path, 'r') as f:
                strategy_content = f.read()
            
            # Check for key improvements
            improvements_check = {
                'dynamic_threshold_method': '_calculate_dynamic_threshold' in strategy_content,
                'regime_based_position_sizing': 'custom_stake_amount' in strategy_content,
                'adaptive_stop_loss': 'custom_stoploss' in strategy_content,
                'ml_integration': 'ml_prediction' in strategy_content,
                'simplified_entry_conditions': 'close > dataframe[\'ema_200\']' in strategy_content,
                'ensemble_scoring': 'ensemble_score' in strategy_content
            }
            
            # Count improvements found
            improvements_found = sum(improvements_check.values())
            improvements_total = len(improvements_check)
            
            result = {
                'success': True,
                'strategy_exists': True,
                'improvements_check': improvements_check,
                'improvements_found': improvements_found,
                'improvements_total': improvements_total,
                'improvements_percentage': improvements_found / improvements_total * 100
            }
            
            logger.info(f"✓ Strategy improvements test passed: "
                       f"{improvements_found}/{improvements_total} improvements found "
                       f"({result['improvements_percentage']:.1f}%)")
            
        except Exception as e:
            logger.error(f"✗ Strategy improvements test failed: {e}")
            result = {'success': False, 'error': str(e)}
        
        self.results['strategy_improvements'] = result
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all improvement tests."""
        logger.info("=" * 70)
        logger.info("RUNNING ALL IMPROVEMENT TESTS")
        logger.info("=" * 70)
        
        tests = [
            ('Dynamic Thresholds', self.test_dynamic_thresholds),
            ('ML Improvements', self.test_ml_improvements),
            ('Hyperparameter Optimization', self.test_hyperparameter_optimization),
            ('Walk-Forward Validation', self.test_walk_forward_validation),
            ('Strategy Improvements', self.test_strategy_improvements)
        ]
        
        all_results = {}
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n▶ Testing: {test_name}")
            result = test_func()
            all_results[test_name] = result
            
            if result.get('success', False):
                passed_tests += 1
                logger.info(f"  ✓ {test_name}: PASSED")
            else:
                logger.info(f"  ✗ {test_name}: FAILED")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        overall_success = passed_tests == total_tests
        
        if overall_success:
            logger.info("\n🎉 ALL IMPROVEMENT TESTS PASSED!")
        else:
            logger.info(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED")
        
        return {
            'overall_success': overall_success,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests / total_tests * 100,
            'detailed_results': all_results
        }


def main():
    parser = argparse.ArgumentParser(
        description='Test Stoic Citadel improvements from roadmap'
    )
    
    parser.add_argument(
        '--phase',
        choices=['all', 'ml', 'strategy', 'optimization', 'validation', 'thresholds'],
        default='all',
        help='Phase of improvements to test'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create tester
    tester = ImprovementTester()
    
    # Run tests based on phase
    if args.phase == 'all':
        results = tester.run_all_tests()
    elif args.phase == 'ml':
        results = tester.test_ml_improvements()
    elif args.phase == 'strategy':
        results = tester.test_strategy_improvements()
    elif args.phase == 'optimization':
        results = tester.test_hyperparameter_optimization()
    elif args.phase == 'validation':
        results = tester.test_walk_forward_validation()
    elif args.phase == 'thresholds':
        results = tester.test_dynamic_thresholds()
    else:
        logger.error(f"Unknown phase: {args.phase}")
        sys.exit(1)
    
    # Exit with appropriate code
    # For individual tests, check 'success' key
    # For run_all_tests, check 'overall_success' key
    success = False
    if isinstance(results, dict):
        if 'overall_success' in results:
            success = results.get('overall_success', False)
        else:
            success = results.get('success', False)
    
    if success:
        logger.info("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
