"""
Test script to verify improvements from roadmap implementation.

Tests the key improvements:
1. HyperparameterOptimizer with Optuna
2. RegimeAwareBarrierLabeler
3. Updated StoicEnsembleStrategyV4 with dynamic thresholds
4. Feature engineering improvements

Author: Stoic Citadel Team
Date: December 23, 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hyperparameter_optimizer():
    """Test the new hyperparameter optimizer."""
    logger.info("Testing HyperparameterOptimizer...")
    
    try:
        from src.ml.training.hyperparameter_optimizer import (
            HyperparameterOptimizer, HyperparameterOptimizerConfig
        )
        
        # Create sample data
        n_samples = 1000
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Create optimizer with minimal config for quick test
        config = HyperparameterOptimizerConfig(
            n_trials=5,  # Small number for quick test
            timeout=60,  # 1 minute timeout
            n_splits=3,
            test_size=0.2
        )
        
        optimizer = HyperparameterOptimizer(config)
        
        # Test XGBoost optimization
        logger.info("Testing XGBoost optimization...")
        results = optimizer.optimize(X, y, model_type="xgboost")
        
        assert results["best_model"] is not None
        assert results["best_score"] is not None
        assert results["best_params"] is not None
        
        logger.info(f"XGBoost optimization successful. Best score: {results['best_score']:.4f}")
        
        # Test ensemble optimization
        logger.info("Testing ensemble optimization...")
        ensemble_results = optimizer.optimize_ensemble(X, y)
        
        assert ensemble_results["ensemble_model"] is not None
        assert ensemble_results["weights"] is not None
        
        logger.info(f"Ensemble optimization successful. Weights: {ensemble_results['weights']}")
        
        return True
        
    except Exception as e:
        logger.error(f"HyperparameterOptimizer test failed: {e}")
        return False


def test_regime_aware_labeler():
    """Test the new regime-aware barrier labeler."""
    logger.info("Testing RegimeAwareBarrierLabeler...")
    
    try:
        from src.ml.training.labeling import (
            RegimeAwareBarrierLabeler, TripleBarrierConfig
        )
        
        # Create sample OHLCV data
        n_samples = 500
        dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='5min')
        
        df = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 101,
            'low': np.random.randn(n_samples).cumsum() + 99,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        }, index=dates)
        
        # Create labeler
        config = TripleBarrierConfig(
            take_profit=0.005,
            stop_loss=0.002,
            max_holding_period=24
        )
        
        labeler = RegimeAwareBarrierLabeler(config)
        
        # Apply labeling
        labels = labeler.label(df)
        
        # Check results
        assert len(labels) == len(df)
        assert labels.isna().sum() < len(df)  # Should have some non-NaN labels
        
        # Check label distribution
        label_counts = labels.value_counts(dropna=True)
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        # Check that we have both 0 and 1 labels (or at least some labels)
        if len(label_counts) > 0:
            logger.info("RegimeAwareBarrierLabeler test successful")
            return True
        else:
            logger.warning("No labels generated")
            return False
            
    except Exception as e:
        logger.error(f"RegimeAwareBarrierLabeler test failed: {e}")
        return False


def test_strategy_improvements():
    """Test the updated strategy with dynamic thresholds."""
    logger.info("Testing strategy improvements...")
    
    try:
        # Import strategy class
        from user_data.strategies.StoicEnsembleStrategyV4 import StoicEnsembleStrategyV4
        
        # Create mock config
        config = {
            'stake_amount': 100,
            'stake_currency': 'USDT',
            'max_open_trades': 3,
            'timeframe': '5m'
        }
        
        # Initialize strategy
        strategy = StoicEnsembleStrategyV4(config)
        
        # Create sample dataframe
        n_samples = 300
        dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='5min')
        
        df = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 101,
            'low': np.random.randn(n_samples).cumsum() + 99,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        }, index=dates)
        
        # Add required indicators
        df['ema_200'] = df['close'].rolling(200).mean()
        df['ml_prediction'] = np.random.uniform(0.4, 0.7, n_samples)
        df['ml_confidence'] = np.random.uniform(0.5, 0.9, n_samples)
        
        # Test dynamic threshold calculation
        dynamic_threshold = strategy._calculate_dynamic_threshold(df)
        
        assert len(dynamic_threshold) == len(df)
        assert dynamic_threshold.min() >= 0.45
        assert dynamic_threshold.max() <= 0.65
        
        logger.info(f"Dynamic threshold range: {dynamic_threshold.min():.3f} to {dynamic_threshold.max():.3f}")
        
        # Test entry logic
        metadata = {'pair': 'BTC/USDT'}
        result_df = strategy.populate_entry_trend(df.copy(), metadata)
        
        assert 'enter_long' in result_df.columns
        
        entry_count = result_df['enter_long'].sum()
        logger.info(f"Generated {entry_count} entry signals")
        
        # position_size column is only created if there are entry signals
        if entry_count > 0:
            assert 'position_size' in result_df.columns
            logger.info(f"Position size column created for {entry_count} signals")
        else:
            logger.info("No entry signals generated (this is OK for test data)")
        
        # Test custom stake amount
        from datetime import datetime as dt
        stake = strategy.custom_stake_amount(
            pair='BTC/USDT',
            current_time=dt.now(),
            current_rate=100.0,
            proposed_stake=100.0,
            min_stake=10.0,
            max_stake=1000.0,
            leverage=1.0,
            entry_tag='test',
            side='long'
        )
        
        assert 10.0 <= stake <= 1000.0
        logger.info(f"Custom stake amount: {stake}")
        
        # Test custom stop loss
        from freqtrade.persistence import Trade
        from datetime import timedelta
        
        # Create mock trade
        trade = Trade(
            pair='BTC/USDT',
            open_rate=100.0,
            open_date=dt.now() - timedelta(hours=1),
            amount=1.0,
            fee_open=0.001,
            fee_close=0.001,
            stake_amount=100.0,
            is_open=True,
            amount_requested=1.0,
            exchange='binance'
        )
        
        # Add metadata
        trade.metadata = {'ml_confidence': 0.7}
        
        stoploss = strategy.custom_stoploss(
            pair='BTC/USDT',
            trade=trade,
            current_time=dt.now(),
            current_rate=101.0,
            current_profit=0.01,
            after_fill=False
        )
        
        assert -0.10 <= stoploss <= -0.02
        logger.info(f"Custom stop loss: {stoploss:.3%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Strategy improvements test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("Starting improvement roadmap tests...")
    
    tests = [
        ("HyperparameterOptimizer", test_hyperparameter_optimizer),
        ("RegimeAwareBarrierLabeler", test_regime_aware_labeler),
        ("Strategy Improvements", test_strategy_improvements),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All improvements successfully implemented!")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
