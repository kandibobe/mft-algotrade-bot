"""
Reality Check (Drift Analysis)
==============================

Verifies that Backtest logic matches Live execution logic.
Compares simulated signals on past 24h data vs actual recorded signals in DB.
"""

import logging
import pandas as pd
import glob
import os
import joblib
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy.exc import OperationalError

from src.config.manager import ConfigurationManager
from src.data.loader import get_ohlcv
from src.database.db_manager import DatabaseManager
from src.database.models import SignalRecord
from src.strategies.core_logic import StoicLogic
from src.ml.model_loader import get_production_model

logger = logging.getLogger(__name__)

def run_reality_check(pairs: List[str] = None, hours: int = 24):
    """
    Run the reality check simulation.
    """
    config = ConfigurationManager.get_config()
    pairs = pairs or config.pairs
    
    logger.info("="*70)
    logger.info("🕵️ REALITY CHECK (DRIFT ANALYSIS)")
    logger.info(f"Checking last {hours} hours (Data-Driven) for {pairs}")
    logger.info("="*70)
    
    for pair in pairs:
        logger.info(f"\n🔍 Analyzing {pair}...")
        
        # 1. Load OHLCV Data (Load enough history)
        # We load plenty of history to ensure we catch the end of data and have enough for indicators
        lookback_days = (hours / 24) + 30 # +30 days buffer to be safe
        load_start_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        try:
            ohlcv = get_ohlcv(
                symbol=pair,
                timeframe="5m", # Hardcoded for now, should come from config
                start=load_start_date
            )
        except Exception as e:
            logger.error(f"Failed to load data for {pair}: {e}")
            continue
            
        if ohlcv.empty:
            logger.warning(f"No data for {pair}")
            continue
            
        # 2. Determine Time Window from Data
        end_date = ohlcv.index[-1]
        start_date = end_date - timedelta(hours=hours)
        
        logger.info(f"Analyzing Data Window: {start_date} to {end_date}")
        
        # 3. Fetch Live Signals from DB for this specific window
        live_data = []
        try:
            with DatabaseManager.session() as session:
                live_signals_query = session.query(SignalRecord).filter(
                    SignalRecord.timestamp >= start_date,
                    SignalRecord.timestamp <= end_date,
                    SignalRecord.symbol == pair
                ).all()
                
                # Convert to DataFrame
                for s in live_signals_query:
                    live_data.append({
                        'timestamp': s.timestamp,
                        'symbol': s.symbol,
                        'live_signal': s.signal_type,
                        'live_confidence': s.model_confidence,
                        'live_regime': s.regime
                    })
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            logger.warning("Proceeding with empty live data due to DB connection issue.")
        except Exception as e:
            logger.error(f"Error fetching live signals: {e}")
        
        if not live_data:
            logger.warning(f"Running Backtest Simulation only (No matching live signals found for {pair} in this window)...")
            live_df = pd.DataFrame(columns=['timestamp', 'symbol', 'live_signal', 'live_confidence', 'live_regime'])
        else:
            live_df = pd.DataFrame(live_data)
            # Round timestamp to nearest timeframe (assuming 5m) to match OHLCV
            live_df['timestamp'] = live_df['timestamp'].dt.round('5min')

        # 4. Load Model & Generate Features
        model, engineer, feature_names = get_production_model(pair)
        
        if model is None:
            # Fallback: Look for latest local .joblib model
            try:
                safe_pair = pair.replace('/', '_')
                search_pattern = f"user_data/models/*{safe_pair}*.joblib"
                files = glob.glob(search_pattern)
                
                if files:
                    latest_file = max(files, key=os.path.getctime)
                    logger.warning(f"⚠️ Using latest local model: {latest_file}")
                    
                    loaded_model = joblib.load(latest_file)
                    
                    # Handle different save formats
                    if isinstance(loaded_model, dict):
                        model = loaded_model.get('model')
                        # Try to get engineer if available, otherwise might need to create one
                        engineer = loaded_model.get('engineer')
                        feature_names = loaded_model.get('feature_names', [])
                    else:
                        # Assume it's the model object itself
                        model = loaded_model
                        # We still need an engineer and feature_names
                        # If we can't find them, this might fail later, but we try our best.
                        from src.ml.training.feature_engineering import FeatureEngineer
                        engineer = FeatureEngineer() 
                        # Try to find scaler
                        scaler_path = latest_file.replace('.joblib', '_scaler.joblib')
                        if os.path.exists(scaler_path):
                             engineer.load_scaler(scaler_path)
                             logger.info(f"Loaded scaler from {scaler_path}")
                        else:
                             logger.warning("No scaler found for fallback model. Fitting scaler on current data to allow execution (results may be inaccurate).")
                             engineer.needs_fitting = True
                        
                        # Use default features if not known? Or hope the model has feature_name_ property?
                        if hasattr(model, "feature_name_"):
                            feature_names = model.feature_name_
                        else:
                            # Fallback to hardcoded or empty? empty might crash.
                            # Let's hope for the best or use all numeric columns
                            feature_names = [] 
                            
            except Exception as e:
                logger.error(f"Fallback model loading failed: {e}")

        if model is None or engineer is None:
            logger.warning(f"Skipping {pair} - No production model found (and fallback failed).")
            continue
            
        # Generate Features
        try:
            if getattr(engineer, 'needs_fitting', False):
                # Manual fitting to avoid dropping low-variance features that the model might expect
                logger.info("Manually fitting scaler (preserving low-variance features)...")
                
                # 1. Prepare data
                features = engineer.prepare_data(ohlcv.copy())
                
                # 2. Validate without dropping low variance
                engineer.validate_features(features, fix_issues=True, raise_on_error=False, drop_low_variance=False)
                
                # 3. Fit scaler
                engineer._fit_scale_features(features)
                engineer._is_fitted = True
                
                # 4. Ensure all expected features exist (fill missing with 0)
                if feature_names:
                    for col in feature_names:
                        if col not in features.columns:
                            features[col] = 0.0
                            logger.warning(f"Feature {col} missing after engineering. Filled with 0.")
            else:
                features = engineer.transform(ohlcv.copy())
            
            # Predict
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features[feature_names])
                ohlcv['ml_prediction'] = probs[:, 1]
            else:
                ohlcv['ml_prediction'] = model.predict(features[feature_names])
                
        except Exception as e:
            logger.error(f"Error during ML inference for {pair}: {e}")
            continue
            
        # 5. Run Strategy Logic
        # Populate Indicators
        df = StoicLogic.populate_indicators(ohlcv)
        
        # Generate Signals
        df = StoicLogic.populate_entry_exit_signals(df)
        
        # Map Backtest Signals to 'buy', 'sell', 'hold'
        def map_signal(row):
            if row['enter_long'] == 1:
                return 'buy'
            elif row['exit_long'] == 1:
                return 'sell'
            else:
                return 'hold'
        
        df['sim_signal'] = df.apply(map_signal, axis=1)
        
        # 6. Compare
        # Filter DataFrame to analysis window
        df_analysis = df[(df.index >= start_date) & (df.index <= end_date)].copy()
        
        # Merge with Live Data
        # We perform left merge on simulated data (since we want to see how simulations match reality)
        # Actually we want to compare overlap. 
        
        pair_live = live_df # live_df is already filtered for this pair
        
        merged = pd.merge(
            df_analysis[['sim_signal', 'ml_prediction', 'close']],
            pair_live,
            left_index=True,
            right_on='timestamp',
            how='left'
        )
        
        # Fill missing live signals with 'hold' (assuming if not logged, it was hold/nothing)
        merged['live_signal'] = merged['live_signal'].fillna('hold')
        
        # Calculate Match Rate
        total_candles = len(merged)
        matches = merged[merged['sim_signal'] == merged['live_signal']]
        match_count = len(matches)
        match_rate = (match_count / total_candles) * 100 if total_candles > 0 else 0
        
        # Identify Discrepancies
        discrepancies = merged[merged['sim_signal'] != merged['live_signal']]
        
        # 7. Report
        print(f"\n=== REALITY CHECK REPORT: {pair} ===")
        print(f"Time Range: {start_date} to {end_date}")
        print(f"Candles Analyzed: {total_candles}")
        print(f"Match Rate: {match_rate:.2f}%")
        print(f"Discrepancies: {len(discrepancies)}")
        
        if not discrepancies.empty:
            print("-" * 60)
            print(f"{'Timestamp':<25} | {'Live':<6} | {'Sim':<6} | {'Sim Conf':<8}")
            print("-" * 60)
            for idx, row in discrepancies.head(10).iterrows(): # Show top 10
                ts_str = str(row['timestamp'])
                print(f"{ts_str:<25} | {row['live_signal']:<6} | {row['sim_signal']:<6} | {row['ml_prediction']:.4f}")
            if len(discrepancies) > 10:
                print(f"... and {len(discrepancies) - 10} more.")
            print("-" * 60)

if __name__ == "__main__":
    run_reality_check()
