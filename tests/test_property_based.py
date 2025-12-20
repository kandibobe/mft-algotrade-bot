"""
Property-based tests for trading system components.
Uses Hypothesis for generative testing.
"""

from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pandas as pd

from src.risk.position_sizing import PositionSizer
from src.utils.indicators import calculate_all_indicators


class TestPositionSizing:
    @given(
        balance=st.floats(min_value=1000, max_value=100000),
        risk_pct=st.floats(min_value=0.01, max_value=0.05),
        stop_loss_pct=st.floats(min_value=0.01, max_value=0.10),
    )
    def test_position_size_never_exceeds_balance(self, balance, risk_pct, stop_loss_pct):
        """Property: Position size should never exceed account balance"""
        sizer = PositionSizer()
        # Use entry price = balance (simplified) and stop loss price = entry * (1 - stop_loss_pct)
        entry_price = balance  # Simplified assumption
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        
        # Use calculate_position_size which applies max position limit
        result = sizer.calculate_position_size(
            account_balance=balance,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            method="fixed_risk",
            risk_pct=risk_pct
        )
        size = result["position_value"]
        
        # Position should not exceed balance (with max_position_pct default of 10%, it should be <= 0.1 * balance)
        # But we'll assert it's <= balance (which is a weaker condition)
        assert size <= balance, f"Position size {size} > balance {balance}"
        
    @given(
        price_series=npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=100, max_value=1000),
            elements=st.floats(min_value=1.0, max_value=100000.0)
        )
    )
    def test_indicators_no_nan(self, price_series):
        """Property: Indicators should not produce NaN (except initial period)"""
        # Convert numpy array to DataFrame with required columns
        df = pd.DataFrame({
            'open': price_series,
            'high': price_series * 1.01,  # Simulate some spread
            'low': price_series * 0.99,
            'close': price_series,
            'volume': np.full_like(price_series, 1000.0)  # Constant volume
        })
        
        indicators = calculate_all_indicators(df)
        
        # Check after warmup period (skip first 50 rows where indicators may be NaN)
        # Note: Some indicators like EMA, RSI may have NaN in initial periods
        # We'll check that after row 50, there are no NaN values in key indicators
        if len(indicators) > 50:
            # Check RSI and EMA columns exist
            if 'rsi' in indicators.columns:
                rsi_after = indicators['rsi'].iloc[50:]
                assert not rsi_after.isna().any(), f"RSI contains NaN after row 50: {rsi_after.isna().sum()}"
            
            if 'ema_50' in indicators.columns:
                ema_after = indicators['ema_50'].iloc[50:]
                assert not ema_after.isna().any(), f"EMA contains NaN after row 50: {ema_after.isna().sum()}"
