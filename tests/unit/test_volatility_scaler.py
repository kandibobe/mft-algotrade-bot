"""
Unit tests for the VolatilityScaler.
"""

import pytest
from src.risk.volatility_scaler import VolatilityScaler, VolatilityScalerConfig

def test_volatility_scaler_initialization():
    """Test that the scaler can be initialized with default and custom configs."""
    scaler_default = VolatilityScaler()
    assert scaler_default.config.target_volatility == 0.15

    custom_config = VolatilityScalerConfig(target_volatility=0.20, min_multiplier=0.6)
    scaler_custom = VolatilityScaler(custom_config)
    assert scaler_custom.config.target_volatility == 0.20
    assert scaler_custom.config.min_multiplier == 0.6

def test_volatility_at_target():
    """When current volatility is at the target, the multiplier should be exactly in the middle of min/max."""
    config = VolatilityScalerConfig(min_multiplier=0.5, max_multiplier=1.5)
    scaler = VolatilityScaler(config)
    
    # Sigmoid of 0 is 0.5, so the result should be the midpoint.
    expected_multiplier = 0.5 + 0.5 * (1.5 - 0.5)
    assert scaler.calculate_multiplier(config.target_volatility) == pytest.approx(expected_multiplier)

def test_low_volatility():
    """When volatility is lower than target, multiplier should be high."""
    config = VolatilityScalerConfig(target_volatility=0.20, min_multiplier=0.5, max_multiplier=2.0)
    scaler = VolatilityScaler(config)
    
    low_vol = 0.10 # Half of target
    multiplier = scaler.calculate_multiplier(low_vol)
    assert multiplier > 1.0
    assert multiplier == pytest.approx(2.0, abs=0.02) # Should be close to max

def test_high_volatility():
    """When volatility is higher than target, multiplier should be low."""
    config = VolatilityScalerConfig(target_volatility=0.20, min_multiplier=0.5, max_multiplier=2.0)
    scaler = VolatilityScaler(config)
    
    high_vol = 0.40 # Double the target
    multiplier = scaler.calculate_multiplier(high_vol)
    assert multiplier < 1.0
    assert multiplier == pytest.approx(0.5, abs=0.01) # Should be close to min

def test_zero_volatility():
    """When volatility is zero, we should get the maximum multiplier."""
    config = VolatilityScalerConfig(min_multiplier=0.5, max_multiplier=2.0)
    scaler = VolatilityScaler(config)
    assert scaler.calculate_multiplier(0) == 2.0

def test_clipping():
    """Test that the multiplier is always clipped within the min/max range."""
    config = VolatilityScalerConfig(min_multiplier=0.7, max_multiplier=1.8)
    scaler = VolatilityScaler(config)

    # Extremely low vol
    assert scaler.calculate_multiplier(0.0001) <= config.max_multiplier
    # Extremely high vol
    assert scaler.calculate_multiplier(100.0) >= config.min_multiplier

def test_curve_aggressiveness():
    """Test the effect of the curve_aggressiveness parameter."""
    config_low_agg = VolatilityScalerConfig(target_volatility=0.2, curve_aggressiveness=1.0)
    scaler_low_agg = VolatilityScaler(config_low_agg)

    config_high_agg = VolatilityScalerConfig(target_volatility=0.2, curve_aggressiveness=20.0)
    scaler_high_agg = VolatilityScaler(config_high_agg)

    vol = 0.25 # Slightly above target
    
    multiplier_low_agg = scaler_low_agg.calculate_multiplier(vol)
    multiplier_high_agg = scaler_high_agg.calculate_multiplier(vol)
    
    # High aggressiveness should result in a faster drop, so a lower multiplier
    assert multiplier_high_agg < multiplier_low_agg
    
    # Check they are both below the midpoint
    midpoint = config_low_agg.min_multiplier + 0.5 * (config_low_agg.max_multiplier - config_low_agg.min_multiplier)
    assert multiplier_low_agg < midpoint
    assert multiplier_high_agg < midpoint
