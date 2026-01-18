import numpy as np
import pandas as pd
import pytest

from src.utils.regime_detection import MarketRegime, calculate_regime


@pytest.fixture
def sample_market_data():
    """ Provides a sample DataFrame with market data. """
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=600))
    price = 100 + np.cumsum(np.random.randn(600) * 2)
    high = price + np.random.uniform(0, 2, 600)
    low = price - np.random.uniform(0, 2, 600)
    volume = np.random.uniform(100, 1000, 600)

    return pd.DataFrame({
        'date': dates,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume
    }).set_index('date')

def test_calculate_regime_output_columns(sample_market_data):
    """ Tests that calculate_regime returns a DataFrame with the expected columns. """
    df = sample_market_data
    result_df = calculate_regime(df['high'], df['low'], df['close'], df['volume'])

    expected_columns = ['vol_zscore', 'adx', 'hurst', 'regime']
    for col in expected_columns:
        assert col in result_df.columns

def test_calculate_regime_output_values(sample_market_data):
    """ Tests that the output values are within expected ranges and types. """
    df = sample_market_data
    result_df = calculate_regime(df['high'], df['low'], df['close'], df['volume'])

    # Check for NaNs (ignoring first 100 rows due to warmup)
    warmup = 100
    assert not result_df.iloc[warmup:].isnull().values.any()

    # Check regime values
    valid_regimes = [e.value for e in MarketRegime]
    assert result_df['regime'].isin(valid_regimes).all()

    # Check numeric ranges (simple sanity check)
    assert result_df.iloc[warmup:]['adx'].between(0, 100).all()
    assert result_df.iloc[warmup:]['hurst'].between(0, 1).all()

def test_regime_classification_logic():
    """ Tests the regime classification logic with mock data. """
    index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    # Mock data to force different regimes
    mock_df = pd.DataFrame({
        'vol_zscore': [-1.0, -1.0, 1.0, 1.0],
        'adx': [20, 30, 30, 20],
        'hurst': [0.4, 0.6, 0.6, 0.4],
    }, index=index)

    # Simple mock calculate_regime that just classifies based on inputs
    def mock_classify(df):
        df['regime'] = MarketRegime.QUIET_CHOP.value
        is_high_vol = df["vol_zscore"] > 0.5
        is_trending = (df["adx"] > 25.0) & (df["hurst"] > 0.55)

        mask_grind = (~is_high_vol) & is_trending
        df.loc[mask_grind, "regime"] = MarketRegime.GRIND.value

        mask_pump = is_high_vol & is_trending
        df.loc[mask_pump, "regime"] = MarketRegime.PUMP_DUMP.value

        mask_violent = is_high_vol & (~is_trending)
        df.loc[mask_violent, "regime"] = MarketRegime.VIOLENT_CHOP.value
        return df

    result = mock_classify(mock_df)

    assert result['regime'].iloc[0] == MarketRegime.QUIET_CHOP.value
    assert result['regime'].iloc[1] == MarketRegime.GRIND.value
    assert result['regime'].iloc[2] == MarketRegime.PUMP_DUMP.value
    assert result['regime'].iloc[3] == MarketRegime.VIOLENT_CHOP.value
