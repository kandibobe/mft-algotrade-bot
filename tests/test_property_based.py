import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

# Assuming your indicator functions are in src.utils.indicators
from src.utils.indicators import vwma, zscore_indicator


# Define a strategy for generating a DataFrame suitable for financial analysis
# It should have 'high', 'low', 'close', 'volume' columns
@st.composite
def financial_dataframe_strategy(draw):
    num_rows = draw(st.integers(min_value=20, max_value=200)) # Ensure enough data for indicators

    return draw(data_frames([
        column('open', dtype=float, elements=st.floats(min_value=90, max_value=110, allow_nan=False, allow_infinity=False)),
        column('high', dtype=float, elements=st.floats(min_value=100, max_value=120, allow_nan=False, allow_infinity=False)),
        column('low', dtype=float, elements=st.floats(min_value=80, max_value=100, allow_nan=False, allow_infinity=False)),
        column('close', dtype=float, elements=st.floats(min_value=90, max_value=110, allow_nan=False, allow_infinity=False)),
        column('volume', dtype=float, elements=st.floats(min_value=1000, max_value=100000, allow_nan=False, allow_infinity=False))
    ], index=st.integers(min_value=0, max_value=num_rows-1).map(lambda i: pd.RangeIndex(start=i, stop=i+num_rows, step=1))))

@given(df=financial_dataframe_strategy())
def test_zscore_properties(df):
    """
    Test properties of the zscore_indicator.
    - Output should have the same index as the input.
    - Output should not contain NaNs after the initial window.
    - Z-score of a constant series should be close to zero.
    """
    window_size = 20
    if len(df) < window_size:
        pytest.skip("DataFrame too small for window")

    result_df = zscore_indicator(df.copy(), window=window_size)

    pd.testing.assert_index_equal(df.index, result_df.index)

    # After the initial window, there should be no NaNs
    assert not result_df['zscore'][window_size - 1:].isnull().any()

    # Test with a constant series
    df['constant'] = 100.0
    result_constant = zscore_indicator(df[['constant']].rename(columns={'constant': 'close'}), window=window_size)
    # The z-score of a constant series should be 0 (or very close due to float precision)
    assert (result_constant['zscore'][window_size - 1:].abs() < 1e-9).all()

@given(df=financial_dataframe_strategy())
def test_vwma_properties(df):
    """
    Test properties of the vwma (Volume Weighted Moving Average).
    - Output should have the same index.
    - VWMA should be within the range of high and low prices over the window.
    - For constant volume, VWMA should be equal to a simple moving average (SMA).
    """
    window_size = 14
    if len(df) < window_size:
        pytest.skip("DataFrame too small for window")

    result_df = vwma(df.copy(), window=window_size)

    pd.testing.assert_index_equal(df.index, result_df.index)

    # After the initial window, VWMA should be a valid number
    assert not result_df[f'vwma_{window_size}'][window_size - 1:].isnull().any()

    # The VWMA should logically be between the min low and max high of the lookback period
    min_low = df['low'].rolling(window=window_size).min()
    max_high = df['high'].rolling(window=window_size).max()

    assert (result_df[f'vwma_{window_size}'][window_size - 1:] >= min_low[window_size - 1:]).all()
    assert (result_df[f'vwma_{window_size}'][window_size - 1:] <= max_high[window_size - 1:]).all()

    # If volume is constant, VWMA should equal SMA
    df_const_vol = df.copy()
    df_const_vol['volume'] = 1.0

    result_vwma_const = vwma(df_const_vol, window=window_size)
    sma = df_const_vol['close'].rolling(window=window_size).mean()

    pd.testing.assert_series_equal(
        result_vwma_const[f'vwma_{window_size}'][window_size - 1:],
        sma[window_size - 1:],
        check_names=False,
        atol=1e-9
    )
