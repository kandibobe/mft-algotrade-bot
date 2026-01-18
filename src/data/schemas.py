import pandera as pa
from pandera.typing import Series


class OHLCVSchema(pa.DataFrameModel):
    """
    Schema for validating OHLCV data.
    Ensures that the essential columns are present and have the correct data types.
    """
    open: Series[float] = pa.Field(ge=0)
    high: Series[float] = pa.Field(ge=0)
    low: Series[float] = pa.Field(ge=0)
    close: Series[float] = pa.Field(ge=0)
    volume: Series[float] = pa.Field(ge=0)

    @pa.check('high')
    def high_ge_low(cls, high: Series[float], *, low: Series[float]) -> bool:
        """Check that the high is always greater than or equal to the low."""
        return (high >= low).all()

    class Config:
        strict = False  # Allow other columns to be present
        coerce = True   # Coerce data types if possible
