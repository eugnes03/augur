import pandas as pd
import pandera as pa
from pandera.typing import Index, Series


def ohlc_consistent(df: pd.DataFrame) -> pd.Series:
    """low must be the min and high must be the max of each bar."""
    return (
        (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["low"] <= df["high"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
    )


class BarSchema(pa.DataFrameModel):
    """
    Single-ticker OHLCV bars: DatetimeIndex 'timestamp' (UTC), columns
    open/high/low/close/adj_close (float64) and volume (int64).
    """

    open: Series[float] = pa.Field(gt=0)
    high: Series[float] = pa.Field(gt=0)
    low: Series[float] = pa.Field(gt=0)
    close: Series[float] = pa.Field(gt=0)
    adj_close: Series[float] = pa.Field(gt=0)
    volume: Series[int] = pa.Field(ge=0)

    timestamp: Index[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "UTC"},
        check_name=True,
        unique=True,
    )

    class Config:
        strict = True
        coerce = False

    @pa.dataframe_check
    def ohlc_consistent_check(cls, df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]  # noqa: N805
        return ohlc_consistent(df)

    @pa.dataframe_check
    def index_sorted(cls, df: pd.DataFrame) -> bool:  # type: ignore[misc]  # noqa: N805
        return bool(df.index.is_monotonic_increasing)


class PanelBarSchema(pa.DataFrameModel):
    """
    Multi-ticker OHLCV bar panel, as produced by stack_universe_bars:
    MultiIndex (timestamp, ticker), one row per pair, same OHLCV columns
    as BarSchema.
    """

    open: Series[float] = pa.Field(gt=0)
    high: Series[float] = pa.Field(gt=0)
    low: Series[float] = pa.Field(gt=0)
    close: Series[float] = pa.Field(gt=0)
    adj_close: Series[float] = pa.Field(gt=0)
    volume: Series[int] = pa.Field(ge=0)

    timestamp: Index[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "UTC"},
        check_name=True,
    )
    ticker: Index[str] = pa.Field(check_name=True)

    class Config:
        strict = True
        coerce = False
        multiindex_unique = ["timestamp", "ticker"]

    @pa.dataframe_check
    def ohlc_consistent_check(cls, df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]  # noqa: N805
        return ohlc_consistent(df)

    @pa.dataframe_check
    def timestamp_sorted(cls, df: pd.DataFrame) -> bool:  # type: ignore[misc]  # noqa: N805
        """Rows must be sorted by the outer (timestamp) index level."""
        return bool(df.index.get_level_values("timestamp").is_monotonic_increasing)
