import pandas as pd
import pandera as pa
from pandera.typing import Index, Series


def _ohlc_consistent(df: pd.DataFrame) -> pd.Series:
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
    Pandera schema for a single-ticker OHLCV bar DataFrame.

    Shape:
    - DatetimeIndex named 'timestamp', dtype datetime64[ns, UTC]
    - columns: open, high, low, close, adj_close (float64), volume (int64)
    - strict=True, coerce=False
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
    def ohlc_consistent(cls, df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]  # noqa: N805
        return _ohlc_consistent(df)

    @pa.dataframe_check
    def index_sorted(cls, df: pd.DataFrame) -> bool:  # type: ignore[misc]  # noqa: N805
        return bool(df.index.is_monotonic_increasing)


class PanelBarSchema(pa.DataFrameModel):
    """
    Pandera schema for a multi-ticker OHLCV bar panel in long format, as
    produced by ingest.stack_universe_bars.

    Shape:
    - MultiIndex (timestamp, ticker): timestamp is datetime64[ns, UTC],
      ticker is a yfinance-compatible string; one row per pair
    - columns: open, high, low, close, adj_close (float64), volume (int64)
    - strict=True, coerce=False
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
    def ohlc_consistent(cls, df: pd.DataFrame) -> pd.Series:  # type: ignore[misc]  # noqa: N805
        return _ohlc_consistent(df)

    @pa.dataframe_check
    def timestamp_sorted(cls, df: pd.DataFrame) -> bool:  # type: ignore[misc]  # noqa: N805
        """Rows must be sorted by the outer (timestamp) index level."""
        return bool(df.index.get_level_values("timestamp").is_monotonic_increasing)
