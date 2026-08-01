import pandas as pd
import yfinance as yf

from augur.schemas import BarSchema

_SESSION_CLOSE_TIME = "16:00"
_SESSION_CLOSE_TZ = "America/New_York"

_COLUMN_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def fetch_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for a single ticker and return a DataFrame
    validated against BarSchema.
    """
    raw = _fetch_raw(ticker, start, end)
    bars = _normalize(raw, ticker)
    return BarSchema.validate(bars)


def _fetch_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Call yfinance and return its raw output, untouched.

    No normalization here. This function's only job is the network
    call, so it can be mocked out in tests — everything else should
    be testable without hitting the network.

    auto_adjust=False is required to get the 'Adj Close' column that
    BarSchema needs; without it yfinance folds the adjustment into
    'Close' and drops the column entirely.
    """
    return pd.DataFrame(yf.download(ticker, start=start, end=end, auto_adjust=False))


def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Convert yfinance's raw DataFrame into BarSchema-compliant shape.
    """
    bars = _select_ticker_columns(raw, ticker)
    bars.index = _to_session_close_utc(bars.index)
    return _coerce_dtypes(bars)


def _select_ticker_columns(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's (Price, Ticker) MultiIndex columns to BarSchema names."""
    single_ticker = pd.DataFrame(raw.xs(ticker, axis=1, level="Ticker"))
    renamed = single_ticker.rename(columns=_COLUMN_RENAME)
    renamed.columns.name = None
    return renamed[list(_COLUMN_RENAME.values())]


def _to_session_close_utc(index: pd.Index) -> pd.DatetimeIndex:
    """
    Convert yfinance's midnight-of-trading-day index into this
    project's session-close convention: 16:00 ET, expressed as
    tz-aware UTC.
    """
    session_close = pd.DatetimeIndex(index).normalize() + pd.Timedelta(
        hours=int(_SESSION_CLOSE_TIME.split(":")[0])
    )
    utc_close = session_close.tz_localize(_SESSION_CLOSE_TZ).tz_convert("UTC")
    return pd.DatetimeIndex(utc_close.astype("datetime64[ns, UTC]"), name="timestamp")


def _coerce_dtypes(bars: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to BarSchema's expected dtypes (float64 prices, int64 volume)."""
    bars = bars.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "adj_close": "float64",
            "volume": "int64",
        }
    )
    return bars
