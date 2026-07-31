import pandas as pd
import yfinance as yf  # noqa: F401


def fetch_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for a single ticker and return a DataFrame
    validated against BarSchema.

    Composition (to be implemented):
    - calls _fetch_raw() to get unmodified yfinance output
    - calls _normalize() to convert it into BarSchema-compliant shape
    - should not contain any yfinance-specific logic itself — that
      belongs in the two helpers below, so this function stays a thin,
      testable composition point.
    """
    raise NotImplementedError


def _fetch_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Call yfinance and return its raw output, untouched.

    No normalization here. This function's only job is the network
    call, so it can be mocked out in tests — everything else should
    be testable without hitting the network.
    """
    raise NotImplementedError


def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Convert yfinance's raw DataFrame into BarSchema-compliant shape.

    Concerns to handle here (not yet implemented):
    - column renaming / selection to match BarSchema
    - timestamp handling: yfinance's index needs to end up tz-aware
      UTC, datetime64[ns, UTC] resolution specifically
    - session-close convention: this project hardcodes 16:00 ET for
      US equities as the bar timestamp
    - dtype coercion to match schema (float64 for prices, int64 for
      volume)
    """
    raise NotImplementedError
