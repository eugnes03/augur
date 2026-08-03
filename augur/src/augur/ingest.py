import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from augur.schemas import BarSchema, PanelBarSchema, ohlc_consistent
from augur.universe import get_universe

logger = logging.getLogger(__name__)

_SESSION_CLOSE_TIME = "16:00"
_SESSION_CLOSE_TZ = "America/New_York"

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "bars" / "daily"

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
    Fetch daily OHLCV bars for a ticker, validated against BarSchema.
    Rows with inconsistent OHLC values are dropped, not the whole ticker.
    """
    raw = _fetch_raw(ticker, start, end)
    bars = _normalize(raw, ticker)
    bars = _drop_inconsistent_rows(bars, ticker)
    return BarSchema.validate(bars)


def fetch_universe_bars(
    start: str, end: str, tickers: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """
    Fetch validated bars for every ticker in the universe (or `tickers`).
    A ticker that fails to fetch, or ends up with zero rows, is logged
    and skipped rather than failing the whole batch.
    """
    tickers = tickers if tickers is not None else get_universe()
    bars: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            ticker_bars = fetch_bars(ticker, start, end)
        except Exception:
            logger.warning("Failed to fetch bars for %s", ticker, exc_info=True)
            continue
        if ticker_bars.empty:
            logger.warning("Fetched zero rows for %s, skipping", ticker)
            continue
        bars[ticker] = ticker_bars
    return bars


def stack_universe_bars(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reshape {ticker: bars} into one long, PanelBarSchema-validated DataFrame."""
    stacked = pd.concat(bars, names=["ticker", "timestamp"])
    stacked = stacked.swaplevel().sort_index()
    return PanelBarSchema.validate(stacked)


def _fetch_raw(
    ticker: str, start: str, end: str, cache_dir: Path = _CACHE_DIR
) -> pd.DataFrame:
    """
    Fetch raw yfinance OHLCV, caching to `cache_dir` as parquet (no
    TTL/invalidation). auto_adjust=False keeps 'Adj Close', which
    yfinance otherwise folds into 'Close' and drops.
    """
    cache_path = cache_dir / f"{ticker}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.DataFrame(pd.read_parquet(cache_path))

    raw = pd.DataFrame(yf.download(ticker, start=start, end=end, auto_adjust=False))
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_path)
    return raw


def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Convert yfinance's raw DataFrame into BarSchema-compliant shape.
    """
    bars = _select_ticker_columns(raw, ticker)
    bars.index = _to_session_close_utc(bars.index)
    return _coerce_dtypes(bars)


def _drop_inconsistent_rows(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Drop rows with internally inconsistent OHLC values. Observed cases
    were isolated vendor glitches, not splits, so only the bad row is
    dropped rather than the whole ticker.
    """
    consistent = ohlc_consistent(bars)
    n_dropped = int((~consistent).sum())
    if n_dropped:
        logger.warning(
            "Dropping %d/%d rows for %s with inconsistent OHLC values: %s",
            n_dropped,
            len(bars),
            ticker,
            list(bars.index[~consistent]),
        )
    return bars[consistent]


def _select_ticker_columns(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's (Price, Ticker) MultiIndex columns to BarSchema names."""
    single_ticker = pd.DataFrame(raw.xs(ticker, axis=1, level="Ticker"))
    renamed = single_ticker.rename(columns=_COLUMN_RENAME)
    renamed.columns.name = None
    return renamed[list(_COLUMN_RENAME.values())]


def _to_session_close_utc(index: pd.Index) -> pd.DatetimeIndex:
    """Convert yfinance's midnight index to this project's 16:00 ET session close, as UTC."""
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
