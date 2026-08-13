"""Shared pytest fixtures for the augur test suite."""

import pandas as pd
import pytest


@pytest.fixture
def valid_bars() -> pd.DataFrame:
    """A minimal, BarSchema-compliant OHLCV DataFrame."""
    index = pd.DatetimeIndex(
        ["2024-01-02 16:00", "2024-01-03 16:00"],
        tz="UTC",
        name="timestamp",
    ).astype("datetime64[ns, UTC]")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "adj_close": [101.0, 102.0],
            "volume": [1_000, 1_500],
            "stock_splits": [0.0, 0.0],
            "dividends": [0.25, 0],
        },
        index=index,
    )
