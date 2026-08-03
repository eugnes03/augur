"""Tests for augur.signals."""

import numpy as np
import pandas as pd

from augur.signals import trailing_momentum


def _panel(prices: dict[str, list[float]], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    frames = {
        ticker: pd.DataFrame({"adj_close": values}, index=timestamps)
        for ticker, values in prices.items()
    }
    stacked = pd.concat(frames, names=["ticker", "timestamp"])
    return stacked.swaplevel().sort_index()


def test_trailing_momentum_no_cross_ticker_leakage() -> None:
    """Each ticker's momentum is computed only from its own price history."""
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC", name="timestamp")
    panel = _panel(
        {
            "A": [100.0, 110.0, 121.0, 100.0],
            "B": [200.0, 100.0, 50.0, 200.0],
        },
        timestamps,
    )

    result = trailing_momentum(panel, lookback=1)

    expected_a = [np.nan, np.log(1.1), np.log(1.1), np.log(100 / 121)]
    expected_b = [np.nan, np.log(0.5), np.log(0.5), np.log(4.0)]
    for t, ea, eb in zip(timestamps, expected_a, expected_b, strict=True):
        np.testing.assert_allclose(result[(t, "A")], ea, equal_nan=True)
        np.testing.assert_allclose(result[(t, "B")], eb, equal_nan=True)
