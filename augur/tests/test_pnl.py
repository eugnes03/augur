"""Tests for augur.pnl."""

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from augur.pnl import strategy_returns


def test_strategy_returns_hand_computed() -> None:
    """Daily portfolio return is the weighted sum of forward returns across tickers."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="timestamp")
    weights = pd.DataFrame(
        {"A": [0.5, 1.0, 0.0], "B": [-0.5, 0.0, 1.0]},
        index=dates,
    )
    forward_returns = pd.DataFrame(
        {"A": [0.10, -0.05, 0.02], "B": [0.20, 0.03, -0.01]},
        index=dates,
    )

    result = strategy_returns(weights, forward_returns)

    expected = pd.Series([-0.05, -0.05, -0.01], index=dates)
    pd.testing.assert_series_equal(result, expected)


def test_strategy_returns_nan_return_on_held_ticker_makes_date_nan() -> None:
    """A NaN return for a nonzero-weight ticker means the day's total is unknown."""
    dates = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC", name="timestamp")
    weights = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=dates)
    forward_returns = pd.DataFrame({"A": [np.nan], "B": [0.01]}, index=dates)

    result = strategy_returns(weights, forward_returns)

    assert result.isna().all()


@given(st.lists(st.floats(allow_nan=True, allow_infinity=False, width=32), min_size=1, max_size=5))
def test_strategy_returns_is_zero_when_all_weights_are_zero(row: list[float]) -> None:
    """A date with all-zero weights has exactly zero strategy return, regardless of returns."""
    dates = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC", name="timestamp")
    columns = [f"ticker_{i}" for i in range(len(row))]
    weights = pd.DataFrame([[0.0] * len(row)], index=dates, columns=columns)
    forward_returns = pd.DataFrame([row], index=dates, columns=columns)

    result = strategy_returns(weights, forward_returns)

    assert result.iloc[0] == 0.0


def test_strategy_returns_nan_return_on_unheld_ticker_is_ignored() -> None:
    """A NaN return for a zero-weight ticker is harmless: it doesn't affect the total."""
    dates = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC", name="timestamp")
    weights = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=dates)
    forward_returns = pd.DataFrame({"A": [0.02], "B": [np.nan]}, index=dates)

    result = strategy_returns(weights, forward_returns)

    assert result.iloc[0] == 0.02


def test_strategy_returns_nan_weight_makes_date_nan() -> None:
    """A NaN weight (position unknown) makes the day's total unknown too."""
    dates = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC", name="timestamp")
    weights = pd.DataFrame({"A": [np.nan], "B": [0.5]}, index=dates)
    forward_returns = pd.DataFrame({"A": [0.01], "B": [0.02]}, index=dates)

    result = strategy_returns(weights, forward_returns)

    assert result.isna().all()
