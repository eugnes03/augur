"""Tests for augur.pnl."""

import pandas as pd

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
