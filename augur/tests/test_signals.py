"""Tests for augur.signals."""

import numpy as np
import pandas as pd

from augur.signals import combine_signals, short_term_reversal, trailing_momentum


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


def test_trailing_momentum_nan_when_lookback_exceeds_history() -> None:
    """A lookback longer than the available history should be NaN, not raise or wrap."""
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="timestamp")
    panel = _panel({"A": [100.0, 110.0, 121.0]}, timestamps)

    result = trailing_momentum(panel, lookback=5)

    assert result.isna().all()


def test_short_term_reversal_correct_sign() -> None:
    """Reversal is the negative of momentum on the same price history."""
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC", name="timestamp")
    panel = _panel(
        {
            "A": [100.0, 110.0, 121.0, 100.0],
            "B": [200.0, 100.0, 50.0, 200.0],
        },
        timestamps,
    )

    result = short_term_reversal(panel, lookback=1)

    expected_a = [np.nan, -np.log(1.1), -np.log(1.1), -np.log(100 / 121)]
    expected_b = [np.nan, -np.log(0.5), -np.log(0.5), -np.log(4.0)]
    for t, ea, eb in zip(timestamps, expected_a, expected_b, strict=True):
        np.testing.assert_allclose(result[(t, "A")], ea, equal_nan=True)
        np.testing.assert_allclose(result[(t, "B")], eb, equal_nan=True)


def test_combine_signals_averages_cross_sectional_zscores() -> None:
    """Combined signal is the per-date average of each input signal's z-score."""
    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    index = pd.MultiIndex.from_tuples(
        [(timestamp, "A"), (timestamp, "B"), (timestamp, "C")],
        names=["timestamp", "ticker"],
    )
    signal_a = pd.Series([1.0, 2.0, 3.0], index=index)
    signal_b = pd.Series([10.0, 10.0, 40.0], index=index)

    result = combine_signals({"a": signal_a, "b": signal_b})

    expected = pd.Series([-0.78868, -0.28868, 1.07735], index=index)
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-4)


def test_combine_signals_large_raw_scale_does_not_dominate() -> None:
    """Z-scoring first means rescaling one signal's raw magnitude doesn't change the combination."""
    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    index = pd.MultiIndex.from_tuples(
        [(timestamp, "A"), (timestamp, "B"), (timestamp, "C")],
        names=["timestamp", "ticker"],
    )
    signal_a = pd.Series([1.0, 2.0, 3.0], index=index)
    signal_b = pd.Series([10.0, 10.0, 40.0], index=index)

    result_natural_scale = combine_signals({"a": signal_a, "b": signal_b})
    result_10x_scale = combine_signals({"a": signal_a, "b": signal_b * 10.0})

    np.testing.assert_allclose(result_natural_scale.to_numpy(), result_10x_scale.to_numpy())


def test_combine_signals_respects_explicit_weights() -> None:
    """Weighting one signal at 1.0 and the other at 0.0 reduces to that signal's z-score alone."""
    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    index = pd.MultiIndex.from_tuples(
        [(timestamp, "A"), (timestamp, "B"), (timestamp, "C")],
        names=["timestamp", "ticker"],
    )
    signal_a = pd.Series([1.0, 2.0, 3.0], index=index)
    signal_b = pd.Series([10.0, 10.0, 40.0], index=index)

    result = combine_signals({"a": signal_a, "b": signal_b}, weights={"a": 1.0, "b": 0.0})

    expected = pd.Series([-1.0, 0.0, 1.0], index=index)
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-4)
