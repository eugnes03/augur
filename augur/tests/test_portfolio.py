"""Tests for augur.portfolio."""

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from augur.portfolio import apply_rebalance_frequency, equal_weight, rank_weight


@st.composite
def _momentum_and_selection(draw: st.DrawFn) -> tuple[pd.Series, int, int]:
    tickers = draw(
        st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=10, unique=True)
    )
    values = draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, width=32),
            min_size=len(tickers),
            max_size=len(tickers),
        )
    )
    n_long = draw(st.integers(min_value=1, max_value=len(tickers) - 1))
    n_short = draw(st.integers(min_value=1, max_value=len(tickers) - n_long))
    return pd.Series(values, index=tickers), n_long, n_short


def _momentum() -> pd.Series:
    return pd.Series(
        [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        index=["A", "B", "C", "D", "E", "F"],
    )


def test_equal_weight_known_expected_weights() -> None:
    """Top 2 long at +1/2 each, bottom 2 short at -1/2 each, rest zero, dollar-neutral."""
    result = equal_weight(_momentum(), n_long=2, n_short=2)

    expected = pd.Series(
        [0.5, 0.5, 0.0, 0.0, -0.5, -0.5],
        index=["A", "B", "C", "D", "E", "F"],
    )
    pd.testing.assert_series_equal(result, expected)
    assert result.sum() == pytest.approx(0.0)


def test_rank_weight_known_expected_weights() -> None:
    """Weight magnitude increases with rank extremity within each side, dollar-neutral."""
    result = rank_weight(_momentum(), n_long=2, n_short=2)

    expected = pd.Series(
        [2 / 3, 1 / 3, 0.0, 0.0, -1 / 3, -2 / 3],
        index=["A", "B", "C", "D", "E", "F"],
    )
    pd.testing.assert_series_equal(result, expected)
    assert result.sum() == pytest.approx(0.0)


def test_equal_weight_raises_when_selection_exceeds_universe() -> None:
    """n_long + n_short > available tickers must raise, not silently overlap."""
    with pytest.raises(ValueError, match="n_long \\+ n_short"):
        equal_weight(_momentum(), n_long=4, n_short=4)


def test_rank_weight_raises_when_selection_exceeds_universe() -> None:
    """n_long + n_short > available tickers must raise, not silently overlap."""
    with pytest.raises(ValueError, match="n_long \\+ n_short"):
        rank_weight(_momentum(), n_long=4, n_short=4)


def test_equal_weight_allows_selection_exactly_covering_universe() -> None:
    """n_long + n_short == available tickers is the boundary, not an error."""
    result = equal_weight(_momentum(), n_long=3, n_short=3)

    expected = pd.Series(
        [1 / 3, 1 / 3, 1 / 3, -1 / 3, -1 / 3, -1 / 3],
        index=["A", "B", "C", "D", "E", "F"],
    )
    pd.testing.assert_series_equal(result, expected)


def test_equal_weight_ties_broken_by_input_order() -> None:
    """A momentum tie at the long/short cutoff is broken by input order, deterministically."""
    tied = pd.Series([5.0, 3.0, 3.0, 0.0], index=["A", "B", "C", "D"])

    result = equal_weight(tied, n_long=2, n_short=1)

    assert result["B"] == pytest.approx(0.5)
    assert result["C"] == pytest.approx(0.0)


def test_equal_weight_ties_follow_input_order_when_reversed() -> None:
    """Reversing the tied tickers' input order flips which one is selected."""
    tied = pd.Series([5.0, 3.0, 3.0, 0.0], index=["A", "C", "B", "D"])

    result = equal_weight(tied, n_long=2, n_short=1)

    assert result["C"] == pytest.approx(0.5)
    assert result["B"] == pytest.approx(0.0)


def test_apply_rebalance_frequency_holds_prior_weight_between_rebalances() -> None:
    """Non-rebalance days carry forward the last rebalance day's weights."""
    dates = pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC", name="timestamp")
    weights = pd.DataFrame(
        {"A": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], "B": [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4]},
        index=dates,
    )

    result = apply_rebalance_frequency(weights, frequency=5)

    expected = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5], "B": [-1.0, -1.0, -1.0, -1.0, -1.0, -0.5, -0.5]},
        index=dates,
    )
    pd.testing.assert_frame_equal(result, expected)


@given(_momentum_and_selection())
def test_equal_weight_is_always_dollar_neutral_and_fully_invested(
    momentum_and_selection: tuple[pd.Series, int, int],
) -> None:
    """For any valid momentum series, weights sum to ~0 and abs(weights) sum to ~2."""
    momentum, n_long, n_short = momentum_and_selection

    weights = equal_weight(momentum, n_long=n_long, n_short=n_short)

    assert weights.sum() == pytest.approx(0.0, abs=1e-9)
    assert weights.abs().sum() == pytest.approx(2.0, abs=1e-9)
