"""Tests for augur.portfolio."""

import pandas as pd
import pytest

from augur.portfolio import equal_weight, rank_weight


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
