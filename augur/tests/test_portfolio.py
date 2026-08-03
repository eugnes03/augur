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
