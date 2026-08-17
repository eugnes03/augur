"""Tests for augur.risk."""

import logging

import numpy as np
import pandas as pd
import pytest

from augur.risk import statistical_factors


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC", name="timestamp")


def _correlated_panel() -> pd.DataFrame:
    """
    A and B are near-identical linear combinations of a shared underlying
    series (small independent noise added to each), so they should load
    heavily on the same factor. C is fully independent noise.
    """
    rng = np.random.default_rng(0)
    n = 200
    common = rng.normal(size=n)
    noise = rng.normal(scale=0.01, size=(n, 3))

    return pd.DataFrame(
        {
            "A": common + noise[:, 0],
            "B": 2.0 * common + noise[:, 1],
            "C": noise[:, 2],
        },
        index=_dates(n),
    )


def test_correlated_tickers_load_on_the_same_dominant_factor() -> None:
    """A and B (built from a shared series) should dominate factor_0 with matching sign."""
    loadings, _, explained_variance_ratio = statistical_factors(_correlated_panel(), n_factors=2)
    factor_0 = loadings["factor_0"].astype(float)

    assert np.sign(factor_0["A"]) == np.sign(factor_0["B"])
    assert abs(factor_0["A"]) > abs(factor_0["C"])
    assert abs(factor_0["B"]) > abs(factor_0["C"])
    assert explained_variance_ratio["factor_0"] > explained_variance_ratio["factor_1"]


def test_explained_variance_ratio_sums_to_at_most_one_and_is_descending() -> None:
    """Explained variance ratios never exceed total variance and are sorted largest-first."""
    _, _, explained_variance_ratio = statistical_factors(_correlated_panel(), n_factors=3)

    assert explained_variance_ratio.sum() <= 1.0 + 1e-9
    assert list(explained_variance_ratio) == sorted(explained_variance_ratio, reverse=True)


def test_sign_normalization_makes_largest_loading_positive() -> None:
    """Each factor's largest-magnitude loading should be positive, by construction."""
    loadings, _, _ = statistical_factors(_correlated_panel(), n_factors=2)

    for column in loadings.columns:
        factor_loadings = loadings[column].astype(float)
        largest_magnitude_ticker = factor_loadings.abs().idxmax()
        assert factor_loadings[largest_magnitude_ticker] > 0


def test_ticker_with_nan_in_window_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A ticker with any NaN in the window is excluded from loadings and named in a warning."""
    panel = _correlated_panel()
    panel.loc[panel.index[5], "C"] = np.nan

    with caplog.at_level(logging.WARNING, logger="augur.risk"):
        loadings, factor_returns, _ = statistical_factors(panel, n_factors=2)

    assert "C" not in loadings.index
    assert "C" not in factor_returns.columns
    assert any("C" in record.message for record in caplog.records)


def test_factor_returns_indexed_by_date_and_loadings_by_ticker() -> None:
    """Output shapes: loadings is (ticker x factor), factor_returns is (date x factor)."""
    panel = _correlated_panel()

    loadings, factor_returns, explained_variance_ratio = statistical_factors(panel, n_factors=2)

    assert list(loadings.index) == list(panel.columns)
    assert list(loadings.columns) == ["factor_0", "factor_1"]
    assert list(factor_returns.index) == list(panel.index)
    assert list(factor_returns.columns) == ["factor_0", "factor_1"]
    assert len(explained_variance_ratio) == 2
