"""Tests for augur.costs."""

import pandas as pd
import pytest

from augur.costs import net_of_costs


def _weights() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"A": [0.5, 0.5, 1.0], "B": [-0.5, -0.5, 0.0], "C": [0.0, 0.0, -1.0]},
        index=dates,
    )


def test_net_of_costs_hand_computed() -> None:
    """100bps cost on a unit of turnover is a flat 1% drag; a no-turnover date is untouched."""
    weights = _weights()
    returns = pd.Series([0.01, 0.02, -0.01], index=weights.index)

    result = net_of_costs(returns, weights, cost_bps=100.0)

    expected = pd.Series([0.01, 0.02, -0.02], index=weights.index)
    pd.testing.assert_series_equal(result, expected, check_names=False, check_freq=False)


def test_net_of_costs_zero_bps_is_a_no_op() -> None:
    """A 0bps cost should leave returns untouched."""
    weights = _weights()
    returns = pd.Series([0.01, 0.02, -0.01], index=weights.index)

    result = net_of_costs(returns, weights, cost_bps=0.0)

    pd.testing.assert_series_equal(result, returns, check_freq=False)


def test_net_of_costs_drops_dates_missing_from_returns() -> None:
    """A weights date outside returns' index shouldn't inject a spurious cost-only row."""
    weights = _weights()
    returns = pd.Series([0.01, 0.02], index=weights.index[:2])

    result = net_of_costs(returns, weights, cost_bps=100.0)

    assert list(result.index) == list(weights.index[:2])


def test_net_of_costs_higher_bps_means_lower_returns() -> None:
    """A more expensive cost assumption should only ever reduce (or match) net returns."""
    weights = _weights()
    returns = pd.Series([0.01, 0.02, -0.01], index=weights.index)

    cheap = net_of_costs(returns, weights, cost_bps=5.0)
    expensive = net_of_costs(returns, weights, cost_bps=50.0)

    assert (expensive <= cheap + 1e-12).all()
    assert expensive.iloc[-1] < cheap.iloc[-1]  # the turnover date must actually differ


def test_net_of_costs_matches_pytest_approx_on_turnover_day() -> None:
    """Sanity check against pytest.approx directly, not just series equality."""
    weights = _weights()
    returns = pd.Series([0.01, 0.02, -0.01], index=weights.index)

    result = net_of_costs(returns, weights, cost_bps=10.0)

    assert result.iloc[2] == pytest.approx(-0.01 - 0.0010)
