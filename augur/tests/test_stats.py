"""Tests for augur.stats."""

import numpy as np
import pandas as pd
import pytest

from augur.stats import (
    annualized_return,
    annualized_volatility,
    gross_exposure,
    max_drawdown,
    net_exposure,
    sharpe_ratio,
    total_return,
    turnover,
)


def test_total_return_two_ten_percent_days() -> None:
    """Two independent +10% log-return days compound to +21% total simple return."""
    returns = pd.Series([np.log(1.1), np.log(1.1)])
    assert total_return(returns) == pytest.approx(0.21)


def test_annualized_return_matches_total_return_at_period_boundary() -> None:
    """With periods_per_year == len(returns), annualized return equals total return."""
    returns = pd.Series([np.log(1.1), np.log(1.1)])
    assert annualized_return(returns, periods_per_year=2) == pytest.approx(0.21)


def test_annualized_volatility_hand_computed() -> None:
    """std([0.01, -0.01], ddof=1) == 0.01*sqrt(2); annualized by sqrt(periods_per_year)."""
    returns = pd.Series([0.01, -0.01])
    expected = 0.01 * np.sqrt(2) * np.sqrt(4)
    assert annualized_volatility(returns, periods_per_year=4) == pytest.approx(expected)


def test_sharpe_ratio_hand_computed() -> None:
    """mean=0.02, std(ddof=1)=0.01 for [0.01, 0.03, 0.02]; periods_per_year=1 avoids scaling."""
    returns = pd.Series([0.01, 0.03, 0.02])
    assert sharpe_ratio(returns, periods_per_year=1) == pytest.approx(2.0)


def test_max_drawdown_hand_computed() -> None:
    """+10% then -20% then +5%: trough is 0.88 vs. peak 1.1, a -20% drawdown."""
    returns = pd.Series([np.log(1.1), np.log(0.8), np.log(1.05)])
    assert max_drawdown(returns) == pytest.approx(-0.2)


def test_max_drawdown_monotonic_gains_is_zero() -> None:
    """A series that only ever rises has no drawdown."""
    returns = pd.Series([np.log(1.05), np.log(1.02), np.log(1.01)])
    assert max_drawdown(returns) == pytest.approx(0.0)


def _weights() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {"A": [0.5, 0.5, 1.0], "B": [-0.5, -0.5, 0.0], "C": [0.0, 0.0, -1.0]},
        index=dates,
    )


def test_turnover_hand_computed() -> None:
    """Turnover is half the gross weight change; a date with no weight change is zero."""
    result = turnover(_weights())

    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(0.5 * (0.5 + 0.5 + 1.0))


def test_gross_exposure_hand_computed() -> None:
    """Gross exposure is the sum of absolute weights per date."""
    result = gross_exposure(_weights())

    expected = pd.Series([1.0, 1.0, 2.0], index=_weights().index)
    pd.testing.assert_series_equal(result, expected)


def test_net_exposure_hand_computed() -> None:
    """Net exposure is the sum of signed weights per date."""
    result = net_exposure(_weights())

    expected = pd.Series([0.0, 0.0, 0.0], index=_weights().index)
    pd.testing.assert_series_equal(result, expected)
