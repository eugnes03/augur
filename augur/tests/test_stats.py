"""Tests for augur.stats."""

import numpy as np
import pandas as pd
import pytest

from augur.stats import (
    alpha,
    annualized_return,
    annualized_volatility,
    beta,
    deflated_sharpe_ratio,
    expected_max_sharpe_ratio,
    gross_exposure,
    max_drawdown,
    net_exposure,
    probabilistic_sharpe_ratio,
    random_trial_sharpe_ratios,
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


def test_probabilistic_sharpe_ratio_is_half_at_the_benchmark() -> None:
    """Benchmarking against the observed Sharpe itself gives a z-score of 0, so PSR == 0.5."""
    returns = pd.Series([0.01, 0.03, 0.02, -0.01, 0.015])
    observed_sharpe = returns.mean() / returns.std()

    assert probabilistic_sharpe_ratio(returns, observed_sharpe) == pytest.approx(0.5)


def test_probabilistic_sharpe_ratio_decreases_as_benchmark_rises() -> None:
    """A tougher benchmark Sharpe should only make the observed Sharpe less likely to clear it."""
    returns = pd.Series([0.01, 0.03, 0.02, -0.01, 0.015])

    assert probabilistic_sharpe_ratio(returns, 0.0) > probabilistic_sharpe_ratio(returns, 5.0)


def test_expected_max_sharpe_ratio_hand_computed() -> None:
    """Bailey & Lopez de Prado (2014) eq. 10, hand-computed for 3 trials with unit std."""
    trial_sharpe_ratios = np.array([-1.0, 0.0, 1.0])

    assert expected_max_sharpe_ratio(trial_sharpe_ratios) == pytest.approx(0.852804496150695)


def test_expected_max_sharpe_ratio_requires_at_least_two_trials() -> None:
    """A single trial has no spread to estimate E[max SR] from."""
    with pytest.raises(ValueError, match="at least 2 trials"):
        expected_max_sharpe_ratio(np.array([1.0]))


def test_deflated_sharpe_ratio_matches_psr_at_expected_max_benchmark() -> None:
    """deflated_sharpe_ratio is just PSR benchmarked at expected_max_sharpe_ratio's output."""
    returns = pd.Series([0.01, 0.03, 0.02, -0.01, 0.015])
    trial_sharpe_ratios = np.array([-1.0, 0.0, 1.0])

    expected = probabilistic_sharpe_ratio(
        returns, expected_max_sharpe_ratio(trial_sharpe_ratios)
    )
    assert deflated_sharpe_ratio(returns, trial_sharpe_ratios) == pytest.approx(expected)


def test_random_trial_sharpe_ratios_shape_and_reproducibility() -> None:
    """A seeded generator produces one Sharpe ratio per trial, deterministically."""
    asset_returns = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (50, 4)))

    result_a = random_trial_sharpe_ratios(asset_returns, n_trials=20, rng=np.random.default_rng(1))
    result_b = random_trial_sharpe_ratios(asset_returns, n_trials=20, rng=np.random.default_rng(1))

    assert result_a.shape == (20,)
    np.testing.assert_array_equal(result_a, result_b)


def test_random_trial_sharpe_ratios_weights_are_dollar_neutral() -> None:
    """Each trial's weights should net to zero and sum to unit gross exposure."""
    asset_returns = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (50, 4)))
    rng = np.random.default_rng(2)

    raw_weights = rng.normal(size=(1000, asset_returns.shape[1]))
    raw_weights -= raw_weights.mean(axis=1, keepdims=True)
    weights = raw_weights / np.abs(raw_weights).sum(axis=1, keepdims=True)

    np.testing.assert_allclose(weights.sum(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.abs(weights).sum(axis=1), 1.0, atol=1e-10)


def _capm_series() -> tuple[pd.Series, pd.Series]:
    """Returns built as exactly 0.001 + 1.5*benchmark (zero noise): beta=1.5, alpha_period=0.001."""
    benchmark_returns = pd.Series([0.01, -0.02, 0.03, 0.0, 0.015])
    returns = 0.001 + 1.5 * benchmark_returns
    return returns, benchmark_returns


def test_beta_hand_computed_zero_noise() -> None:
    """A returns series built as an exact linear function of the benchmark recovers its slope."""
    returns, benchmark_returns = _capm_series()
    assert beta(returns, benchmark_returns) == pytest.approx(1.5)


def test_alpha_hand_computed_zero_noise() -> None:
    """Same zero-noise series recovers its per-period intercept, annualized by periods_per_year."""
    returns, benchmark_returns = _capm_series()
    assert alpha(returns, benchmark_returns, periods_per_year=1) == pytest.approx(0.001)
    assert alpha(returns, benchmark_returns, periods_per_year=252) == pytest.approx(0.001 * 252)


def test_beta_aligns_on_mismatched_index() -> None:
    """Dates present in only one series should be dropped before regressing, not raise."""
    returns, benchmark_returns = _capm_series()
    returns_missing_first = returns.iloc[1:]
    benchmark_missing_last = benchmark_returns.iloc[:-1]
    overlap_labels = returns_missing_first.index.intersection(benchmark_missing_last.index)

    assert beta(returns_missing_first, benchmark_missing_last) == pytest.approx(
        beta(returns.loc[overlap_labels], benchmark_returns.loc[overlap_labels])
    )


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
