import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS_PER_YEAR = 252
_EULER_MASCHERONI = 0.5772156649015329  # standard symbol: gamma


def total_return(returns: pd.Series) -> float:
    """Cumulative simple-return growth over the full period of periodic log `returns`."""
    return float(np.expm1(returns.sum()))


def annualized_return(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """CAGR: per-year growth rate compounding to total_return over len(returns) periods."""
    growth = 1.0 + total_return(returns)
    return float(growth ** (periods_per_year / len(returns)) - 1.0)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Standard deviation of `returns`, scaled to an annual horizon."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe ratio of `returns`, assuming a flat 0% risk-free rate."""
    vol = annualized_volatility(returns, periods_per_year)
    return float(returns.mean() * periods_per_year / vol)


def random_trial_sharpe_ratios(
    asset_returns: pd.DataFrame, n_trials: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Per-period Sharpe ratios of `n_trials` random dollar-neutral portfolios,
    trial population for deflated_sharpe_ratio() when no real parameter sweep exists.
    """
    rng = rng if rng is not None else np.random.default_rng()
    raw_weights = rng.normal(size=(n_trials, asset_returns.shape[1]))
    raw_weights -= raw_weights.mean(axis=1, keepdims=True)  # net exposure 0
    weights = raw_weights / np.abs(raw_weights).sum(axis=1, keepdims=True)  # gross exposure 1

    trial_returns = asset_returns.to_numpy() @ weights.T
    sharpe_ratios: np.ndarray = trial_returns.mean(axis=0) / trial_returns.std(axis=0, ddof=1)
    return sharpe_ratios


def expected_max_sharpe_ratio(trial_sharpe_ratios: np.ndarray) -> float:
    """
    E[max Sharpe] over independent trials, null-hypothesis benchmark for deflated_sharpe_ratio().
    E[max SR_N] = std(SR) * [(1-gamma)*Z^-1(1-1/N) + gamma*Z^-1(1-1/(N*e))],
    Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014), eq. 10.
    """
    trial_sharpe_ratios = np.asarray(trial_sharpe_ratios, dtype=float)
    n_trials = len(trial_sharpe_ratios)
    if n_trials < 2:
        raise ValueError("expected_max_sharpe_ratio needs at least 2 trials")

    sharpe_std = trial_sharpe_ratios.std(ddof=1)
    non_asymptotic_term = (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    asymptotic_term = _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_std * (non_asymptotic_term + asymptotic_term))


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe: float) -> float:
    """
    P(true per-period Sharpe of `returns` exceeds `benchmark_sharpe`), correcting for
    skew, kurtosis, and sample length rather than assuming Gaussian returns.
    PSR(SR*) = Z[(SR - SR*) * sqrt(T-1) / sqrt(1 - skew*SR + (kurtosis-1)/4*SR^2)],
    Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (2012), eq. 6.
    """
    n_obs = len(returns)
    observed_sharpe = float(returns.mean() / returns.std())
    skew = float(returns.skew())  # type: ignore[arg-type]
    kurtosis = float(returns.kurtosis()) + 3.0  # type: ignore[arg-type]  # pandas: excess kurtosis

    variance_term = 1.0 - skew * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    z_score = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(variance_term)
    return float(norm.cdf(z_score))


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpe_ratios: np.ndarray) -> float:
    """
    PSR of `returns` benchmarked against the expected max Sharpe of `trial_sharpe_ratios`,
    correcting the observed Sharpe for having been selected out of many trials.
    """
    benchmark_sharpe = expected_max_sharpe_ratio(trial_sharpe_ratios)
    return probabilistic_sharpe_ratio(returns, benchmark_sharpe)


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """OLS beta of `returns` on `benchmark_returns`: cov(returns, benchmark) / var(benchmark)."""
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")
    covariance = aligned_returns.cov(aligned_benchmark)
    variance = aligned_benchmark.var()
    return float(covariance / variance)


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized OLS alpha: r = alpha_period + beta*r_benchmark + e, alpha_period*periods_per_year.
    """
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")
    strategy_beta = beta(returns, benchmark_returns)
    period_alpha = aligned_returns.mean() - strategy_beta * aligned_benchmark.mean()
    return float(period_alpha * periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline in cumulative wealth from `returns` (a negative fraction)."""
    wealth = np.exp(returns.cumsum())
    return float((wealth / wealth.cummax() - 1.0).min())


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per date: half the gross change in weights since the prior date."""
    result: pd.Series = 0.5 * weights.diff().abs().sum(axis=1)
    return result


def gross_exposure(weights: pd.DataFrame) -> pd.Series:
    """Sum of absolute position weights per date; ~2 for a fully-invested dollar-neutral book."""
    result: pd.Series = weights.abs().sum(axis=1)
    return result


def net_exposure(weights: pd.DataFrame) -> pd.Series:
    """Sum of signed position weights per date; ~0 for a dollar-neutral book."""
    result: pd.Series = weights.sum(axis=1)
    return result
