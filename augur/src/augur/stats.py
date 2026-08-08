import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


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
