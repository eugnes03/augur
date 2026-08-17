import pandas as pd

from augur.returns import log_return


def trailing_momentum(panel: pd.DataFrame, lookback: int) -> pd.Series:
    """Trailing `lookback`-day log return of adj_close, per ticker."""
    adj_close = panel['adj_close']
    return adj_close.groupby(level='ticker').transform(lambda s: log_return(s, lookback))


def short_term_reversal(panel: pd.DataFrame, lookback: int) -> pd.Series:
    """Negative trailing `lookback`-day log return of adj_close, per ticker."""
    return -trailing_momentum(panel, lookback)


def trailing_volatility(panel: pd.DataFrame, lookback: int) -> pd.Series:
    """Trailing `lookback`-day realized volatility (std of daily log returns), per ticker."""
    daily_returns = panel['adj_close'].groupby(level='ticker').transform(
        lambda s: log_return(s, periods=1)
    )
    return daily_returns.groupby(level='ticker').transform(lambda s: s.rolling(lookback).std())


def combine_signals(
    signals: dict[str, pd.Series], weights: dict[str, float] | None = None
) -> pd.Series:
    """
    Reduce multiple aligned signals to one weighted sum of their cross-sectional
    z-scores. Z-scoring first keeps a signal's raw scale from dominating the
    combination. Defaults to equal weighting across signals.
    """
    weights = weights if weights is not None else dict.fromkeys(signals, 1.0 / len(signals))
    zscored = {
        name: _cross_sectional_zscore(signal) * weights[name] for name, signal in signals.items()
    }
    combined: pd.Series = pd.concat(zscored, axis=1).sum(axis=1)
    return combined


def _cross_sectional_zscore(signal: pd.Series) -> pd.Series:
    """Standardize `signal` within each date's cross-section (mean 0, std 1)."""
    by_date = signal.groupby(level='timestamp')
    return (signal - by_date.transform('mean')) / by_date.transform('std')
