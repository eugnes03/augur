import pandas as pd

from augur.returns import log_return


def trailing_momentum(panel: pd.DataFrame, lookback: int) -> pd.Series:
    """Trailing `lookback`-day log return of adj_close, per ticker."""
    adj_close = panel['adj_close']
    return adj_close.groupby(level='ticker').transform(lambda s: log_return(s, lookback))


def short_term_reversal(panel: pd.DataFrame, lookback: int) -> pd.Series:
    """Negative trailing `lookback`-day log return of adj_close, per ticker."""
    return -trailing_momentum(panel, lookback)


def combine_signals(signals: list[pd.Series]) -> pd.Series:
    """Reduce multiple aligned signals to one by averaging their cross-sectional z-scores."""
    zscored = [_cross_sectional_zscore(signal) for signal in signals]
    combined: pd.Series = pd.concat(zscored, axis=1).mean(axis=1)
    return combined


def _cross_sectional_zscore(signal: pd.Series) -> pd.Series:
    """Standardize `signal` within each date's cross-section (mean 0, std 1)."""
    by_date = signal.groupby(level='timestamp')
    return (signal - by_date.transform('mean')) / by_date.transform('std')
