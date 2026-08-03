import pandas as pd

from augur.returns import log_return


def trailing_momentum(panel: pd.DataFrame, lookback: int) -> pd.Series:
    """Trailing `lookback`-day log return of adj_close, per ticker."""
    adj_close = panel['adj_close']
    return adj_close.groupby(level='ticker').transform(lambda s: log_return(s, lookback))
