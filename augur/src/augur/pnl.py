import pandas as pd


def strategy_returns(weights: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """
    Compute daily portfolio log return as the weighted sum of forward
    returns.

    `weights` and `forward_returns` are both wide DataFrames indexed by
    timestamp (rows) with one column per ticker. Caller is responsible
    for alignment: `weights` at time t must already be paired with the
    return realized from t to t+1 (e.g. by shifting forward_returns
    before calling). This function does no shifting of its own.
    """
    return (weights * forward_returns).sum(axis=1)
