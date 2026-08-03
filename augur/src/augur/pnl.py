import numpy as np
import pandas as pd


def strategy_returns(weights: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """
    Daily portfolio log return: weighted sum of forward returns. Caller
    must pre-align `weights` at t with the return realized t->t+1. A
    held (nonzero-weight) position with an unknown return makes that date NaN.
    """
    contribution = weights * forward_returns
    daily_return = contribution.sum(axis=1, skipna=True)

    weight_unknown = weights.isna()
    held_return_unknown = weights.ne(0) & forward_returns.isna()
    unreliable = (weight_unknown | held_return_unknown).any(axis=1)
    daily_return[unreliable] = np.nan

    return daily_return
