import pandas as pd

from augur import stats


def net_of_costs(returns: pd.Series, weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    """
    Strategy `returns` minus a linear cost-per-unit-turnover drag: cost_bps applied to
    each date's one-way turnover in `weights` (stats.turnover), not to gross exposure.
    """
    cost = cost_bps / 10_000 * stats.turnover(weights)
    result: pd.Series = (returns - cost.reindex(returns.index, fill_value=0.0)).dropna()
    return result
