import numpy as np
import pandas as pd


def log_return(prices: pd.Series, periods: int) -> pd.Series:
    result: pd.Series = np.log(prices).diff(periods)
    return result
