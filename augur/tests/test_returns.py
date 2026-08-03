"""Tests for augur.returns."""

import numpy as np
import pandas as pd

from augur.returns import log_return


def test_log_return_on_known_price_series() -> None:
    """log_return over a known price series matches hand-computed values."""
    prices = pd.Series([100.0, 110.0, 121.0])

    result = log_return(prices, periods=1)

    expected = pd.Series([np.nan, np.log(1.1), np.log(1.1)])
    pd.testing.assert_series_equal(result, expected)
