import pandas as pd  # noqa: F401
import pandera as pa  # noqa: F401


class BarSchema:
    """
    Pandera schema for a single-ticker OHLCV bar DataFrame.

    Expected shape (to be implemented):
    - DatetimeIndex named 'timestamp', dtype datetime64[ns, UTC]
    - columns: open, high, low, close, adj_close (float64), volume (int64)
    - strict=True, coerce=False

    TODO (not yet implemented):
    - field-level constraints (e.g. non-negative volume, positive prices)
    - a dataframe-level check for OHLC consistency
      (low <= open/close <= high)
    - a dataframe-level check for index monotonicity (sorted, no duplicates)
    """

    raise NotImplementedError("BarSchema fields and checks not yet defined")
