"""Tests for augur.schemas.BarSchema."""

import pandas as pd
import pandera as pa
import pytest

from augur.schemas import BarSchema


def test_valid_frame_passes_validation(valid_bars: pd.DataFrame) -> None:
    """A well-formed OHLCV DataFrame should validate against BarSchema."""
    BarSchema.validate(valid_bars)


def test_naive_timestamp_rejected(valid_bars: pd.DataFrame) -> None:
    """A DatetimeIndex without a timezone should fail validation."""
    naive_index = pd.DatetimeIndex(valid_bars.index).tz_localize(None)
    naive = valid_bars.set_index(naive_index)

    with pytest.raises(pa.errors.SchemaError):
        BarSchema.validate(naive)


def test_ohlc_violation_rejected(valid_bars: pd.DataFrame) -> None:
    """A bar where low > open/close/high (or high < any) should fail validation."""
    invalid = valid_bars.copy()
    invalid.loc[invalid.index[0], "low"] = invalid.loc[invalid.index[0], "high"] + 1

    with pytest.raises(pa.errors.SchemaError):
        BarSchema.validate(invalid)


def test_non_utc_timezone_rejected(valid_bars: pd.DataFrame) -> None:
    """A tz-aware index in a timezone other than UTC should fail validation."""
    non_utc_index = pd.DatetimeIndex(valid_bars.index).tz_convert("America/New_York")
    non_utc = valid_bars.set_index(non_utc_index)

    with pytest.raises(pa.errors.SchemaError):
        BarSchema.validate(non_utc)


def test_duplicate_timestamps_rejected(valid_bars: pd.DataFrame) -> None:
    """An index with duplicate timestamps should fail validation."""
    duplicated = valid_bars.set_index(
        pd.DatetimeIndex(
            [valid_bars.index[0], valid_bars.index[0]],
            name="timestamp",
        )
    )

    with pytest.raises(pa.errors.SchemaError):
        BarSchema.validate(duplicated)


def test_unsorted_index_rejected(valid_bars: pd.DataFrame) -> None:
    """A non-monotonic (unsorted) index should fail validation."""
    unsorted = valid_bars.set_index(valid_bars.index[::-1])

    with pytest.raises(pa.errors.SchemaError):
        BarSchema.validate(unsorted)


def test_missing_columns_rejected(valid_bars: pd.DataFrame) -> None:
    """A DataFrame missing one or more required OHLCV columns should fail validation."""
    missing_column = valid_bars.drop(columns=["volume"])

    with pytest.raises(pa.errors.SchemaError):
        BarSchema.validate(missing_column)


def test_negative_volume_rejected(valid_bars: pd.DataFrame) -> None:
    """A row with negative volume should fail validation."""
    invalid = valid_bars.copy()
    invalid.loc[invalid.index[0], "volume"] = -1

    with pytest.raises(pa.errors.SchemaError):
        BarSchema.validate(invalid)


def test_wrong_dtype_rejected(valid_bars: pd.DataFrame) -> None:
    """Columns with an incorrect dtype (e.g. price as object) should fail validation."""
    invalid = valid_bars.copy()
    invalid["open"] = invalid["open"].astype(object)

    with pytest.raises(pa.errors.SchemaError):
        BarSchema.validate(invalid)
