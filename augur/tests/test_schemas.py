"""Tests for augur.schemas.BarSchema."""


def test_valid_frame_passes_validation() -> None:
    """A well-formed OHLCV DataFrame should validate against BarSchema."""


def test_naive_timestamp_rejected() -> None:
    """A DatetimeIndex without a timezone should fail validation."""


def test_ohlc_violation_rejected() -> None:
    """A bar where low > open/close/high (or high < any) should fail validation."""


def test_non_utc_timezone_rejected() -> None:
    """A tz-aware index in a timezone other than UTC should fail validation."""


def test_duplicate_timestamps_rejected() -> None:
    """An index with duplicate timestamps should fail validation."""


def test_unsorted_index_rejected() -> None:
    """A non-monotonic (unsorted) index should fail validation."""


def test_missing_columns_rejected() -> None:
    """A DataFrame missing one or more required OHLCV columns should fail validation."""


def test_negative_volume_rejected() -> None:
    """A row with negative volume should fail validation."""


def test_wrong_dtype_rejected() -> None:
    """Columns with an incorrect dtype (e.g. price as object) should fail validation."""
