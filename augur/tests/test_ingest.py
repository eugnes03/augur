"""Tests for augur.ingest."""


def test_normalize_runs_without_network_access() -> None:
    """_normalize() should transform a raw DataFrame without any network calls."""


def test_normalize_timestamp_conversion_correctness() -> None:
    """_normalize() should convert yfinance timestamps to tz-aware UTC datetime64[ns]."""


def test_normalize_dtype_coercion() -> None:
    """_normalize() should coerce columns to BarSchema's expected dtypes."""
