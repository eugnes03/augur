"""Tests for augur.ingest."""

from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf

from augur import ingest


def _raw_yfinance_shape(ticker: str) -> pd.DataFrame:
    """A minimal DataFrame matching yfinance's raw (Price, Ticker) MultiIndex columns."""
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], [ticker]],
        names=["Price", "Ticker"],
    )
    return pd.DataFrame(
        [[100.0, 102.0, 99.0, 101.0, 101.0, 1_000], [101.0, 103.0, 100.0, 102.0, 102.0, 1_500]],
        index=index,
        columns=columns,
    )


def test_normalize_runs_without_network_access() -> None:
    """_normalize() should transform a raw DataFrame without any network calls."""


def test_normalize_timestamp_conversion_correctness() -> None:
    """_normalize() should convert yfinance timestamps to tz-aware UTC datetime64[ns]."""


def test_normalize_dtype_coercion() -> None:
    """_normalize() should coerce columns to BarSchema's expected dtypes."""


def test_drop_inconsistent_rows_removes_only_bad_rows(valid_bars: pd.DataFrame) -> None:
    """A row with low > close should be dropped; other rows must survive."""
    bars = valid_bars.copy()
    bars.loc[bars.index[0], "low"] = bars.loc[bars.index[0], "close"] + 1.0

    result = ingest._drop_inconsistent_rows(bars, "TEST")

    assert len(result) == len(valid_bars) - 1
    assert bars.index[0] not in result.index
    assert bars.index[1] in result.index


def test_drop_inconsistent_rows_keeps_all_consistent_rows(valid_bars: pd.DataFrame) -> None:
    """A fully consistent frame should pass through unchanged."""
    result = ingest._drop_inconsistent_rows(valid_bars, "TEST")

    pd.testing.assert_frame_equal(result, valid_bars)


def test_fetch_raw_writes_cache_file_on_first_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache miss should hit the network and write a parquet file for next time."""
    calls = 0

    def _fake_download(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return _raw_yfinance_shape("TEST")

    monkeypatch.setattr(yf, "download", _fake_download)

    result = ingest._fetch_raw("TEST", "2024-01-01", "2024-01-04", cache_dir=tmp_path)

    assert calls == 1
    assert (tmp_path / "TEST_2024-01-01_2024-01-04.parquet").exists()
    pd.testing.assert_frame_equal(result, _raw_yfinance_shape("TEST"))


def test_fetch_raw_reads_cache_on_second_call_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Once cached, subsequent calls for the same key must not call yf.download again."""
    calls = 0

    def _fake_download(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return _raw_yfinance_shape("TEST")

    monkeypatch.setattr(yf, "download", _fake_download)

    ingest._fetch_raw("TEST", "2024-01-01", "2024-01-04", cache_dir=tmp_path)
    result = ingest._fetch_raw("TEST", "2024-01-01", "2024-01-04", cache_dir=tmp_path)

    assert calls == 1
    pd.testing.assert_frame_equal(result, _raw_yfinance_shape("TEST"))
