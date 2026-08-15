"""Tests for augur.universe."""

import pytest

from augur import universe


def _fake_nasdaq_100_tickers() -> list[str]:
    """Stand-in for a scraped ticker list, deliberately unsorted."""
    return ["ZZZZ", "AAAA"]


def test_get_universe_returns_nasdaq_100_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_universe() is NASDAQ-100 only for now; NORDIC_UNIVERSE is parked, not mixed in."""
    monkeypatch.setattr(universe, "_fetch_nasdaq_100_tickers", _fake_nasdaq_100_tickers)

    result = universe.get_universe()

    assert result == ["ZZZZ", "AAAA"]


def test_get_universe_returns_a_fresh_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating the returned list must not affect the module-level fixture used to build it."""
    fixture_tickers = ["ZZZZ", "AAAA"]
    monkeypatch.setattr(universe, "_fetch_nasdaq_100_tickers", lambda: fixture_tickers)

    result = universe.get_universe()
    result.append("MUTATED")

    assert "MUTATED" not in fixture_tickers
