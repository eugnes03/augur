"""Tests for augur.universe."""

import pytest

from augur import universe


def _fake_nasdaq_100_tickers() -> list[str]:
    """Stand-in for a scraped ticker list, deliberately unsorted."""
    return ["ZZZZ", "AAAA"]


def test_get_universe_combines_nordic_and_nasdaq(monkeypatch: pytest.MonkeyPatch) -> None:
    """The combined universe is the static Nordic list plus scraped NASDAQ-100 tickers."""
    monkeypatch.setattr(universe, "_fetch_nasdaq_100_tickers", _fake_nasdaq_100_tickers)

    result = universe.get_universe()

    assert result == universe.NORDIC_UNIVERSE + ["ZZZZ", "AAAA"]


def test_get_universe_returns_a_fresh_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutating the returned list must not affect the module-level NORDIC_UNIVERSE."""
    monkeypatch.setattr(universe, "_fetch_nasdaq_100_tickers", _fake_nasdaq_100_tickers)

    result = universe.get_universe()
    result.append("MUTATED")

    assert "MUTATED" not in universe.NORDIC_UNIVERSE
