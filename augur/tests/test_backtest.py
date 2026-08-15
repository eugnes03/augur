"""Tests for augur.backtest."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from augur import backtest, ingest, pnl, portfolio
from augur.config import Config


def _session_close_index(start: str, periods: int) -> pd.DatetimeIndex:
    """Business-day session-close timestamps, matching ingest's UTC 16:00 convention."""
    dates = pd.bdate_range(start, periods=periods)
    index = pd.DatetimeIndex(dates, name="timestamp").tz_localize("UTC") + pd.Timedelta(hours=16)
    return pd.DatetimeIndex(index.astype("datetime64[ns, UTC]"), name="timestamp")


def _make_bars(daily_log_returns: list[float], start: str) -> pd.DataFrame:
    """A single-ticker BarSchema-compliant panel whose adj_close follows a log-return path."""
    index = _session_close_index(start, len(daily_log_returns))
    price = np.exp(np.cumsum(daily_log_returns))
    bars = pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "adj_close": price,
            "volume": 1_000,
            "stock_splits": 0.0,
            "dividends": 0.0,
        },
        index=index,
    )
    return bars.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "adj_close": "float64",
            "volume": "int64",
            "stock_splits": "float64",
            "dividends": "float64",
        }
    )


def _fake_fetch_universe_bars(
    bars: dict[str, pd.DataFrame],
) -> Callable[..., dict[str, pd.DataFrame]]:
    def _fetch(*_args: object, **_kwargs: object) -> dict[str, pd.DataFrame]:
        return bars

    return _fetch


def test_run_backtest_synthetic_panel_matches_hand_computed_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    End-to-end: two tickers, momentum + reversal combined into one long/short pick a day,
    daily pnl from next-day returns.
    """
    # AAA's daily log return alternates sign and halves in size each day (-0.5, +0.25,
    # -0.125, ...) while BBB never moves. Each day's move is larger than, and opposite in
    # sign to, the prior day, so it dominates the running 2-day sum that reversal is built
    # on. That makes momentum(1) and reversal always agree on AAA's direction instead of
    # cancelling, so every day's long/short pick is unambiguous.
    aaa_returns = [0.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125]
    bbb_returns = [0.0] * len(aaa_returns)
    bars = {
        "AAA": _make_bars(aaa_returns, "2024-01-02"),
        "BBB": _make_bars(bbb_returns, "2024-01-02"),
    }
    monkeypatch.setattr(ingest, "fetch_universe_bars", _fake_fetch_universe_bars(bars))
    config = Config(
        start="2024-01-02",
        end="2024-01-16",
        momentum_lookback=1,
        reversal_lookback=2,
        n_long=1,
        n_short=1,
    )

    result = backtest.run_backtest(config)

    # BBB is always flat, so the realized return each day is just AAA's next-day return,
    # signed by whichever ticker AAA's combined signal picked long the day before.
    expected = pd.Series(
        [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
        index=_session_close_index("2024-01-03", 6),
    )
    pd.testing.assert_series_equal(result.returns, expected, check_names=False)


def test_run_backtest_uses_next_day_return_not_same_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    run_backtest must align a day's weights with the *following* day's return
    (`weights.shift(1)` in run_backtest), not the return realized the same day the
    weights were decided. Using the same-day return here would flip the sign.
    """
    aaa_returns = [0.0, 0.10, -0.20, 0.05, 0.30]
    bbb_returns = [0.0] * len(aaa_returns)
    bars = {
        "AAA": _make_bars(aaa_returns, "2024-01-02"),
        "BBB": _make_bars(bbb_returns, "2024-01-02"),
    }
    monkeypatch.setattr(ingest, "fetch_universe_bars", _fake_fetch_universe_bars(bars))
    config = Config(
        start="2024-01-02",
        end="2024-01-10",
        momentum_lookback=1,
        reversal_lookback=10,  # never warms up in this five-day panel; momentum alone decides
        n_long=1,
        n_short=1,
    )
    target_date = pd.Timestamp("2024-01-04 16:00", tz="UTC")

    result = backtest.run_backtest(config)

    assert result.returns[target_date] == pytest.approx(0.05)

    # Reconstruct the same pipeline but without the lookahead-safety shift, to show it's a
    # genuinely different, wrong number rather than a distinction without a difference.
    panel = backtest._load_panel(config)
    combined_signal = backtest._combined_signal(panel, config)
    weights = backtest._daily_weights(combined_signal, config)
    weights = portfolio.apply_rebalance_frequency(weights, config.rebalance_frequency)
    forward_returns = backtest._forward_returns(panel)
    same_day_returns = pnl.strategy_returns(weights, forward_returns).dropna()

    assert same_day_returns[target_date] == pytest.approx(-0.05)


def test_daily_weights_returns_zero_row_when_below_warmup_threshold() -> None:
    """
    A date where fewer tickers have a signal than n_long + n_short requires can't be split
    into a valid long/short portfolio, so _daily_weights must fall back to an all-zero row
    for that date instead of raising.
    """
    config = Config(
        start="2024-01-01",
        end="2024-01-03",
        momentum_lookback=1,
        reversal_lookback=1,
        n_long=1,
        n_short=1,
    )
    warm_date = pd.Timestamp("2024-01-01", tz="UTC")
    live_date = pd.Timestamp("2024-01-02", tz="UTC")
    index = pd.MultiIndex.from_tuples(
        [
            (warm_date, "AAA"),
            (warm_date, "BBB"),
            (warm_date, "CCC"),
            (live_date, "AAA"),
            (live_date, "BBB"),
            (live_date, "CCC"),
        ],
        names=["timestamp", "ticker"],
    )
    # Only AAA has a real signal on warm_date: 1 available ticker < n_long + n_short (2).
    combined_signal = pd.Series([1.0, np.nan, np.nan, 1.0, 0.5, -1.0], index=index)

    result = backtest._daily_weights(combined_signal, config)

    assert (result.loc[warm_date] == 0.0).all()
    assert result.loc[live_date, "AAA"] == 1.0
    assert result.loc[live_date, "BBB"] == 0.0
    assert result.loc[live_date, "CCC"] == -1.0
