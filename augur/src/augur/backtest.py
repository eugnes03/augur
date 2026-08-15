from dataclasses import dataclass

import pandas as pd

from augur import ingest, pnl, portfolio, returns, signals, universe
from augur.config import Config


@dataclass(frozen=True)
class BacktestResult:
    """Daily strategy pnl alongside the portfolio weights that produced it."""

    returns: pd.Series
    weights: pd.DataFrame


def run_backtest(config: Config) -> BacktestResult:
    """Run the full momentum+reversal long/short pipeline for `config`."""
    panel = _load_panel(config)
    combined_signal = _combined_signal(panel, config)
    weights = _daily_weights(combined_signal, config)
    weights = portfolio.apply_rebalance_frequency(weights, config.rebalance_frequency)
    forward_returns = _forward_returns(panel)
    aligned_weights = weights.shift(1).reindex(forward_returns.index)
    daily_returns = pnl.strategy_returns(aligned_weights, forward_returns).dropna()
    return BacktestResult(returns=daily_returns, weights=weights)


def benchmark_returns(config: Config, ticker: str = universe.BENCHMARK_TICKER) -> pd.Series:
    """
    Forward daily log return of `ticker` (default: the NASDAQ-100 index), using the same
    (t, t+1] indexing as run_backtest()'s returns so the two align for alpha/beta regression.
    """
    bars = ingest.fetch_bars(ticker, config.start, config.end)
    result: pd.Series = returns.log_return(bars["adj_close"], periods=1).shift(-1).dropna()
    return result


def _load_panel(config: Config) -> pd.DataFrame:
    tickers = config.universe if config.universe is not None else universe.get_universe()
    raw_bars = ingest.fetch_universe_bars(start=config.start, end=config.end, tickers=tickers)
    return ingest.stack_universe_bars(raw_bars)


def _combined_signal(panel: pd.DataFrame, config: Config) -> pd.Series:
    momentum = signals.trailing_momentum(panel, lookback=config.momentum_lookback)
    reversal = signals.short_term_reversal(panel, lookback=config.reversal_lookback)
    return signals.combine_signals({"momentum": momentum, "reversal": reversal})


def _daily_weights(combined_signal: pd.Series, config: Config) -> pd.DataFrame:
    def cross_section_weights(cross_section: pd.Series) -> pd.Series:
        available = cross_section.dropna()
        if len(available) < config.n_long + config.n_short:
            # Lookback warmup period: not enough tickers have a signal yet.
            return pd.Series(0.0, index=cross_section.index)
        return portfolio.equal_weight(
            available, n_long=config.n_long, n_short=config.n_short
        ).reindex(cross_section.index, fill_value=0.0)

    return combined_signal.unstack("ticker").apply(cross_section_weights, axis=1)  # noqa: PD010


def _forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    daily_returns = panel["adj_close"].groupby(level="ticker").transform(
        lambda s: returns.log_return(s, periods=1)
    )
    return daily_returns.unstack("ticker").shift(-1)  # noqa: PD010
