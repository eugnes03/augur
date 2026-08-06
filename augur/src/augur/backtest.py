from dataclasses import dataclass

import pandas as pd

from augur import ingest, pnl, portfolio, returns, signals, universe


@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    momentum_lookback: int
    reversal_lookback: int
    n_long: int
    n_short: int
    rebalance_frequency: int = 1


def run_backtest(config: BacktestConfig) -> pd.Series:
    """Run the full momentum+reversal long/short pipeline for `config`, returning daily pnl."""
    panel = _load_panel(config)
    combined_signal = _combined_signal(panel, config)
    weights = _daily_weights(combined_signal, config)
    weights = portfolio.apply_rebalance_frequency(weights, config.rebalance_frequency)
    forward_returns = _forward_returns(panel)
    aligned_weights = weights.shift(1).reindex(forward_returns.index)
    return pnl.strategy_returns(aligned_weights, forward_returns).dropna()


def _load_panel(config: BacktestConfig) -> pd.DataFrame:
    raw_bars = ingest.fetch_universe_bars(
        start=config.start, end=config.end, tickers=universe.get_universe()
    )
    return ingest.stack_universe_bars(raw_bars)


def _combined_signal(panel: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    momentum = signals.trailing_momentum(panel, lookback=config.momentum_lookback)
    reversal = signals.short_term_reversal(panel, lookback=config.reversal_lookback)
    return signals.combine_signals([momentum, reversal])


def _daily_weights(combined_signal: pd.Series, config: BacktestConfig) -> pd.DataFrame:
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
