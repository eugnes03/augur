import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    from dataclasses import dataclass

    import matplotlib.pyplot as plt
    import pandas as pd

    from augur import ingest, pnl, portfolio, returns, signals, stats, universe

    @dataclass(frozen=True)
    class BacktestConfig:
        start: str
        end: str
        lookback: int
        n_long: int
        n_short: int

    config = BacktestConfig(
        start="2024-01-01",
        end="2024-12-31",
        lookback=20,
        n_long=3,
        n_short=3,
    )
    return (
        config,
        ingest,
        pd,
        plt,
        pnl,
        portfolio,
        returns,
        signals,
        stats,
        universe,
    )


@app.cell
def _(universe):
    tickers = universe.get_universe()
    return (tickers,)


@app.cell
def _(config, ingest, tickers):
    raw_bars = ingest.fetch_universe_bars(start=config.start, end=config.end, tickers=tickers)
    panel = ingest.stack_universe_bars(raw_bars)
    return (panel,)


@app.cell
def _(config, panel, signals):
    momentum = signals.trailing_momentum(panel, lookback=config.lookback)
    return (momentum,)


@app.cell
def _(config, momentum, pd, portfolio):
    def _daily_weights(cross_section: pd.Series) -> pd.Series:
        available = cross_section.dropna()
        if len(available) < config.n_long + config.n_short:
            # Lookback warmup period: not enough tickers have momentum yet.
            return pd.Series(0.0, index=cross_section.index)
        return portfolio.equal_weight(
            available, n_long=config.n_long, n_short=config.n_short
        ).reindex(cross_section.index, fill_value=0.0)

    weights = momentum.unstack("ticker").apply(_daily_weights, axis=1)
    return (weights,)


@app.cell
def _(panel, returns):
    daily_returns = panel["adj_close"].groupby(level="ticker").transform(
        lambda s: returns.log_return(s, periods=1)
    )
    forward_returns = daily_returns.unstack("ticker").shift(-1)
    return (forward_returns,)


@app.cell
def _(forward_returns, pnl, weights):
    aligned_weights = weights.shift(1).reindex(forward_returns.index)
    daily_pnl = pnl.strategy_returns(aligned_weights, forward_returns).dropna()
    return (daily_pnl,)


@app.cell
def _(daily_pnl, plt):
    cumulative_return = daily_pnl.cumsum()
    fig, ax = plt.subplots()
    cumulative_return.plot(ax=ax)
    ax.set_title("Cumulative log return: trailing-momentum long/short")
    ax.set_ylabel("Cumulative log return")
    fig
    return


@app.cell
def _(daily_pnl, pd, stats):
    performance_stats = pd.Series(
        {
            "total_return": stats.total_return(daily_pnl),
            "annualized_return": stats.annualized_return(daily_pnl),
            "annualized_volatility": stats.annualized_volatility(daily_pnl),
            "sharpe_ratio": stats.sharpe_ratio(daily_pnl),
            "max_drawdown": stats.max_drawdown(daily_pnl),
        }
    )
    performance_stats
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
