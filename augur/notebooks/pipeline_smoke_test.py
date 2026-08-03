import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt

    from augur import ingest, pnl, portfolio, returns, signals, universe

    return ingest, plt, pnl, portfolio, returns, signals, universe


@app.cell
def _(universe):
    tickers = universe.get_universe()
    return (tickers,)


@app.cell
def _(ingest, tickers):
    raw_bars = ingest.fetch_universe_bars(start="2024-01-01", end="2024-12-31", tickers=tickers)
    panel = ingest.stack_universe_bars(raw_bars)
    return (panel,)


@app.cell
def _(panel, signals):
    momentum = signals.trailing_momentum(panel, lookback=20)
    return (momentum,)


@app.cell
def _(momentum, portfolio):
    n_long, n_short = 3, 3
    weights = momentum.unstack("ticker").apply(
        lambda cross_section: portfolio.equal_weight(
            cross_section.dropna(), n_long=n_long, n_short=n_short
        ).reindex(cross_section.index, fill_value=0.0),
        axis=1,
    )
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


if __name__ == "__main__":
    app.run()
