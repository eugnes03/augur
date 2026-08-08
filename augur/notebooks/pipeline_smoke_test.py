import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd

    from augur import stats
    from augur.backtest import run_backtest
    from augur.config import Config

    config = Config(
        start="2024-01-01",
        end="2024-12-31",
        momentum_lookback=20,
        reversal_lookback=5,
        n_long=3,
        n_short=3,
    )
    return Config, config, pd, plt, run_backtest, stats


@app.cell
def _(config, run_backtest):
    daily_pnl = run_backtest(config).returns
    return (daily_pnl,)


@app.cell
def _(daily_pnl, plt):
    cumulative_return = daily_pnl.cumsum()
    fig, ax = plt.subplots()
    cumulative_return.plot(ax=ax)
    ax.set_title("Cumulative log return: momentum+reversal long/short")
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
def _(Config, config, run_backtest):
    # Naive chronological split, not a parameter-selection tool.
    in_sample_config = Config(**{**config.__dict__, "start": "2024-01-01", "end": "2024-08-31"})
    out_of_sample_config = Config(**{**config.__dict__, "start": "2024-09-01", "end": "2024-12-31"})

    in_sample_pnl = run_backtest(in_sample_config).returns
    out_of_sample_pnl = run_backtest(out_of_sample_config).returns
    return in_sample_pnl, out_of_sample_pnl


@app.cell
def _(in_sample_pnl, out_of_sample_pnl, pd, stats):
    def _stats_row(returns: pd.Series) -> dict[str, float]:
        return {
            "total_return": stats.total_return(returns),
            "annualized_return": stats.annualized_return(returns),
            "annualized_volatility": stats.annualized_volatility(returns),
            "sharpe_ratio": stats.sharpe_ratio(returns),
            "max_drawdown": stats.max_drawdown(returns),
        }

    train_test_comparison = pd.DataFrame(
        {
            "in_sample": _stats_row(in_sample_pnl),
            "out_of_sample": _stats_row(out_of_sample_pnl),
        }
    )
    train_test_comparison
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
