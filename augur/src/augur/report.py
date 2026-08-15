from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from augur import stats


def write_report(
    returns: pd.Series,
    weights: pd.DataFrame,
    path: Path,
    trial_sharpe_ratios: np.ndarray | None = None,
    benchmark_returns: pd.Series | None = None,
) -> None:
    """
    Write a markdown stats table and a cumulative-return PNG for `returns` to `path`.
    Pass `trial_sharpe_ratios`, the per-period Sharpe of every grid point in a real
    parameter sweep, to also report the deflated Sharpe ratio against that sweep.
    Pass `benchmark_returns` to also report alpha/beta against that benchmark.
    """
    png_path = path.with_suffix(".png")
    _save_cumulative_return_plot(returns, png_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(
        returns, weights, png_path.name, trial_sharpe_ratios, benchmark_returns
    )
    path.write_text(markdown)


def _save_cumulative_return_plot(returns: pd.Series, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    returns.cumsum().plot(ax=ax)
    ax.set_title("Cumulative log return")
    ax.set_ylabel("Cumulative log return")
    fig.savefig(png_path)
    plt.close(fig)


def _deflation_stats(returns: pd.Series, trial_sharpe_ratios: np.ndarray) -> dict[str, float]:
    """n_trials: real sweep grid size the deflation is corrected for."""
    return {
        "n_trials": len(trial_sharpe_ratios),
        "expected_max_sharpe_ratio": stats.expected_max_sharpe_ratio(trial_sharpe_ratios),
        "deflated_sharpe_ratio": stats.deflated_sharpe_ratio(returns, trial_sharpe_ratios),
    }


def _regression_stats(returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, float]:
    return {
        "beta": stats.beta(returns, benchmark_returns),
        "alpha_annualized": stats.alpha(returns, benchmark_returns),
    }


def _render_markdown(
    returns: pd.Series,
    weights: pd.DataFrame,
    png_name: str,
    trial_sharpe_ratios: np.ndarray | None,
    benchmark_returns: pd.Series | None,
) -> str:
    performance_stats = {
        "total_return": stats.total_return(returns),
        "annualized_return": stats.annualized_return(returns),
        "annualized_volatility": stats.annualized_volatility(returns),
        "sharpe_ratio": stats.sharpe_ratio(returns),
        "max_drawdown": stats.max_drawdown(returns),
        "avg_turnover_per_rebalance": stats.turnover(weights).mean(),
        "avg_gross_exposure": stats.gross_exposure(weights).mean(),
        "avg_net_exposure": stats.net_exposure(weights).mean(),
    }
    if trial_sharpe_ratios is not None:
        performance_stats.update(_deflation_stats(returns, trial_sharpe_ratios))
    if benchmark_returns is not None:
        performance_stats.update(_regression_stats(returns, benchmark_returns))
    rows = "\n".join(f"| {name} | {value:.4f} |" for name, value in performance_stats.items())
    return (
        "# Backtest report\n\n"
        "| stat | value |\n"
        "| --- | --- |\n"
        f"{rows}\n\n"
        f"![Cumulative return]({png_name})\n"
    )
