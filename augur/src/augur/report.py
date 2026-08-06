from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from augur import stats


def write_report(returns: pd.Series, path: Path) -> None:
    """Write a markdown stats table and a cumulative-return PNG for `returns` to `path`."""
    png_path = path.with_suffix(".png")
    _save_cumulative_return_plot(returns, png_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown(returns, png_path.name))


def _save_cumulative_return_plot(returns: pd.Series, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    returns.cumsum().plot(ax=ax)
    ax.set_title("Cumulative log return")
    ax.set_ylabel("Cumulative log return")
    fig.savefig(png_path)
    plt.close(fig)


def _render_markdown(returns: pd.Series, png_name: str) -> str:
    performance_stats = {
        "total_return": stats.total_return(returns),
        "annualized_return": stats.annualized_return(returns),
        "annualized_volatility": stats.annualized_volatility(returns),
        "sharpe_ratio": stats.sharpe_ratio(returns),
        "max_drawdown": stats.max_drawdown(returns),
    }
    rows = "\n".join(f"| {name} | {value:.4f} |" for name, value in performance_stats.items())
    return (
        "# Backtest report\n\n"
        "| stat | value |\n"
        "| --- | --- |\n"
        f"{rows}\n\n"
        f"![Cumulative return]({png_name})\n"
    )
