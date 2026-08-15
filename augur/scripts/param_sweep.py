import logging
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from augur import report, stats
from augur.backtest import BacktestResult, run_backtest
from augur.config import Config

_REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

_IN_SAMPLE_START = "2024-01-01"
_IN_SAMPLE_END = "2024-08-31"
_LOOKBACKS = [10, 20, 40]
_SELECTION_SIZES = [2, 3, 4]

_base_config = Config(
    start=_IN_SAMPLE_START,
    end=_IN_SAMPLE_END,
    momentum_lookback=20,
    reversal_lookback=5,
    n_long=3,
    n_short=3,
)


def _configs() -> list[Config]:
    """Every (lookback, selection size) grid point: this run's actual trial count."""
    return [
        replace(_base_config, momentum_lookback=lookback, n_long=n, n_short=n)
        for lookback in _LOOKBACKS
        for n in _SELECTION_SIZES
    ]


def _grid_row(config: Config, result: BacktestResult) -> dict[str, float]:
    """Summarize one already-run grid point's Sharpe/turnover/drawdown."""
    return {
        "lookback": config.momentum_lookback,
        "n_long_short": config.n_long,
        "sharpe_ratio": stats.sharpe_ratio(result.returns),
        "avg_turnover": stats.turnover(result.weights).mean(),
        "max_drawdown": stats.max_drawdown(result.returns),
    }


def run_sweep() -> tuple[pd.DataFrame, list[BacktestResult]]:
    """
    Run every grid point in-sample once, returning both the summary table and each
    point's raw BacktestResult (the latter feeds the best config's deflated Sharpe).
    """
    configs = _configs()
    results = [run_backtest(config) for config in configs]
    grid = pd.DataFrame(
        _grid_row(config, result) for config, result in zip(configs, results, strict=True)
    )
    return grid, results


def _render_markdown(grid: pd.DataFrame) -> str:
    header = "| " + " | ".join(grid.columns) + " |"
    separator = "| " + " | ".join("---" for _ in grid.columns) + " |"
    rows = "\n".join(
        "| " + " | ".join(f"{value:g}" for value in row) + " |"
        for row in grid.itertuples(index=False)
    )
    return f"# Parameter sensitivity sweep (in-sample only)\n\n{header}\n{separator}\n{rows}\n"


def _period_sharpe_ratios(results: list[BacktestResult]) -> np.ndarray:
    """
    Per-period (non-annualized) Sharpe of each grid point's returns: the units
    deflated_sharpe_ratio() and its inputs are defined in, unlike the annualized
    "sharpe_ratio" column in the sensitivity table above.
    """
    return np.array([result.returns.mean() / result.returns.std() for result in results])


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    grid, results = run_sweep()
    path = _REPORTS_DIR / f"{date.today()}-param-sweep.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown(grid))

    best = int(grid["sharpe_ratio"].idxmax())
    best_result = results[best]
    trial_sharpe_ratios = _period_sharpe_ratios(results)
    report.write_report(
        best_result.returns,
        best_result.weights,
        _REPORTS_DIR / f"{date.today()}-param-sweep-best.md",
        trial_sharpe_ratios,
    )


if __name__ == "__main__":
    main()
