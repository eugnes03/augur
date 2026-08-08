import logging
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from augur import stats
from augur.backtest import run_backtest
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


def _grid_row(config: Config) -> dict[str, float]:
    """Run one grid point in-sample and summarize Sharpe/turnover/drawdown."""
    result = run_backtest(config)
    return {
        "lookback": config.momentum_lookback,
        "n_long_short": config.n_long,
        "sharpe_ratio": stats.sharpe_ratio(result.returns),
        "avg_turnover": stats.turnover(result.weights).mean(),
        "max_drawdown": stats.max_drawdown(result.returns),
    }


def run_sweep() -> pd.DataFrame:
    """Tabulate Sharpe/turnover/max-drawdown across a lookback x selection-size grid, in-sample."""
    rows = [
        _grid_row(replace(_base_config, momentum_lookback=lookback, n_long=n, n_short=n))
        for lookback in _LOOKBACKS
        for n in _SELECTION_SIZES
    ]
    return pd.DataFrame(rows)


def _render_markdown(grid: pd.DataFrame) -> str:
    header = "| " + " | ".join(grid.columns) + " |"
    separator = "| " + " | ".join("---" for _ in grid.columns) + " |"
    rows = "\n".join(
        "| " + " | ".join(f"{value:g}" for value in row) + " |"
        for row in grid.itertuples(index=False)
    )
    return f"# Parameter sensitivity sweep (in-sample only)\n\n{header}\n{separator}\n{rows}\n"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    grid = run_sweep()
    path = _REPORTS_DIR / f"{date.today()}-param-sweep.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown(grid))


if __name__ == "__main__":
    main()
