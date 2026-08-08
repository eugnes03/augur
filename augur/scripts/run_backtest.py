import logging
from datetime import date
from pathlib import Path

from augur import report
from augur.backtest import run_backtest
from augur.config import Config

_REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

config = Config(
    start="2024-01-01",
    end="2024-12-31",
    momentum_lookback=20,
    reversal_lookback=5,
    n_long=3,
    n_short=3,
)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    result = run_backtest(config)
    report.write_report(result.returns, result.weights, _REPORTS_DIR / f"{date.today()}.md")


if __name__ == "__main__":
    main()
