from datetime import date
from pathlib import Path

from augur import report
from augur.backtest import BacktestConfig, run_backtest

_REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

config = BacktestConfig(
    start="2024-01-01",
    end="2024-12-31",
    momentum_lookback=20,
    reversal_lookback=5,
    n_long=3,
    n_short=3,
)


def main() -> None:
    daily_returns = run_backtest(config)
    report.write_report(daily_returns, _REPORTS_DIR / f"{date.today()}.md")


if __name__ == "__main__":
    main()
