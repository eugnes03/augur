import logging
from datetime import date
from pathlib import Path

from augur import report
from augur.backtest import benchmark_returns, run_backtest
from augur.config import Config
from augur.costs import net_of_costs

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
    net_returns = net_of_costs(result.returns, result.weights, config.transaction_cost_bps)
    report.write_report(
        result.returns,
        result.weights,
        _REPORTS_DIR / f"{date.today()}.md",
        benchmark_returns=benchmark_returns(config),
        net_returns=net_returns,
    )


if __name__ == "__main__":
    main()
