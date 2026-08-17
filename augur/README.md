# augur

A small quantitative research framework for backtesting a cross-sectional
long/short equity strategy, built with point-in-time correctness and
lookahead-safe alignment as first-class concerns rather than afterthoughts.

## Pipeline

```
config
  |
  v
universe/schemas -> ingest -> returns/signals -> portfolio/pnl -> stats/report
```

- **config**: a single `Config` dataclass holding every backtest parameter
  (date range, lookbacks, selection sizes, universe). Constructed once per
  run and threaded through the rest of the pipeline.
- **universe/schemas**: the static ticker universe and the `pandera`
  schemas (`BarSchema`, `PanelBarSchema`) that every OHLCV frame is
  validated against.
- **ingest**: fetches and caches daily bars per ticker (via `yfinance`),
  normalizes them to the schema, and stacks them into one long panel.
- **returns/signals**: trailing-momentum and short-term-reversal signals
  computed per ticker, combined into one cross-sectional score via
  z-scoring.
- **portfolio/pnl**: turns the combined signal into dollar-neutral
  long/short weights and computes the resulting daily strategy return.
- **costs**: a linear transaction cost model. Applies a cost-per-bps drag
  on turnover, so gross and net returns can be compared side by side.
- **stats/report**: summary statistics (Sharpe, drawdown, turnover) and a
  markdown report of a run.
- **risk**: a standalone factor risk model. Decomposes the return
  correlation matrix into statistical factors with PCA. It's a static
  single-window fit and isn't wired into `run_backtest`, so treat it as
  a diagnostic you run separately, not a live risk check.

`backtest.py` is the orchestrator. It wires these stages together into
`run_backtest(config)` and owns the lookahead-safe alignment between a
day's decided weights and the return realized on the following day.

Underneath all of this, the goal is to test trading ideas without lying
to yourself about performance. A backtest can look better than it would
in live trading in a lot of quiet ways. Weights that see tomorrow's
return. A universe that only contains today's survivors. Costs that
never get charged. A parameter sweep whose best result gets reported as
if it were the only one tried. Each stage here exists to close off one
of those.

## Running it

```
make check   # ruff check, mypy --strict, pytest
make run     # runs scripts/run_backtest.py, writes a report to reports/
```

## Deferred / out of scope

This is a research scaffold, not a production trading system. Deliberately
not yet implemented:

- Slippage beyond the linear cost model in `costs.py`
- Position limits and sector constraints
- A point-in-time risk model. `risk.py` fits statistical factors over one
  fixed window; it doesn't roll or refit over time
- Purged cross-validation for parameter selection
- Experiment tracking
- Additional data providers beyond `yfinance`
- Point-in-time universe membership (survivorship bias is not handled)
- Live execution
- Automated parameter optimization (the current sweep in
  `scripts/param_sweep.py` is diagnostic, not an optimizer)
