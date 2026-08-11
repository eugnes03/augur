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
- **stats/report**: summary statistics (Sharpe, drawdown, turnover) and a
  markdown report of a run.

`backtest.py` is the orchestrator: it wires these stages together into
`run_backtest(config)` and owns the lookahead-safe alignment between a
day's decided weights and the return realized on the following day.

## Running it

```
make check   # ruff check, mypy --strict, pytest
make run     # runs scripts/run_backtest.py, writes a report to reports/
```

## Deferred / out of scope

This is a research scaffold, not a production trading system. Deliberately
not yet implemented:

- Transaction costs / slippage
- Risk models (factor exposure, position limits, sector constraints)
- Purged cross-validation for parameter selection
- Experiment tracking
- Additional data providers beyond `yfinance`
- Point-in-time universe membership (survivorship bias is not handled)
- Live execution
- Automated parameter optimization (the current sweep in
  `scripts/param_sweep.py` is diagnostic, not an optimizer)
