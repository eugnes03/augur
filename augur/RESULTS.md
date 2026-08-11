# Results

The default configuration (`scripts/run_backtest.py`) runs a dollar-neutral
long/short strategy combining trailing momentum and short-term reversal
signals across a fixed 10-name Nordic equity universe over 2024, rebalanced
daily, with long and short legs picked from the momentum+reversal composite
score each day. The pipeline enforces point-in-time correctness and
lookahead-safe alignment throughout: signals are computed only from data
known as of the decision date, and a day's weights are shifted forward one
day before being priced against returns, so no decision ever sees a return
it wouldn't have known yet. That said, at this sample size, with 10 tickers,
roughly 240 trading days, and a single naive chronological in-sample split,
the resulting Sharpe ratios are not statistically distinguishable from
noise: the parameter sensitivity sweep in `scripts/param_sweep.py`, which
re-runs the backtest across a grid of lookback and selection-size choices,
shows performance swinging substantially across neighboring, equally
plausible parameter choices, which is the signature of a result driven by
sampling noise rather than a robust edge. This should be read as a working,
correctly-wired research pipeline, not a validated trading strategy.
