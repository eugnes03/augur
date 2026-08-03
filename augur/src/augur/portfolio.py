import pandas as pd


def equal_weight(momentum: pd.Series, n_long: int, n_short: int) -> pd.Series:
    """
    Dollar-neutral equal-weight portfolio from a single-date momentum
    cross-section indexed by ticker.

    Longs the top `n_long` tickers by momentum at +1/n_long each, shorts
    the bottom `n_short` tickers at -1/n_short each, and zero-weights
    everything else. Weights sum to 0.
    """
    ranked = momentum.sort_values(ascending=False)
    weights = pd.Series(0.0, index=momentum.index)
    weights[ranked.index[:n_long]] = 1.0 / n_long
    weights[ranked.index[-n_short:]] = -1.0 / n_short
    return weights


def rank_weight(momentum: pd.Series, n_long: int, n_short: int) -> pd.Series:
    """
    Dollar-neutral rank-weighted portfolio from a single-date momentum
    cross-section indexed by ticker.

    Longs the top `n_long` tickers by momentum, shorts the bottom
    `n_short`, with weight magnitude proportional to rank within each
    side (the most extreme momentum gets the largest weight). Each side
    is normalized to sum to +1 / -1, so total weights sum to 0.
    """
    ranked = momentum.sort_values(ascending=False)
    weights = pd.Series(0.0, index=momentum.index)

    long_tickers = ranked.index[:n_long]
    long_ranks = pd.Series(range(n_long, 0, -1), index=long_tickers, dtype=float)
    weights[long_tickers] = long_ranks / long_ranks.sum()

    short_tickers = ranked.index[-n_short:]
    short_ranks = pd.Series(range(1, n_short + 1), index=short_tickers, dtype=float)
    weights[short_tickers] = -short_ranks / short_ranks.sum()

    return weights
