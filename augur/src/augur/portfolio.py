import pandas as pd


def _validate_selection_size(momentum: pd.Series, n_long: int, n_short: int) -> None:
    """
    Raise if n_long + n_short exceeds the number of available tickers.
    Otherwise the long/short index slices overlap and silently break
    the dollar-neutral (+1/-1) invariant with no error.
    """
    n_available = len(momentum)
    if n_long + n_short > n_available:
        raise ValueError(
            f"n_long + n_short ({n_long + n_short}) exceeds available tickers "
            f"({n_available})"
        )


def equal_weight(momentum: pd.Series, n_long: int, n_short: int) -> pd.Series:
    """
    Dollar-neutral equal-weight portfolio from a momentum cross-section.
    Longs the top `n_long` tickers at +1/n_long each, shorts the bottom
    `n_short` at -1/n_short each; raises if n_long + n_short is too large.
    """
    _validate_selection_size(momentum, n_long, n_short)
    # kind="stable": ties keep their input order instead of quicksort's.
    ranked = momentum.sort_values(ascending=False, kind="stable")
    weights = pd.Series(0.0, index=momentum.index)
    weights[ranked.index[:n_long]] = 1.0 / n_long
    weights[ranked.index[-n_short:]] = -1.0 / n_short
    return weights


def rank_weight(momentum: pd.Series, n_long: int, n_short: int) -> pd.Series:
    """
    Dollar-neutral rank-weighted portfolio: longs top `n_long`, shorts
    bottom `n_short`, weight magnitude proportional to rank within each
    side. Raises if n_long + n_short exceeds available tickers.
    """
    _validate_selection_size(momentum, n_long, n_short)
    # kind="stable": ties keep their input order instead of quicksort's.
    ranked = momentum.sort_values(ascending=False, kind="stable")
    weights = pd.Series(0.0, index=momentum.index)

    long_tickers = ranked.index[:n_long]
    long_ranks = pd.Series(range(n_long, 0, -1), index=long_tickers, dtype=float)
    weights[long_tickers] = long_ranks / long_ranks.sum()

    short_tickers = ranked.index[-n_short:]
    short_ranks = pd.Series(range(1, n_short + 1), index=short_tickers, dtype=float)
    weights[short_tickers] = -short_ranks / short_ranks.sum()

    return weights
