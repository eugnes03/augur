import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def statistical_factors(
    returns: pd.DataFrame, n_factors: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    PCA-decompose a (date x ticker) returns panel into `n_factors` statistical
    factors via eigendecomposition of the return correlation matrix.

    Static single-window fit -- NOT point-in-time correct. A deliberate first
    pass; do not use this to explain risk at dates before the window ends.

    `returns` is expected to already be a wide (date x ticker) panel of plain
    daily returns, as opposed to backtest.py's _forward_returns() panel which
    is shifted forward for use as a trading target. The shift semantics differ
    enough (risk wants the return realized *in* the window, not the one
    following it) that this module builds no shared unstack helper with
    backtest.py -- callers are expected to hand in an already-wide panel.

    Returns:
    - loadings: (ticker x factor), each asset's exposure to each factor
    - factor_returns: (date x factor), the factor return time series
    - explained_variance_ratio: length-n_factors Series, fraction of total
      variance each factor explains
    """
    clean_returns = _drop_tickers_with_nan(returns)
    standardized = (clean_returns - clean_returns.mean()) / clean_returns.std()

    correlation = standardized.corr()
    eigenvalues, eigenvectors = np.linalg.eigh(correlation.to_numpy())
    eigenvalues, eigenvectors = _sort_descending(eigenvalues, eigenvectors)

    top_eigenvalues = eigenvalues[:n_factors]
    top_eigenvectors = _normalize_sign(eigenvectors[:, :n_factors])

    factor_columns = [f"factor_{i}" for i in range(n_factors)]
    loadings = pd.DataFrame(
        top_eigenvectors, index=clean_returns.columns, columns=factor_columns
    )
    factor_returns = pd.DataFrame(
        standardized.to_numpy() @ top_eigenvectors,
        index=clean_returns.index,
        columns=factor_columns,
    )
    explained_variance_ratio = pd.Series(
        top_eigenvalues / eigenvalues.sum(), index=factor_columns
    )
    return loadings, factor_returns, explained_variance_ratio


def _drop_tickers_with_nan(returns: pd.DataFrame) -> pd.DataFrame:
    """Complete-case only: drop any ticker with a NaN anywhere in the window."""
    tickers_with_nan = returns.columns[returns.isna().any()]
    if len(tickers_with_nan):
        logger.warning(
            "Dropping %d/%d tickers with a NaN in the window: %s",
            len(tickers_with_nan),
            returns.shape[1],
            list(tickers_with_nan),
        )
    return returns.drop(columns=tickers_with_nan)


def _sort_descending(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """np.linalg.eigh returns eigenvalues ascending; factors are conventionally largest-first."""
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _normalize_sign(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Eigenvectors are only defined up to a sign flip. Fix each factor's sign so
    its largest-magnitude loading is positive, for reproducible results.
    """
    largest_magnitude_row = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[largest_magnitude_row, range(eigenvectors.shape[1])])
    return eigenvectors * signs
