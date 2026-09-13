from dataclasses import dataclass
from typing import Literal

WeightingScheme = Literal["equal", "rank", "volatility_target"]


@dataclass(frozen=True)
class Config:
    """Single source of truth for backtest parameters, constructed once in the entrypoint."""

    start: str
    end: str
    momentum_lookback: int
    reversal_lookback: int
    n_long: int
    n_short: int
    rebalance_frequency: int = 1
    universe: list[str] | None = None
    # 10 bps one-way: a conservative literature rule-of-thumb for liquid US large-caps
    # (spread + slippage), not measured from this project's own data. Revisit with a
    # data-derived estimate (e.g. a high-low spread proxy) if costs end up load-bearing.
    transaction_cost_bps: float = 10.0
    weighting_scheme: WeightingScheme = "equal"
    # Only used when weighting_scheme == "volatility_target".
    volatility_lookback: int = 20
