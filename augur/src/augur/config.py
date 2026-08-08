from dataclasses import dataclass


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
