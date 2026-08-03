"""
Defines the static research universe used by ingest and downstream
feature/signal code.

Intentionally simple: a single hardcoded list of yfinance-compatible
ticker strings, with no point-in-time membership or survivorship-bias
handling. See schemas.py and ingest.py for the project's general style —
type-hinted, small, one concept per module.
"""

from typing import Final

NORDIC_UNIVERSE: Final[list[str]] = [
    "VOLV-B.ST",  # Volvo B - Stockholm
    "ERIC-B.ST",  # Ericsson B - Stockholm
    "HM-B.ST",  # H&M B - Stockholm
    "INVE-B.ST",  # Investor B - Stockholm
    "NOVO-B.CO",  # Novo Nordisk B - Copenhagen
    "MAERSK-B.CO",  # Maersk B - Copenhagen
    "EQNR.OL",  # Equinor - Oslo
    "DNB.OL",  # DNB Bank - Oslo
    "NOKIA.HE",  # Nokia - Helsinki
    "SAMPO.HE",  # Sampo A - Helsinki
]


def get_universe() -> list[str]:
    """
    Return the current research universe as a list of yfinance-compatible
    ticker strings.

    This is a static, hardcoded universe for now — no point-in-time
    membership, no survivorship-bias handling. That's a deliberate
    simplification (see project notes); revisit only if/when the
    research shows it matters.
    """
    return NORDIC_UNIVERSE.copy()
