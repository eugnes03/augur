"""Static research universe used by ingest and downstream signal code."""

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
    Return the research universe as yfinance-compatible ticker strings.
    Static and hardcoded for now, deliberately -- no point-in-time
    membership or survivorship-bias handling.
    """
    return NORDIC_UNIVERSE.copy()
