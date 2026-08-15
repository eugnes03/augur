"""Research universe used by ingest and downstream signal code."""

from typing import Final

import pandas as pd

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

_NASDAQ_100_WIKI_URL: Final[str] = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
# Wikipedia 403s the default urllib user agent
_WIKI_STORAGE_OPTIONS: Final[dict[str, str]] = {"User-Agent": "augur-research/0.1"}


def _fetch_nasdaq_100_tickers() -> list[str]:
    """Scrape current NASDAQ-100 constituent tickers from Wikipedia's constituents table."""
    tables = pd.read_html(_NASDAQ_100_WIKI_URL, storage_options=_WIKI_STORAGE_OPTIONS)
    tickers: list[str] = tables[0]["Ticker"].astype(str).str.strip().tolist()
    return sorted(tickers)


def get_universe() -> list[str]:
    """
    Return the research universe as yfinance-compatible ticker strings.
    Nordic large caps are static, NASDAQ-100 constituents are scraped live
    from Wikipedia. No point-in-time membership or survivorship-bias
    handling for either leg yet.
    """
    return NORDIC_UNIVERSE + _fetch_nasdaq_100_tickers()
