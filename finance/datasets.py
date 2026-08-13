"""Phase H.1 — the closed set of market-data datasets this project will request.

A dataset is the ONLY thing the rest of the system names. It maps to exactly one
reviewed Alpha Vantage `function`, carries its own freshness class and TTL, and
declares which arguments are meaningful for cache-key normalization. Nothing
outside this table can be requested: `resolve_dataset` fails closed, so neither
the local LLM nor a workflow can reach an unreviewed provider endpoint.

Freshness classes are deliberate and conservative. `DELAYED` means exactly that —
Alpha Vantage's free tier returns end-of-day / delayed data, so nothing here is
labelled REALTIME. A dataset is only ever described as realtime when the provider
response AND the account entitlement establish it, which this project does not
currently verify; see `finance/normalization.py` for how freshness is reported.
"""

from dataclasses import dataclass, field
from typing import Tuple

from tools.base import ToolFailure
from tools.models import MARKET_DATA_UNSUPPORTED_DATASET

# Bumped whenever the stored payload shape or normalization contract changes, so
# older records are invalidated rather than silently reinterpreted.
CACHE_SCHEMA_VERSION = 1


class Freshness:
    DELAYED = "delayed"          # intraday-ish but not entitlement-verified realtime
    END_OF_DAY = "end_of_day"    # settled daily bars
    PERIODIC = "periodic"        # filings-driven; changes only on a reporting event
    REFERENCE = "reference"      # slow-moving descriptive data
    STREAMING = "streaming"      # news and similar continuously-appended feeds


@dataclass(frozen=True)
class Dataset:
    """One reviewed provider endpoint."""

    dataset_id: str
    function: str
    freshness: str
    ttl_seconds: int
    # Argument names that are part of the request AND the cache key, in a fixed
    # order. Anything not listed is rejected before a request is built.
    argument_names: Tuple[str, ...] = ()
    required_arguments: Tuple[str, ...] = ("symbol",)
    defaults: dict = field(default_factory=dict)
    # A short, non-sensitive label used in user-facing freshness reporting.
    label: str = ""

    def describe(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "function": self.function,
            "freshness": self.freshness,
            "ttl_seconds": self.ttl_seconds,
            "label": self.label,
        }


_DATASETS = {
    "stock_quote": Dataset(
        dataset_id="stock_quote",
        function="GLOBAL_QUOTE",
        freshness=Freshness.DELAYED,
        ttl_seconds=15 * 60,
        argument_names=("symbol",),
        label="latest quote",
    ),
    # TIME_SERIES_DAILY is available on the free tier. Its ADJUSTED sibling is a
    # premium endpoint and returns MARKET_DATA_ENTITLEMENT_REQUIRED on a free
    # key, so it is a separate, opt-in dataset rather than the default.
    "daily_prices": Dataset(
        dataset_id="daily_prices",
        function="TIME_SERIES_DAILY",
        freshness=Freshness.END_OF_DAY,
        ttl_seconds=12 * 3600,
        argument_names=("symbol", "outputsize"),
        defaults={"outputsize": "compact"},
        label="daily price history",
    ),
    "daily_prices_adjusted": Dataset(
        dataset_id="daily_prices_adjusted",
        function="TIME_SERIES_DAILY_ADJUSTED",
        freshness=Freshness.END_OF_DAY,
        ttl_seconds=12 * 3600,
        argument_names=("symbol", "outputsize"),
        defaults={"outputsize": "compact"},
        label="split/dividend-adjusted daily price history",
    ),
    "intraday_prices": Dataset(
        dataset_id="intraday_prices",
        function="TIME_SERIES_INTRADAY",
        freshness=Freshness.DELAYED,
        ttl_seconds=15 * 60,
        argument_names=("symbol", "interval", "outputsize"),
        defaults={"interval": "60min", "outputsize": "compact"},
        label="intraday prices",
    ),
    "company_overview": Dataset(
        dataset_id="company_overview",
        function="OVERVIEW",
        freshness=Freshness.REFERENCE,
        ttl_seconds=7 * 24 * 3600,
        argument_names=("symbol",),
        label="company overview",
    ),
    "income_statement": Dataset(
        dataset_id="income_statement",
        function="INCOME_STATEMENT",
        freshness=Freshness.PERIODIC,
        ttl_seconds=30 * 24 * 3600,
        argument_names=("symbol",),
        label="income statements",
    ),
    "balance_sheet": Dataset(
        dataset_id="balance_sheet",
        function="BALANCE_SHEET",
        freshness=Freshness.PERIODIC,
        ttl_seconds=30 * 24 * 3600,
        argument_names=("symbol",),
        label="balance sheets",
    ),
    "cash_flow": Dataset(
        dataset_id="cash_flow",
        function="CASH_FLOW",
        freshness=Freshness.PERIODIC,
        ttl_seconds=30 * 24 * 3600,
        argument_names=("symbol",),
        label="cash-flow statements",
    ),
    "earnings": Dataset(
        dataset_id="earnings",
        function="EARNINGS",
        freshness=Freshness.PERIODIC,
        ttl_seconds=7 * 24 * 3600,
        argument_names=("symbol",),
        label="earnings history",
    ),
    "news": Dataset(
        dataset_id="news",
        function="NEWS_SENTIMENT",
        freshness=Freshness.STREAMING,
        ttl_seconds=3600,
        argument_names=("symbol", "limit"),
        defaults={"limit": "20"},
        label="company news",
    ),
    "symbol_search": Dataset(
        dataset_id="symbol_search",
        function="SYMBOL_SEARCH",
        freshness=Freshness.REFERENCE,
        ttl_seconds=30 * 24 * 3600,
        argument_names=("keywords",),
        required_arguments=("keywords",),
        label="symbol search",
    ),
}

ALL_DATASET_IDS = tuple(sorted(_DATASETS))


def resolve_dataset(dataset_id) -> Dataset:
    """Return the reviewed Dataset, or raise a controlled failure.

    Fail closed: an unreviewed dataset id never becomes a provider request.
    """
    dataset = _DATASETS.get(dataset_id) if isinstance(dataset_id, str) else None
    if dataset is None:
        raise ToolFailure(
            MARKET_DATA_UNSUPPORTED_DATASET,
            f"{dataset_id!r} is not a supported market-data dataset.",
        )
    return dataset


def describe_all() -> Tuple[dict, ...]:
    return tuple(_DATASETS[d].describe() for d in ALL_DATASET_IDS)
