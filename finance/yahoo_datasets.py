"""Phase H.3 — the closed set of Yahoo Finance (yfinance) datasets this
project will request. Mirrors finance/datasets.py's pattern; see
finance/yahoo_provider.py for how each maps onto a yfinance.Ticker call.
"""

from typing import Tuple

from finance.datasets import Dataset, Freshness
from tools.base import ToolFailure
from tools.models import MARKET_DATA_UNSUPPORTED_DATASET

_YAHOO_DATASETS = {
    "stock_quote": Dataset(
        dataset_id="stock_quote",
        function="fast_info",
        freshness=Freshness.DELAYED,
        ttl_seconds=15 * 60,
        argument_names=("symbol",),
        label="latest quote",
    ),
    "company_profile": Dataset(
        dataset_id="company_profile",
        function="info",
        freshness=Freshness.REFERENCE,
        ttl_seconds=7 * 24 * 3600,
        argument_names=("symbol",),
        label="company profile",
    ),
    "price_history": Dataset(
        dataset_id="price_history",
        function="history",
        freshness=Freshness.END_OF_DAY,
        ttl_seconds=12 * 3600,
        argument_names=("symbol", "period", "interval"),
        defaults={"period": "1y", "interval": "1d"},
        label="daily price history",
    ),
    "corporate_actions": Dataset(
        dataset_id="corporate_actions",
        function="actions",
        freshness=Freshness.PERIODIC,
        ttl_seconds=24 * 3600,
        argument_names=("symbol",),
        label="dividends and splits",
    ),
    "analyst_estimates": Dataset(
        dataset_id="analyst_estimates",
        function="analyst_price_targets",
        freshness=Freshness.PERIODIC,
        ttl_seconds=12 * 3600,
        argument_names=("symbol",),
        label="analyst price targets",
    ),
}

ALL_YAHOO_DATASET_IDS = tuple(sorted(_YAHOO_DATASETS))


def resolve_yahoo_dataset(dataset_id) -> Dataset:
    dataset = _YAHOO_DATASETS.get(dataset_id) if isinstance(dataset_id, str) else None
    if dataset is None:
        raise ToolFailure(
            MARKET_DATA_UNSUPPORTED_DATASET,
            f"{dataset_id!r} is not a supported Yahoo Finance dataset.",
        )
    return dataset


def describe_all_yahoo() -> Tuple[dict, ...]:
    return tuple(_YAHOO_DATASETS[d].describe() for d in ALL_YAHOO_DATASET_IDS)
