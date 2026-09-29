"""Phase H.3 — the closed set of SEC EDGAR datasets this project will request.

Mirrors finance/datasets.py's pattern exactly (same Dataset shape, same
fail-closed resolve_*, same CACHE_SCHEMA_VERSION convention) so the SHARED
coordinator/cache/quota machinery works identically for SEC as it already
does for Alpha Vantage — see finance/coordinator.py's provider_id/
dataset_resolver injection.

Three datasets, no generic SEC URL fetcher:

* ticker_cik_map  — the whole ticker->CIK file, ALWAYS requested under the
  constant symbol "GLOBAL" (see finance/sec_provider.py::resolve_cik) so
  resolving 100 different tickers costs exactly one external call, not 100.
* company_submissions / company_facts — require "cik" as an explicit,
  cache-key-significant argument, resolved by the CALLER first (never
  guessed, never re-derived from a stale mapping) — see
  finance/sec_provider.py.
"""

from dataclasses import dataclass, field
from typing import Tuple

from finance.datasets import Dataset, Freshness
from tools.base import ToolFailure
from tools.models import MARKET_DATA_UNSUPPORTED_DATASET

_SEC_DATASETS = {
    "ticker_cik_map": Dataset(
        dataset_id="ticker_cik_map",
        function="company_tickers",
        freshness=Freshness.REFERENCE,
        ttl_seconds=7 * 24 * 3600,
        argument_names=(),
        required_arguments=(),
        label="SEC ticker-to-CIK mapping",
    ),
    "company_submissions": Dataset(
        dataset_id="company_submissions",
        function="submissions",
        freshness=Freshness.PERIODIC,
        ttl_seconds=6 * 3600,
        argument_names=("symbol", "cik"),
        required_arguments=("symbol", "cik"),
        label="SEC filing submissions",
    ),
    "company_facts": Dataset(
        dataset_id="company_facts",
        function="companyfacts",
        freshness=Freshness.PERIODIC,
        ttl_seconds=24 * 3600,
        argument_names=("symbol", "cik"),
        required_arguments=("symbol", "cik"),
        label="SEC XBRL company facts",
    ),
    # Phase H.4 -- the two datasets that make SEC-FILED management guidance
    # reachable. Quantitative guidance ("we expect full-year 2026 EPS of
    # $x.xx to $y.yy") is never in XBRL company facts; it lives in the
    # earnings-release exhibit attached to an item-2.02 8-K.
    #
    # These are still a CLOSED set, not a generic URL fetcher: both take an
    # accession number that the CALLER must have obtained from
    # company_submissions, both are pinned to the EDGAR Archives path for one
    # specific filing, and `filing_document` additionally constrains the
    # document name (see finance/sec_provider.py::_build_url). There is no
    # argument through which an arbitrary URL, host or path can be reached.
    "filing_index": Dataset(
        dataset_id="filing_index",
        function="filing_index",
        freshness=Freshness.PERIODIC,
        # A filed document is immutable once accepted, so this can be cached
        # hard; only the LIST of filings changes, and that lives in
        # company_submissions with its own 6-hour TTL.
        ttl_seconds=30 * 24 * 3600,
        argument_names=("symbol", "cik", "accession"),
        required_arguments=("symbol", "cik", "accession"),
        label="SEC filing document index",
    ),
    "filing_document": Dataset(
        dataset_id="filing_document",
        function="filing_document",
        freshness=Freshness.PERIODIC,
        ttl_seconds=30 * 24 * 3600,
        argument_names=("symbol", "cik", "accession", "document"),
        required_arguments=("symbol", "cik", "accession", "document"),
        label="SEC filing document",
    ),
}

ALL_SEC_DATASET_IDS = tuple(sorted(_SEC_DATASETS))


def resolve_sec_dataset(dataset_id) -> Dataset:
    """Fail-closed lookup, exactly like finance.datasets.resolve_dataset —
    an unreviewed SEC dataset id never becomes a request."""
    dataset = _SEC_DATASETS.get(dataset_id) if isinstance(dataset_id, str) else None
    if dataset is None:
        raise ToolFailure(
            MARKET_DATA_UNSUPPORTED_DATASET,
            f"{dataset_id!r} is not a supported SEC EDGAR dataset.",
        )
    return dataset


def describe_all_sec() -> Tuple[dict, ...]:
    return tuple(_SEC_DATASETS[d].describe() for d in ALL_SEC_DATASET_IDS)
