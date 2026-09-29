"""Phase H.29 spike -- same-filing CURRENT vs EDGARTOOLS comparison.

Runs the existing production SEC retrieval path (`finance.sec_provider.
SecEdgarClient` + `finance.reported_actuals.discovery` + `finance.documents.
text_normalization`, completely unmodified) and the new spike-only
`finance.documents.edgartools_adapter` against the SAME real filings, and
records structured diffs. Never influences production output -- this script
is not imported by any pipeline code, and `tools.config.finance_sec_provider()`
still defaults to 'current' regardless of what this script measures.

Not a pytest test: it makes real network calls to SEC EDGAR (via both
paths) and to whatever EdgarTools itself contacts, exactly like every prior
live-benchmark script in this project (`scripts/run_live_document_pipeline_
benchmark.py`, `scripts/run_document_pipeline_shadow_sample.py`).
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")

import tools.config as config                                                # noqa: E402
from finance.documents import edgartools_adapter as eta                      # noqa: E402
from finance.documents.text_normalization import (                          # noqa: E402
    normalize_sec_document, select_relevant_item_blocks)
from finance.reported_actuals.discovery import (                            # noqa: E402
    find_reported_actual_filings, select_results_document)
from finance.sec_datasets import resolve_sec_dataset                        # noqa: E402
from finance.sec_provider import SecEdgarClient, resolve_cik                # noqa: E402
from tools.base import ToolFailure                                          # noqa: E402


# ---------------------------------------------------------------------------
# The fixed, pre-declared comparison set (spec section 5: "do not choose
# only successful examples" -- this list is committed to BEFORE any run,
# never edited after seeing results). Deliberately excludes JPM/SAP (section
# 10's diagnostic-only carve-out, run separately below) and is a generic,
# diverse mix of large-cap issuers covering different form/item shapes --
# no ticker-specific production logic results from any of this.
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "KO", "XOM", "UNH", "T", "CAT", "GE", "DIS", "NKE",
    "BA", "PG", "JNJ", "V", "HD", "LOW", "IBM", "ORCL", "CSCO", "INTC",
    "TSM", "SONY",
]

JPM_SAP_DIAGNOSTIC_TICKERS = ["JPM", "SAP"]


class FailureClass:
    DISCOVERY = "DISCOVERY"
    ACCESSION_SELECTION = "ACCESSION_SELECTION"
    FORM_SELECTION = "FORM_SELECTION"
    REPORTING_PERIOD = "REPORTING_PERIOD"
    DOCUMENT_FETCH = "DOCUMENT_FETCH"
    SECTION_SELECTION = "SECTION_SELECTION"
    EXHIBIT_DISCOVERY = "EXHIBIT_DISCOVERY"
    HTML_NORMALIZATION = "HTML_NORMALIZATION"
    TABLE_PRESERVATION = "TABLE_PRESERVATION"
    XBRL_FACT_SELECTION = "XBRL_FACT_SELECTION"
    PROVENANCE = "PROVENANCE"
    LATENCY = "LATENCY"
    OTHER_GENERALIZED = "OTHER_GENERALIZED"


class JpmSapClassification:
    EDGARTOOLS_CORRECTS_CURRENT = "EDGARTOOLS_CORRECTS_CURRENT"
    CURRENT_CORRECT_EDGARTOOLS_REGRESSES = "CURRENT_CORRECT_EDGARTOOLS_REGRESSES"
    BOTH_CORRECT = "BOTH_CORRECT"
    BOTH_WRONG_SHARED_SOURCE = "BOTH_WRONG_SHARED_SOURCE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class TickerComparison:
    ticker: str
    ok: bool = True
    error: Optional[str] = None
    findings: List[dict] = field(default_factory=list)
    current_latest_10k: Optional[dict] = None
    edgartools_latest_10k: Optional[dict] = None
    current_latest_8k: Optional[dict] = None
    edgartools_latest_8k: Optional[dict] = None
    deep_dive_accession: Optional[str] = None
    deep_dive_form: Optional[str] = None
    current_items: Optional[str] = None
    edgartools_items: Optional[str] = None
    current_exhibit_count: Optional[int] = None
    edgartools_exhibit_count: Optional[int] = None
    current_normalized_chars: Optional[int] = None
    edgartools_section_chars: Optional[int] = None
    current_section_found: Optional[bool] = None
    edgartools_section_found: Optional[bool] = None
    latency_current_s: Optional[float] = None
    latency_edgartools_s: Optional[float] = None

    def note(self, failure_class: str, detail: str):
        self.findings.append({"class": failure_class, "detail": detail})


def _client() -> SecEdgarClient:
    return SecEdgarClient()


def _fetch_current(client: SecEdgarClient, cik: str, symbol: str, function_id: str,
                   accession: Optional[str] = None, document: Optional[str] = None):
    dataset = resolve_sec_dataset(function_id)
    args = {"symbol": symbol, "cik": cik}
    if accession:
        args["accession"] = accession
    if document:
        args["document"] = document
    return client.fetch(dataset, args)


def _top_recent(submissions: dict, forms: Tuple[str, ...]) -> Optional[dict]:
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    forms_list = recent.get("form") or []
    wanted = {f.upper() for f in forms}
    best = None
    for i, form in enumerate(forms_list):
        if (form or "").upper() not in wanted:
            continue

        def at(key, default=""):
            values = recent.get(key) or []
            return values[i] if i < len(values) else default

        row = {"form": form, "accession": at("accessionNumber"),
               "filed": at("filingDate"), "report_date": at("reportDate"),
               "items": at("items"), "document": at("primaryDocument")}
        if best is None or row["filed"] > best["filed"]:
            best = row
    return best


def compare_one(ticker: str) -> TickerComparison:
    result = TickerComparison(ticker=ticker)
    client = _client()
    try:
        # -------- CURRENT: discovery --------
        t0 = time.monotonic()
        cik, _name = resolve_cik(_ClientCoordinatorShim(client), ticker)
        submissions_resp = _fetch_current(client, cik, ticker, "company_submissions")
        submissions = submissions_resp.payload
        result.latency_current_s = time.monotonic() - t0

        current_10k = _top_recent(submissions, ("10-K",))
        current_8k = _top_recent(submissions, ("8-K", "8-K/A"))
        result.current_latest_10k = current_10k
        result.current_latest_8k = current_8k

        # -------- EDGARTOOLS: discovery --------
        t0 = time.monotonic()
        et_10ks = eta.discover_filings(ticker, form="10-K", limit=1)
        et_8ks = eta.discover_filings(ticker, form="8-K", limit=1)
        result.latency_edgartools_s = time.monotonic() - t0
        result.edgartools_latest_10k = et_10ks[0].to_dict() if et_10ks else None
        result.edgartools_latest_8k = et_8ks[0].to_dict() if et_8ks else None

        if current_10k and et_10ks:
            if current_10k["accession"] != et_10ks[0].accession:
                result.note(FailureClass.ACCESSION_SELECTION,
                            f"10-K accession disagreement: current={current_10k['accession']} "
                            f"edgartools={et_10ks[0].accession}")
            if current_10k.get("report_date") and et_10ks[0].period_of_report and \
                    current_10k["report_date"] != et_10ks[0].period_of_report:
                result.note(FailureClass.REPORTING_PERIOD,
                            f"10-K period disagreement: current={current_10k.get('report_date')} "
                            f"edgartools={et_10ks[0].period_of_report}")
        elif bool(current_10k) != bool(et_10ks):
            result.note(FailureClass.DISCOVERY, f"10-K discovery mismatch: current={bool(current_10k)} "
                                                 f"edgartools={bool(et_10ks)}")

        if current_8k and et_8ks:
            if current_8k["accession"] != et_8ks[0].accession:
                result.note(FailureClass.ACCESSION_SELECTION,
                            f"8-K accession disagreement: current={current_8k['accession']} "
                            f"edgartools={et_8ks[0].accession}")
        elif bool(current_8k) != bool(et_8ks):
            result.note(FailureClass.DISCOVERY, f"8-K discovery mismatch: current={bool(current_8k)} "
                                                 f"edgartools={bool(et_8ks)}")

        # -------- Deep dive: use the current pipeline's own choice of the
        # most informative filing (an 8-K if it carries financing/earnings-
        # eligible items, else the 10-K), fetched by ACCESSION through BOTH
        # paths for a genuinely same-filing comparison. --------
        deep = None
        if current_8k and any(code in (current_8k.get("items") or "")
                              for code in ("1.01", "2.01", "2.02", "2.03", "2.04", "3.02")):
            deep = current_8k
        elif current_10k:
            deep = current_10k
        if deep is None:
            return result

        result.deep_dive_accession = deep["accession"]
        result.deep_dive_form = deep["form"]
        result.current_items = deep.get("items") or ""

        # CURRENT: index + document + normalize
        try:
            index_resp = _fetch_current(client, cik, ticker, "filing_index", accession=deep["accession"])
            index_rows_html = index_resp.payload.get("directory", index_resp.payload)
        except Exception:  # noqa: BLE001 -- filing_index shape varies; fall back to document fetch only
            index_rows_html = None

        primary_document = deep.get("document") or ""
        if primary_document:
            try:
                doc_resp = _fetch_current(client, cik, ticker, "filing_document",
                                          accession=deep["accession"], document=primary_document)
                raw_text = doc_resp.payload.get("document_text") or ""
                normalized = normalize_sec_document(raw_text)
                result.current_normalized_chars = len(normalized.text) if normalized.ok else 0
                if normalized.ok and deep.get("items"):
                    first_item = deep["items"].split(",")[0].strip()
                    blocks = select_relevant_item_blocks(normalized.text, (first_item,), max_blocks=1)
                    result.current_section_found = bool(blocks)
                    if not blocks:
                        result.note(FailureClass.SECTION_SELECTION,
                                    f"current pipeline found no block for item {first_item}")
            except ToolFailure as e:
                result.note(FailureClass.DOCUMENT_FETCH, f"current filing_document failed: {e.message}")
        else:
            result.note(FailureClass.DOCUMENT_FETCH, "current pipeline had no primary document name")

        # EDGARTOOLS: same accession, sections + exhibits + text
        try:
            et_text, _rep = eta.filing_text(ticker, deep["accession"])
            et_sections = eta.filing_sections(ticker, deep["accession"])
            et_exhibits = eta.filing_exhibits(ticker, deep["accession"])
            result.edgartools_section_chars = sum(len(s.text) for s in et_sections.values())
            result.edgartools_exhibit_count = len(et_exhibits)
            et_filing_ref = eta.get_filing(ticker, deep["accession"])
            result.edgartools_items = et_filing_ref.items
            if deep.get("items"):
                first_item = deep["items"].split(",")[0].strip()
                normalized_item_key = first_item.replace(".", "")
                found = any(normalized_item_key in name for name in et_sections)
                result.edgartools_section_found = found
                if not found:
                    result.note(FailureClass.SECTION_SELECTION,
                                f"edgartools found no section named for item {first_item} "
                                f"(has: {list(et_sections)})")
            if not et_text:
                result.note(FailureClass.HTML_NORMALIZATION, "edgartools returned empty filing text")
        except EdgarToolsAdapterErrorAlias as e:  # noqa: F821 -- resolved below
            result.note(FailureClass.OTHER_GENERALIZED, f"edgartools deep-dive failed: {e}")
        except Exception as e:  # noqa: BLE001
            result.note(FailureClass.OTHER_GENERALIZED, f"edgartools deep-dive failed: {type(e).__name__}: {e}")

        if deep.get("items") and result.edgartools_items is not None:
            current_codes = {c.strip() for c in (deep.get("items") or "").split(",") if c.strip()}
            et_codes = {c.strip() for c in (result.edgartools_items or "").split(",") if c.strip()}
            if current_codes and et_codes and current_codes != et_codes:
                result.note(FailureClass.OTHER_GENERALIZED,
                            f"item-code disagreement: current={current_codes} edgartools={et_codes}")

    except ToolFailure as e:
        result.ok = False
        result.error = f"{e.code}: {e.message}"
    except Exception as e:  # noqa: BLE001
        result.ok = False
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return result


# `eta.EdgarToolsAdapterError` isn't importable by that alias above until
# module load completes; resolve the real class for the except-clause.
EdgarToolsAdapterErrorAlias = eta.EdgarToolsAdapterError


class _ClientCoordinatorShim:
    """`resolve_cik` expects a coordinator with `.fetch(dataset_id, symbol)`
    returning an object with `.payload`. This script talks to `SecEdgarClient`
    directly (no cache/coordinator needed for a one-off spike script), so this
    tiny shim adapts ONE call -- the ticker_cik_map lookup -- to that shape
    without pulling in `MarketDataRequestCoordinator` and a cache DB file."""

    def __init__(self, client: SecEdgarClient):
        self._client = client

    def fetch(self, dataset_id: str, _symbol: str):
        dataset = resolve_sec_dataset(dataset_id)
        return self._client.fetch(dataset, {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--jpm-sap", action="store_true",
                        help="Run the JPM/SAP diagnostic comparison instead of the main set.")
    args = parser.parse_args()

    tickers = JPM_SAP_DIAGNOSTIC_TICKERS if args.jpm_sap else args.tickers

    results: List[TickerComparison] = []
    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        r = compare_one(ticker)
        results.append(r)
        print(f"  ok={r.ok} error={r.error}")
        print(f"  deep_dive={r.deep_dive_accession} ({r.deep_dive_form})")
        print(f"  findings={r.findings}")
        time.sleep(0.3)  # a light, deliberate pace against SEC's own servers

    ok_count = sum(1 for r in results if r.ok)
    all_findings_by_class: Dict[str, int] = {}
    for r in results:
        for f in r.findings:
            all_findings_by_class[f["class"]] = all_findings_by_class.get(f["class"], 0) + 1

    summary = {
        "attempted": len(tickers),
        "ok": ok_count,
        "errors": [{"ticker": r.ticker, "error": r.error} for r in results if not r.ok],
        "findings_by_class": all_findings_by_class,
        "latency_current_mean_s": (sum(r.latency_current_s for r in results if r.latency_current_s) /
                                   max(1, sum(1 for r in results if r.latency_current_s))),
        "latency_edgartools_mean_s": (sum(r.latency_edgartools_s for r in results if r.latency_edgartools_s) /
                                      max(1, sum(1 for r in results if r.latency_edgartools_s))),
        "per_ticker": [
            {
                "ticker": r.ticker, "ok": r.ok, "error": r.error,
                "current_latest_10k": r.current_latest_10k,
                "edgartools_latest_10k": r.edgartools_latest_10k,
                "current_latest_8k": r.current_latest_8k,
                "edgartools_latest_8k": r.edgartools_latest_8k,
                "deep_dive_accession": r.deep_dive_accession,
                "deep_dive_form": r.deep_dive_form,
                "current_items": r.current_items, "edgartools_items": r.edgartools_items,
                "current_exhibit_count": r.current_exhibit_count,
                "edgartools_exhibit_count": r.edgartools_exhibit_count,
                "current_normalized_chars": r.current_normalized_chars,
                "edgartools_section_chars": r.edgartools_section_chars,
                "current_section_found": r.current_section_found,
                "edgartools_section_found": r.edgartools_section_found,
                "latency_current_s": r.latency_current_s,
                "latency_edgartools_s": r.latency_edgartools_s,
                "findings": r.findings,
            }
            for r in results
        ],
    }

    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "per_ticker"}, indent=2, default=str))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
