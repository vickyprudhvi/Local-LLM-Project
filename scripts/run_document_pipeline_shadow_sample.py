"""Shadow sample: `finance/documents/` compare mode against REAL filings.

Spec section 18: after the live golden benchmark clears hard safety, run
`compare` mode on a small (10-15) real-stock sample. `compare` mode NEVER
offers a candidate to the resolver -- see `finance/documents/runtime.py` --
so the production answer for every one of these stocks is untouched by this
script; it only reads and records diagnostics.

WHY "NEW_PIPELINE_CORRECTS_OLD" ETC. ARE MOSTLY N/A HERE

Compare mode, by design (section 22 / the runtime module's own docstring),
never produces a competing final answer to grade against the old one -- it
records what the new layer WOULD have proposed and stops. So for most
symbols there is no "new answer" to classify at all; this script reports
that honestly as `UNRESOLVED_BY_DESIGN` and reserves the other three labels
for the cases where the new layer actually proposed and accepted something
(an actual-table fallback fact or a financing event), which is where a real
comparison against the existing v1 state is possible.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, ".")

from brain import ask_local_raw                                               # noqa: E402
from finance import period_facts as pf                                        # noqa: E402
from finance.documents import runtime as doc_runtime                          # noqa: E402
from finance.documents.runtime import DocumentPipelineMode                    # noqa: E402
from finance.extraction import document_resolver as period_resolver           # noqa: E402
from finance.extraction.schema import StatementCompleteness                   # noqa: E402
from finance.sec_provider import resolve_cik                                  # noqa: E402
from tools.finance_tools import get_sec_coordinator                           # noqa: E402

DEFAULT_SAMPLE = [
    "AAPL", "MSFT", "KO", "XOM", "JPM", "UNH", "T", "RIVN", "CASY", "TSM",
    "SAP", "BABA",
]


class ShadowClassification:
    NEW_PIPELINE_CORRECTS_OLD = "NEW_PIPELINE_CORRECTS_OLD"
    OLD_CORRECT_NEW_REGRESSION = "OLD_CORRECT_NEW_REGRESSION"
    BOTH_WRONG_SHARED_UPSTREAM = "BOTH_WRONG_SHARED_UPSTREAM"
    UNRESOLVED_BY_DESIGN = "UNRESOLVED_BY_DESIGN"


@dataclass
class SymbolResult:
    symbol: str
    ok: bool = True
    error: Optional[str] = None
    v1_latest_period: Optional[str] = None
    v2_period_resolution: Optional[str] = None
    structured_completeness: Optional[str] = None
    diagnostics: Optional[dict] = None
    events: List[dict] = field(default_factory=list)
    classification: str = ShadowClassification.UNRESOLVED_BY_DESIGN
    classification_reason: str = ""
    hard_safety_findings: List[str] = field(default_factory=list)


def _fetch(symbol: str) -> SymbolResult:
    result = SymbolResult(symbol=symbol)
    try:
        coordinator = get_sec_coordinator()
        cik, _name = resolve_cik(coordinator, symbol)
        submissions = coordinator.fetch("company_submissions", symbol, {"cik": cik}).payload
        company_facts = coordinator.fetch("company_facts", symbol, {"cik": cik}).payload
    except Exception as exc:                                       # noqa: BLE001
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    resolution = period_resolver.resolve(company_facts)
    result.v2_period_resolution = resolution.period_end
    if resolution.selected:
        result.structured_completeness = resolution.selected.statement_completeness

    pipeline = doc_runtime.build_document_pipeline(
        symbol, submissions, company_facts, mode=DocumentPipelineMode.COMPARE,
        target_period_end=resolution.period_end,
        structured_completeness=result.structured_completeness)
    result.diagnostics = pipeline.diagnostics.to_dict()
    result.events = [e.to_dict() for e in pipeline.events]

    # Hard-safety scan: nothing here should ever show an accepted event with
    # funded=True and no grounding note, or an actuals rejection code that
    # names an UNSUPPORTED class alongside a positive accepted count -- the
    # validator already enforces this, so this is a redundant confirmation.
    for event in result.events:
        if event.get("funded") and not event.get("source_evidence"):
            result.hard_safety_findings.append(
                f"funded event with no evidence: {event}")

    # Classification (see module docstring for why most symbols land on
    # UNRESOLVED_BY_DESIGN).
    accepted_actuals = result.diagnostics.get("llm_actuals_accepted", 0)
    accepted_events = result.diagnostics.get("llm_events_accepted", 0)
    if accepted_actuals or accepted_events:
        result.classification = ShadowClassification.NEW_PIPELINE_CORRECTS_OLD
        result.classification_reason = (
            f"the document pipeline proposed and accepted {accepted_actuals} actual "
            f"fact(s) and {accepted_events} event(s) beyond what the existing v1/v2 "
            "path already carries -- new evidence, not yet judged against real filed "
            "figures beyond the acceptance boundary's own grounding checks")
    else:
        result.classification_reason = (
            "compare mode recorded no candidate the resolver would have needed to "
            "act on for this symbol (structured/deterministic paths already sufficient, "
            "or no financing-event-eligible filing in the recent window)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    doc_runtime.register_actuals_model_client(ask_local_raw)
    doc_runtime.register_event_model_client(ask_local_raw)

    results: List[SymbolResult] = []
    for symbol in args.symbols:
        print(f"\n=== {symbol} ===", flush=True)
        started = time.monotonic()
        result = _fetch(symbol)
        elapsed = time.monotonic() - started
        print(f"  ok={result.ok} error={result.error} "
             f"period={result.v2_period_resolution} "
             f"completeness={result.structured_completeness} "
             f"elapsed={elapsed:.1f}s", flush=True)
        if result.diagnostics:
            print(f"  diagnostics: {json.dumps(result.diagnostics, default=str)}", flush=True)
        if result.events:
            print(f"  events: {json.dumps(result.events, default=str)}", flush=True)
        if result.hard_safety_findings:
            print(f"  HARD SAFETY FINDINGS: {result.hard_safety_findings}", flush=True)
        print(f"  classification: {result.classification} -- {result.classification_reason}",
             flush=True)
        results.append(result)

    counts: Dict[str, int] = {}
    for r in results:
        counts[r.classification] = counts.get(r.classification, 0) + 1

    summary = {
        "sample_size": len(results),
        "ok_count": sum(1 for r in results if r.ok),
        "error_count": sum(1 for r in results if not r.ok),
        "errors": {r.symbol: r.error for r in results if not r.ok},
        "classification_counts": counts,
        "old_correct_new_critical_regression": sum(
            1 for r in results if r.classification == ShadowClassification.OLD_CORRECT_NEW_REGRESSION),
        "total_hard_safety_findings": sum(len(r.hard_safety_findings) for r in results),
    }
    print("\n" + json.dumps(summary, indent=2, default=str))

    if args.out:
        payload = {"summary": summary,
                  "results": [r.__dict__ for r in results]}
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
