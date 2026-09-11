"""Reported Actuals Source Integration against LIVE SEC filings.

WHY A SCRIPT AND NOT A TEST

It reads EDGAR. A suite that must stay fast, offline and deterministic cannot
own it, and a live issuer's answer changes the day it files.

WHAT IT SHOWS

For one ticker, the two things section 27 asks to compare:

    the CompanyFacts-only path    which period can be seen today
    the source-integration path   which period a filed release reports

plus what came out of the release: the document chosen, the facts accepted,
the candidate's completeness, whether it ADVANCES the period, and what the
existing Actualization resolver then does with it.

It changes no configuration and switches no default. The layers are called
directly rather than through the runtime seam, so a run of this script cannot
alter what a real report would produce.

USAGE

    python -m scripts.run_live_reported_actuals AAA BBB
"""

import argparse
import datetime
import sys
from typing import List, Optional


def run_one(ticker: str, as_of: Optional[str] = None, limit: int = 3) -> dict:
    from finance import actualization_runtime as actualization
    from finance.reported_actuals import runtime as reported_actuals
    from finance.reported_actuals.candidates import (
        company_facts_overlay,
        merge_company_facts,
        restrict_to_period,
    )
    from finance.actualization import reconstruct_ttm
    from finance.extraction.document_resolver import discover_reported_actuals
    from finance.sec_provider import resolve_cik
    from tools import finance_tools

    as_of = as_of or datetime.date.today().isoformat()
    coordinator = finance_tools.get_sec_coordinator()
    cik, name = resolve_cik(coordinator, ticker)
    facts = coordinator.fetch("company_facts", ticker, {"cik": cik}).payload or {}
    submissions = coordinator.fetch("company_submissions", ticker,
                                    {"cik": cik}).payload or {}

    companyfacts_periods = sorted(
        {c.period_end for c in discover_reported_actuals(facts) if c.period_end
         and c.period_end <= as_of})
    companyfacts_latest = companyfacts_periods[-1] if companyfacts_periods else None

    result = reported_actuals.discover_release_candidates(
        ticker, submissions, mode="v2", max_filings=limit,
        fetcher=reported_actuals.SecDocumentFetcher(ticker, coordinator, cik),
        companyfacts_latest_period=companyfacts_latest,
        company_facts=facts, as_of=as_of)
    observation = result.observation

    before, _obs = actualization.resolve_actual_state(
        facts, as_of=as_of, mode="v2")
    after, after_obs = actualization.resolve_actual_state(
        facts, as_of=as_of, mode="v2", extra_candidates=result.candidates,
        extra_facts=result.facts_overlay)

    best = max((c for c in result.candidates if c.period_end),
               key=lambda c: c.period_end, default=None)
    ttm_before = reconstruct_ttm(facts, before.period_end) if before.period_end else None
    merged = merge_company_facts(
        facts, restrict_to_period(result.facts_overlay, after.period_end))
    ttm_after = reconstruct_ttm(merged, after.period_end) if after.period_end else None

    return {
        "ticker": ticker, "name": name, "as_of": as_of,
        "companyfacts_latest": companyfacts_latest,
        "releases_detected": observation.releases_detected,
        "filings_refused": observation.filings_refused,
        "documents": [d.get("document_id") for d in observation.documents_selected],
        "tables_read": observation.tables_read,
        "tables_usable": observation.tables_usable,
        "facts_accepted": observation.facts_accepted,
        "candidate_periods": observation.candidate_periods,
        "candidate_completeness": observation.candidate_completeness,
        "values_populated": observation.values_populated,
        "best_period": best.period_end if best else None,
        "best_completeness": best.statement_completeness if best else None,
        "best_currency": best.currency if best else None,
        "best_period_type": best.period_type if best else None,
        "advances_period": observation.advances_period,
        "same_period_overlap": observation.same_period_overlap,
        "rejection_codes": observation.rejection_codes,
        "resolved_before": before.period_end,
        "source_before": (before.selected_primary_source.form
                          if before.selected_primary_source else None),
        "resolved_after": after.period_end,
        "source_after": (after.selected_primary_source.form
                         if after.selected_primary_source else None),
        "status_after": after.state_status,
        "conflicts": [c.metric for c in after.conflicts],
        "superseded": [s.form for s in after.superseded_sources],
        "ttm_before": (f"{ttm_before.reference_period_end} {ttm_before.status}"
                       if ttm_before else None),
        "ttm_after": (f"{ttm_after.reference_period_end} {ttm_after.status}"
                      if ttm_after else None),
        "notes": observation.notes,
    }


def run_historical(ticker: str, limit: int = 3) -> dict:
    """The capability proof, on real archived filings.

    Section 30 asks for a case where a filed earnings-release exhibit reports a
    NEWER period than the periodic filing available at the time. Whether one
    exists TODAY depends on where every issuer happens to sit in its filing
    calendar, so this reconstructs the window that always exists: the weeks
    between a release and the periodic filing that follows it.

    Every CompanyFacts row dated on or after the release's period is removed --
    which is exactly what the payload contained before the 10-Q was filed --
    and the release is then offered to the resolver. Nothing is fabricated: the
    release is the real filed exhibit, and the facts removed are the ones that
    did not yet exist.
    """
    import copy

    from finance import actualization_runtime as actualization
    from finance.actualization import reconstruct_ttm
    from finance.extraction.document_resolver import discover_reported_actuals
    from finance.reported_actuals import runtime as reported_actuals
    from finance.reported_actuals.candidates import (
        merge_company_facts,
        restrict_to_period,
    )
    from finance.sec_provider import resolve_cik
    from tools import finance_tools

    coordinator = finance_tools.get_sec_coordinator()
    cik, name = resolve_cik(coordinator, ticker)
    facts = coordinator.fetch("company_facts", ticker, {"cik": cik}).payload or {}
    submissions = coordinator.fetch("company_submissions", ticker,
                                    {"cik": cik}).payload or {}

    fetcher = reported_actuals.SecDocumentFetcher(ticker, coordinator, cik)
    probe = reported_actuals.discover_release_candidates(
        ticker, submissions, mode="v2", max_filings=limit,
        fetcher=fetcher, company_facts=facts)
    newest = max((c.period_end for c in probe.candidates if c.period_end),
                 default=None)
    if newest is None:
        return {"ticker": ticker, "error": "no reported-actual candidate"}

    # Roll CompanyFacts back to the day before the release's period end: what
    # the payload held before the periodic filing arrived.
    rolled = copy.deepcopy(facts)
    removed = 0
    for _taxonomy, concepts in (rolled.get("facts") or {}).items():
        for _concept, entry in concepts.items():
            for unit, rows in (entry.get("units") or {}).items():
                keep = [r for r in rows if (r.get("end") or "") < newest]
                removed += len(rows) - len(keep)
                entry["units"][unit] = keep

    # Re-read the release against THAT payload, because the overlay defers to
    # any period the base already reports and the base is the rolled-back one.
    result = reported_actuals.discover_release_candidates(
        ticker, submissions, mode="v2", max_filings=limit,
        fetcher=fetcher, company_facts=rolled)
    best = max((c for c in result.candidates if c.period_end),
               key=lambda c: c.period_end, default=None)
    if best is None:
        return {"ticker": ticker, "error": "no reported-actual candidate"}

    from finance import canonical as canonical_module
    from finance.freshness import build_current_financial_state
    from finance.reported_actuals import unified as unified_module

    v1_state = build_current_financial_state(rolled, ticker, valuation_date=best.issued_at)
    before, _o1 = actualization.resolve_actual_state(
        rolled, as_of=best.issued_at, mode="v2")
    after, obs_after = actualization.resolve_actual_state(
        rolled, as_of=best.issued_at, mode="v2",
        prior_state_metrics=unified_module.prior_state_metrics(v1_state),
        extra_candidates=(best,), extra_facts=result.facts_overlay)
    ttm_before = reconstruct_ttm(rolled, before.period_end) if before.period_end else None

    # The FULL canonical-integration path: unify, rebuild the state, reconcile.
    unified = unified_module.build_unified_actual_facts(
        rolled, resolution=after, observation=obs_after,
        facts_overlay=result.facts_overlay, v1_state=v1_state)
    canon_before = canonical_module.build_canonical_evidence(v1_state, historical_metrics={})
    if unified.active:
        state_after = build_current_financial_state(
            unified.company_facts, ticker, valuation_date=best.issued_at)
        unified.reconcile_with_state(state_after)
        canon_after = canonical_module.build_canonical_evidence(state_after, historical_metrics={})
        ttm_after = reconstruct_ttm(unified.company_facts, state_after.financial_as_of)
        from finance.actualization import assess_dcf_base_freshness
        dcf_base = assess_dcf_base_freshness(
            after.period_end,
            state_after.latest_quarterly_period or state_after.latest_annual_period
            or state_after.financial_as_of)
        _q_ends = {}
        for _m in ("revenue", "operating_income", "net_income",
                   "operating_cash_flow", "capital_expenditure"):
            from finance import ttm as _ttm_mod
            _b = _ttm_mod.build_ttm(unified.company_facts, _m,
                                    reference_end=state_after.financial_as_of)
            _q_ends[_m] = [s.partition("..")[2] for s in (_b.quarters_included or ())]
        hard = unified_module.check_hard_safety(
            unified, rebuilt_state=state_after, ttm=ttm_after,
            canonical_evidence=canon_after, dcf_base_assessment=dcf_base,
            ttm_quarter_ends=_q_ends)
    else:
        state_after = v1_state
        canon_after = canon_before
        ttm_after = ttm_before
        dcf_base = None
        hard = {}

    return {
        "ticker": ticker, "name": name, "historical": True,
        "as_of": best.issued_at,
        "release_period": best.period_end,
        "release_form": best.form,
        "release_completeness": best.statement_completeness,
        "release_values": len(best.values or {}),
        "facts_removed": removed,
        "resolved_before": before.period_end,
        "source_before": (before.selected_primary_source.form
                          if before.selected_primary_source else None),
        "resolved_after": after.period_end,
        "source_after": (after.selected_primary_source.form
                         if after.selected_primary_source else None),
        "status_after": after.state_status,
        "advanced": bool(before.period_end and after.period_end
                         and after.period_end > before.period_end),
        "ttm_before": (f"{ttm_before.reference_period_end} {ttm_before.status}"
                       if ttm_before else None),
        "ttm_after": (f"{ttm_after.reference_period_end} {ttm_after.status}"
                      if ttm_after else None),
        "canonical_revenue_before": canon_before.value("revenue"),
        "canonical_revenue_after": canon_after.value("revenue"),
        "state_as_of_before": v1_state.financial_as_of,
        "state_as_of_after": getattr(state_after, "financial_as_of", None),
        "dcf_base_after": (dcf_base.freshness if dcf_base else None),
        "dcf_base_research_valid": (dcf_base.may_be_research_valid if dcf_base else None),
        "unified_active": unified.active,
        "fallback_metrics": list(unified.fallback_metrics()),
        "hard_safety": {k: v for k, v in hard.items()},
    }


def render_historical(rows: List[dict]) -> str:
    lines = []
    for row in rows:
        if row.get("error"):
            lines.append(f"\n{row['ticker']}: {row['error']}")
            continue
        lines.append(f"\n{row['ticker']}  ({row.get('name')})  "
                     f"as of the release date {row['as_of']}")
        lines.append(f"  release            : {row['release_form']} for "
                     f"{row['release_period']} ({row['release_completeness']}, "
                     f"{row['release_values']} values)")
        lines.append(f"  CompanyFacts rows removed (not yet filed): "
                     f"{row['facts_removed']}")
        lines.append(f"  resolver without it: {row['resolved_before']} "
                     f"({row['source_before']})")
        lines.append(f"  resolver with it   : {row['resolved_after']} "
                     f"({row['source_after']}, {row['status_after']})")
        lines.append(f"  PERIOD ADVANCED    : {row['advanced']}")
        lines.append(f"  canonical state    : {row['state_as_of_before']}  ->  "
                     f"{row['state_as_of_after']}")
        lines.append(f"  canonical revenue  : {row['canonical_revenue_before']}  ->  "
                     f"{row['canonical_revenue_after']}")
        lines.append(f"  TTM                : {row['ttm_before']}  ->  "
                     f"{row['ttm_after']}")
        lines.append(f"  DCF base           : {row['dcf_base_after']} "
                     f"(research-valid: {row['dcf_base_research_valid']})")
        lines.append(f"  fallback metrics   : {row['fallback_metrics'] or '-'}")
        lines.append(f"  HARD SAFETY        : "
                     f"{sum(len(v) for v in row['hard_safety'].values())}")
        for counter, details in row["hard_safety"].items():
            for detail in details:
                lines.append(f"      {counter}: {detail}")
    return "\n".join(lines)


def render(rows: List[dict]) -> str:
    lines = []
    for row in rows:
        if row.get("error"):
            lines.append(f"\n{row['ticker']}: ERROR {row['error']}")
            continue
        lines.append(f"\n{row['ticker']}  ({row.get('name') or '?'})   as of {row['as_of']}")
        lines.append(f"  CompanyFacts latest period : {row['companyfacts_latest']}")
        lines.append(f"  releases detected/refused  : {row['releases_detected']} / "
                     f"{row['filings_refused']}")
        lines.append(f"  documents selected         : {row['documents']}")
        lines.append(f"  tables read/usable         : {row['tables_read']} / "
                     f"{row['tables_usable']}")
        lines.append(f"  facts accepted             : {row['facts_accepted']}")
        lines.append(f"  candidate periods          : {row['candidate_periods']}")
        lines.append(f"  completeness               : {row['candidate_completeness']}")
        lines.append(f"  values populated           : {row['values_populated']}")
        lines.append(f"  newest release period      : {row['best_period']} "
                     f"({row['best_period_type']}, {row['best_completeness']}, "
                     f"{row['best_currency']})")
        lines.append(f"  ADVANCES PERIOD            : {row['advances_period']}")
        lines.append(f"  same-period overlap        : {row['same_period_overlap']}")
        lines.append(f"  resolver  CompanyFacts only: {row['resolved_before']} "
                     f"({row['source_before']})")
        lines.append(f"  resolver  with release     : {row['resolved_after']} "
                     f"({row['source_after']}, {row['status_after']})")
        lines.append(f"  superseded / conflicts     : {row['superseded']} / "
                     f"{row['conflicts']}")
        lines.append(f"  TTM before / after         : {row['ttm_before']}  ->  "
                     f"{row['ttm_after']}")
        if row["rejection_codes"]:
            lines.append(f"  rejection codes            : {row['rejection_codes']}")
        for note in row["notes"]:
            lines.append(f"      {note}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--historical", action="store_true",
                        help="reconstruct the window between a release and the "
                             "periodic filing that followed it")
    args = parser.parse_args()

    if args.historical:
        rows = []
        for ticker in args.tickers:
            try:
                rows.append(run_historical(ticker, limit=args.limit))
            except Exception as failure:                      # noqa: BLE001
                rows.append({"ticker": ticker,
                             "error": f"{type(failure).__name__}: {failure}"})
        print(render_historical(rows))
        advanced = [r["ticker"] for r in rows if r.get("advanced")]
        print(f"\nIssuers where the filed release advanced the reported period: "
              f"{advanced or 'none'}")
        return 0

    rows = []
    for ticker in args.tickers:
        try:
            rows.append(run_one(ticker, as_of=args.as_of, limit=args.limit))
        except Exception as failure:                          # noqa: BLE001
            rows.append({"ticker": ticker,
                         "error": f"{type(failure).__name__}: {failure}"})
    print(render(rows))
    advanced = [r["ticker"] for r in rows if r.get("advances_period")]
    print(f"\nIssuers where a filed release reports a NEWER period than "
          f"CompanyFacts can see: {advanced or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
