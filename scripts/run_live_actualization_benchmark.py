"""Actualization V1 vs V2 against LIVE SEC filings. Section 22's small set.

WHY A SCRIPT AND NOT A TEST

It calls SEC EDGAR. A suite that must stay fast, offline and deterministic
cannot own it, and a live issuer's answer changes the day it files.

WHAT IT DOES AND DOES NOT DO

It fetches one issuer's company facts and runs BOTH layers over them, exactly
as `tests/actualization_benchmark_harness.py` does offline. It changes no
configuration, writes nothing to the report path and switches no default:
`FINANCE_ACTUALIZATION_MODE` stays whatever it is, because the layers are
called directly rather than through the runtime seam. Compare mode is never
used -- it returns V1's answer by design, and reading that as V2's behaviour
is the one mistake that would make the whole comparison meaningless.

It reports only what section 22 asks for:

    V1 selected period / V2 selected period
    V1 source          / V2 source
    TTM end
    fallbacks
    DCF freshness
    hard-safety failures

There is no expected data here. A live issuer has no ground truth this repo
can assert; what a live run can show is whether the two layers AGREE, and
where they do not, which one the filings support -- which a reader decides by
looking at the named forms and dates. A disagreement is reported as a
disagreement, never scored as a failure of either side.

USAGE

    python -m scripts.run_live_actualization_benchmark AAA BBB CCC

The DCF base period is taken as V1's own answer (its latest quarterly or
annual period), because that IS what a live valuation would be built on --
which makes "is the base stale?" a question about the live pipeline rather
than about a number invented for the run.
"""

import argparse
import datetime
import sys
from typing import List, Optional


def _company_facts(ticker: str) -> dict:
    from finance.sec_provider import resolve_cik
    from tools import finance_tools

    coordinator = finance_tools.get_sec_coordinator()
    cik, name = resolve_cik(coordinator, ticker)
    outcome = coordinator.fetch("company_facts", ticker, {"cik": cik})
    return outcome.payload or {}, name


def _hard_safety(resolution, ttm, assessment) -> List[str]:
    """The seven section 7 conditions, checked without expected data.

    Only the ones a live run CAN check: each is a self-consistency property of
    one answer, not a comparison against a known-correct period.
    """
    from finance.actualization import TTM_END_TOLERANCE_DAYS

    failures = []
    published = {f.period_end for f in resolution.fallbacks if f.from_current_period}
    published.discard(None)
    if len(published) > 1:
        failures.append(
            f"SILENT_MIXED_PERIOD_STATE: current-period metrics from {sorted(published)}")
    if published and resolution.period_end and published != {resolution.period_end}:
        failures.append(
            f"LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER: {sorted(published)} "
            f"against {resolution.period_end}")

    for record in resolution.fallbacks:
        if record.from_current_period and record.is_fallback:
            failures.append(
                f"LATEST_QUARTER_METRIC_FROM_OLDER_QUARTER: {record.metric} is "
                "both a fallback and presented as current")

    primary = resolution.selected_primary_source
    if primary is not None and primary.is_prospective:
        failures.append("GUIDANCE_CLASSIFIED_AS_ACTUAL: the primary source is an outlook")
    forms = [s.form for s in resolution.superseded_sources]
    if primary is not None and primary.form in forms:
        failures.append(
            f"DUPLICATE_CURRENT_STATE_SAME_PERIOD: {primary.form} is both primary "
            "and superseded")

    if ttm is not None and resolution.period_end:
        for metric, window in sorted(ttm.windows.items()):
            if not window.ends_at_current_period or not window.end_date:
                continue
            lag = (datetime.date.fromisoformat(resolution.period_end)
                   - datetime.date.fromisoformat(window.end_date)).days
            if lag > TTM_END_TOLERANCE_DAYS:
                failures.append(
                    f"CURRENT_TTM_ENDS_BEFORE_RESOLVED_PERIOD: {metric} ends "
                    f"{window.end_date}, {lag} days before {resolution.period_end}")

    if assessment is not None and assessment.is_stale and assessment.may_be_research_valid:
        failures.append("STALE_DCF_MARKED_VALID_FOR_RESEARCH")
    return failures


def run_one(ticker: str, as_of: Optional[str] = None) -> dict:
    from finance import actualization as act
    from finance import ttm as ttm_module
    from finance.extraction.document_resolver import discover_reported_actuals
    from finance.freshness import build_current_financial_state

    facts, name = _company_facts(ticker)
    as_of = as_of or datetime.date.today().isoformat()

    state = build_current_financial_state(facts, ticker, valuation_date=as_of)
    v1_period = state.financial_as_of
    v1_source = None
    for field_name in ("cash_and_cash_equivalents", "stockholders_equity", "assets"):
        selection = state.balance_sheet.get(field_name)
        if selection is not None and selection.as_of_date == v1_period:
            v1_source = selection.form
            break
    v1_ttm_ends = sorted({s.as_of_date for s in state.flows.values()
                          if s.value is not None and s.as_of_date})

    resolution = act.resolve_current_actual_state(
        discover_reported_actuals(facts), as_of=as_of)
    ttm = act.reconstruct_ttm(facts, resolution.period_end) \
        if resolution.period_end else None

    base = state.latest_quarterly_period or state.latest_annual_period
    assessment = act.assess_dcf_base_freshness(resolution.period_end, base)

    return {
        "ticker": ticker,
        "name": name,
        "as_of": as_of,
        "v1_period": v1_period,
        "v2_period": resolution.period_end,
        "v1_source": v1_source,
        "v2_source": (resolution.selected_primary_source.form
                      if resolution.selected_primary_source else None),
        "v2_status": resolution.state_status,
        "v2_completeness": resolution.statement_completeness,
        "v2_codes": list(resolution.codes),
        "v2_superseded": [s.form for s in resolution.superseded_sources],
        "v1_flow_period_ends": v1_ttm_ends,
        "v2_ttm_reference": (ttm.reference_period_end if ttm else None),
        "v2_ttm_status": (ttm.status if ttm else None),
        "v2_ttm_stale": (list(ttm.stale_metrics) if ttm else []),
        "v2_fallbacks": [f"{f.metric}@{f.period_end}"
                         for f in resolution.fallbacks if f.is_fallback],
        "dcf_base_period": base,
        "dcf_freshness": assessment.freshness,
        "dcf_research_valid": assessment.may_be_research_valid,
        "hard_safety_failures": _hard_safety(resolution, ttm, assessment),
        "period_disagreement": bool(v1_period and resolution.period_end
                                    and v1_period != resolution.period_end),
        "source_disagreement": bool(v1_source and resolution.selected_primary_source
                                    and v1_source != resolution.selected_primary_source.form),
    }


def render(rows: List[dict]) -> str:
    lines = []
    for row in rows:
        if row.get("error"):
            lines.append(f"\n{row['ticker']}: ERROR {row['error']}")
            continue
        lines.append(f"\n{row['ticker']}  ({row.get('name') or '?'})   as of {row['as_of']}")
        lines.append(f"  selected period   V1 {row['v1_period']!s:<12} "
                     f"V2 {row['v2_period']!s:<12}"
                     + ("   <- DISAGREE" if row["period_disagreement"] else ""))
        lines.append(f"  primary source    V1 {row['v1_source']!s:<12} "
                     f"V2 {row['v2_source']!s:<12}"
                     + ("   <- DISAGREE" if row["source_disagreement"] else ""))
        lines.append(f"  V2 state          {row['v2_status']} / {row['v2_completeness']}")
        lines.append(f"  V2 codes          {', '.join(row['v2_codes']) or '-'}")
        lines.append(f"  V2 superseded     {', '.join(row['v2_superseded']) or '-'}")
        lines.append(f"  TTM end           V1 flows end {row['v1_flow_period_ends']}   "
                     f"V2 anchor {row['v2_ttm_reference']} ({row['v2_ttm_status']})")
        lines.append(f"  V2 TTM not current {', '.join(row['v2_ttm_stale']) or '-'}")
        lines.append(f"  fallbacks         {', '.join(row['v2_fallbacks']) or '-'}")
        lines.append(f"  DCF base          {row['dcf_base_period']} -> "
                     f"{row['dcf_freshness']} "
                     f"(research-valid: {row['dcf_research_valid']})")
        failures = row["hard_safety_failures"]
        lines.append(f"  HARD SAFETY       {len(failures)}")
        for failure in failures:
            lines.append(f"      {failure}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args()

    rows = []
    for ticker in args.tickers:
        try:
            rows.append(run_one(ticker, as_of=args.as_of))
        except Exception as failure:                          # noqa: BLE001
            rows.append({"ticker": ticker,
                         "error": f"{type(failure).__name__}: {failure}"})
    print(render(rows))

    total = sum(len(r.get("hard_safety_failures") or []) for r in rows)
    print(f"\nTotal V2 hard-safety failures across the live set: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
