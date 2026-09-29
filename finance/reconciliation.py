"""Phase H.3 — cross-provider reconciliation.

This project routes each CAPABILITY to exactly one provider (Phase 5's
dataset-specific policy — see tools.config.finance_*_provider()); it never
dual-fetches the same fact from two providers to vote on an answer. So the
overlaps worth reconciling are not hypothetical "provider A vs provider B for
the same dataset" cases — they are the specific, REAL places where two
already-selected providers each independently report something related.

Under the default policy (quote/profile from Yahoo, fundamentals from SEC),
exactly one such overlap actually occurs every run: Yahoo's basic shares-
outstanding count (`overview.shares_outstanding`) versus SEC's weighted-
average DILUTED share count embedded in the balance sheet
(`statements.annual.balance_sheet[0].values.shares_outstanding`, sourced from
XBRL's WeightedAverageNumberOfDilutedSharesOutstanding — see
finance/sec_normalization.py). These are legitimately different figures
(basic vs. diluted-weighted-average), not a data-quality bug, but the
difference is exactly the kind of thing a report must show plainly rather
than silently pick one and hide the other — see docs/security/
YAHOO_SEC_PROVIDER_REVIEW.md §4.

Never silently overwrites one value with the other: both are already present
under their own keys in `facts`; this module only ADDS a warning entry when
they diverge materially, naming both providers, both values, and which one
is actually used by the DCF. Precedence itself lives in
`finance/workflow.py::_dcf_inputs_from_facts`, unchanged by this file — as of
the corrective patch below, SEC's weighted-average diluted figure is
preferred (internally consistent with the SEC-sourced income statement and
balance sheet the rest of the DCF is built from) when a genuinely
SEC-sourced balance sheet is present, with Yahoo's basic figure as the
fallback; this module's warning text is worded to match.
"""

from typing import List

# Beyond this relative difference, basic vs. diluted-weighted-average share
# counts are treated as a material divergence worth flagging, not ordinary
# rounding/share-buyback-during-the-period noise.
_SHARE_COUNT_RELATIVE_TOLERANCE = 0.02


def reconcile_facts(facts: dict) -> List[str]:
    """Compare related facts sourced from DIFFERENT providers within one
    already-built `facts` dict. Returns warning strings (Problem 10's
    existing `warnings` list is where these land); never mutates `facts`.
    """
    warnings: List[str] = []
    warnings.extend(_reconcile_share_counts(facts))
    return warnings


def _reconcile_share_counts(facts: dict) -> List[str]:
    overview = facts.get("overview") or {}
    yahoo_shares = overview.get("shares_outstanding")

    statements = facts.get("statements") or {}
    annual_balance = (statements.get("annual") or {}).get("balance_sheet") or []
    sec_shares = None
    sec_period = None
    if annual_balance:
        latest = annual_balance[0]
        if latest.get("dataset_id") == "sec_company_facts":
            sec_shares = (latest.get("values") or {}).get("shares_outstanding")
            sec_period = latest.get("fiscal_date")

    if yahoo_shares is None or sec_shares is None or sec_shares == 0:
        return []

    # MLI corrective patch: compare on the SAME split basis the DCF uses.
    # A weighted-average diluted count from a filing that predates a split
    # is not comparable to Yahoo's current post-split count -- that is a
    # BASIS difference, not a provider conflict, and reporting it as one
    # sent the reader looking for a data-quality problem that did not
    # exist while hiding a real 2x valuation error. See
    # finance.workflow.sec_share_count_split_factor.
    from finance.workflow import split_adjusted_sec_share_count
    adjusted, detail = split_adjusted_sec_share_count(facts)
    split_factor = (detail or {}).get("split_factor", 1.0)
    comparison_shares = adjusted if adjusted else sec_shares

    diff = yahoo_shares - comparison_shares
    relative = abs(diff) / abs(comparison_shares)
    if relative <= _SHARE_COUNT_RELATIVE_TOLERANCE:
        return []

    basis_note = ""
    if split_factor != 1.0:
        applied = ", ".join(f"{s['ratio']:g}-for-1 on {s['date']}"
                           for s in (detail or {}).get("splits_applied", []))
        basis_note = (
            f" (SEC's {sec_shares:,.0f} was restated to {comparison_shares:,.0f} onto the "
            f"current split basis for this comparison, reflecting {applied}; the residual "
            "gap below is what remains after that restatement)")

    return [
        "MARKET_CAP_SHARE_COUNT_MISMATCH: Yahoo's basic shares outstanding "
        f"({yahoo_shares:,.0f}) differs from SEC's weighted-average diluted share "
        f"count for {sec_period} ({comparison_shares:,.0f}) by {relative:.1%}"
        f"{basis_note}. Basic-current and diluted-weighted-average are legitimately "
        "different measures, so a modest gap is expected; a LARGE residual gap after "
        "split restatement is not, and is surfaced here rather than assumed benign. "
        "The DCF uses the SEC weighted-average diluted figure on the current split "
        "basis (internally consistent with the SEC-sourced income statement and "
        "balance sheet the rest of the valuation is built from); Yahoo's basic figure "
        "remains visible under the company overview for comparison, never silently "
        "substituted for the other."
    ]
