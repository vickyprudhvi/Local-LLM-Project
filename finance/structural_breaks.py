"""Phase H.6 — is this company's own history comparable to its present?

THE BUG THIS EXISTS TO FIX (live AT&T, valuation run 2026-08-17)
================================================================
The T report anchored forward reasoning on a historical revenue CAGR of
-7.1% and flagged it against management's outlook as a material conflict.
But AT&T's reported revenue history is:

    FY2021   $168.9B      <- includes WarnerMedia and DirecTV
    FY2022   $120.7B      <- after the separations
    FY2023   $122.4B
    FY2024   $122.3B
    FY2025   $125.6B

The -7.1% is almost entirely the WarnerMedia/DirecTV separations, arithmetic
performed on a company that no longer exists. On the CONTINUING business the
same five years compound at roughly +1.3%. Treating -7.1% as "what this
company grows at" is not conservative; it is measuring one company and
forecasting another.

WHAT THIS MODULE DOES — AND DELIBERATELY DOES NOT DO
====================================================
It classifies `historical_comparability_status` from FILING EVIDENCE only:
discontinued-operations facts the issuer itself tagged, a continuing-
operations restatement of a line the issuer used to report on a total-company
basis, a revenue discontinuity of a magnitude ordinary trading does not
produce, and 8-K item codes for completed acquisitions and dispositions.

The local model is never asked whether a structural break happened, and never
gets to assert one (section 13). A model that is asked "did anything
structural change?" will find something for every company, and a fabricated
break is worse than a missed one because it licenses discarding real history.

A STRUCTURAL_BREAK does not delete the historical CAGR either. It DOWNWEIGHTS
it: finance/forward_assumptions.py drops long-period CAGR below the recent
continuing-operation trend in its precedence, and the report says which
periods are affected and why.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from finance import period_facts as pf


class HistoricalComparability:
    """How much of this company's reported history describes today's company."""

    COMPARABLE = "COMPARABLE"
    PARTIALLY_COMPARABLE = "PARTIALLY_COMPARABLE"
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK"
    UNKNOWN = "UNKNOWN"
    ALL = (COMPARABLE, PARTIALLY_COMPARABLE, STRUCTURAL_BREAK, UNKNOWN)


class BreakKind:
    DISCONTINUED_OPERATIONS = "discontinued_operations"
    CONTINUING_OPERATIONS_RESTATEMENT = "continuing_operations_restatement"
    REVENUE_DISCONTINUITY = "revenue_discontinuity"
    COMPLETED_ACQUISITION_OR_DISPOSITION = "completed_acquisition_or_disposition"
    SEGMENT_REORGANIZATION = "segment_reorganization"
    SHARE_COUNT_TRANSFORMATION = "share_count_transformation"


# us-gaap concepts an issuer tags only when it HAS discontinued operations.
# Their mere presence for a period is the issuer's own statement that the
# period's total-company figures are not comparable with a period that has
# none.
_DISCONTINUED_OPERATIONS_CONCEPTS = (
    "IncomeLossFromDiscontinuedOperationsNetOfTax",
    "IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity",
    "DisposalGroupIncludingDiscontinuedOperationRevenue",
    "CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations",
    "DisposalGroupIncludingDiscontinuedOperationAssets",
)

# A one-year revenue move this large is not organic. Ordinary cyclical
# businesses swing hard, so this is deliberately set well above any normal
# trading year: NVDA's +114% FY2025 and AT&T's -28.5% FY2022 both clear it,
# and the two are distinguished by the CORROBORATING evidence below, not by
# the threshold alone.
_REVENUE_DISCONTINUITY_THRESHOLD = 0.25

# 8-K items that report a COMPLETED transaction changing what the company is.
#   2.01  Completion of Acquisition or Disposition of Assets
#   5.01  Changes in Control of Registrant
_TRANSACTION_ITEMS = ("2.01", "5.01")


@dataclass
class ComparabilityEvidence:
    """One reason the history may not describe the present."""

    kind: str
    period: Optional[str]
    detail: str
    magnitude: Optional[float] = None
    source: str = "sec_company_facts"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "period": self.period, "detail": self.detail,
                "magnitude": self.magnitude, "source": self.source}


@dataclass
class ComparabilityAssessment:
    """The verdict, the evidence for it, and which periods are affected."""

    status: str = HistoricalComparability.UNKNOWN
    evidence: List[ComparabilityEvidence] = field(default_factory=list)
    affected_periods: Tuple[str, ...] = ()
    comparable_from: Optional[str] = None
    summary: str = ""

    @property
    def is_broken(self) -> bool:
        return self.status == HistoricalComparability.STRUCTURAL_BREAK

    def to_dict(self) -> dict:
        return {
            "historical_comparability_status": self.status,
            "evidence": [e.to_dict() for e in self.evidence],
            "affected_periods": list(self.affected_periods),
            "comparable_from": self.comparable_from,
            "summary": self.summary,
        }


def _has_concept(company_facts: dict, concept: str) -> List[dict]:
    usgaap = (company_facts.get("facts") or {}).get("us-gaap") or {}
    entry = usgaap.get(concept) or {}
    rows: List[dict] = []
    for unit_rows in (entry.get("units") or {}).values():
        if isinstance(unit_rows, list):
            rows.extend(unit_rows)
    return rows


def _discontinued_operations_periods(company_facts: dict, since: Optional[str]
                                     ) -> List[ComparabilityEvidence]:
    found: Dict[str, ComparabilityEvidence] = {}
    for concept in _DISCONTINUED_OPERATIONS_CONCEPTS:
        for row in _has_concept(company_facts, concept):
            end = row.get("end")
            value = row.get("val")
            if not end or value in (None, 0):
                continue
            if since and end < since:
                continue
            if end in found:
                continue
            found[end] = ComparabilityEvidence(
                kind=BreakKind.DISCONTINUED_OPERATIONS,
                period=end,
                detail=(f"The issuer tagged {concept} for the period ending {end}, so that "
                        "period's total-company figures include a business that is no longer "
                        "part of the company."),
                magnitude=float(value) if isinstance(value, (int, float)) else None)
    return sorted(found.values(), key=lambda e: e.period or "")


def _continuing_operations_restatement(company_facts: dict) -> Optional[ComparabilityEvidence]:
    """The issuer moved a line onto a continuing-operations basis.

    AT&T is the live case: it reported
    `NetCashProvidedByUsedInOperatingActivities` through fiscal 2025 and
    reports `...ContinuingOperations` for fiscal 2026. Switching a headline
    cash-flow line onto a continuing-operations spelling is the issuer saying
    the two bases are not the same figure.
    """
    pairs = (
        ("NetCashProvidedByUsedInOperatingActivities",
         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
         "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
    )
    for total_concept, continuing_concept in pairs:
        total_rows = [r for r in _has_concept(company_facts, total_concept) if r.get("end")]
        continuing_rows = [r for r in _has_concept(company_facts, continuing_concept)
                           if r.get("end")]
        if not total_rows or not continuing_rows:
            continue
        total_latest = max(r["end"] for r in total_rows)
        continuing_latest = max(r["end"] for r in continuing_rows)
        if continuing_latest > total_latest:
            return ComparabilityEvidence(
                kind=BreakKind.CONTINUING_OPERATIONS_RESTATEMENT,
                period=continuing_latest,
                detail=(f"The issuer stopped reporting {total_concept} after {total_latest} and "
                        f"reports {continuing_concept} through {continuing_latest}. The current "
                        "figures are on a continuing-operations basis and the older ones are "
                        "not, so the two series are not directly comparable."))
    return None


def _revenue_discontinuities(company_facts: dict, max_years: int = 6
                             ) -> List[ComparabilityEvidence]:
    annual = pf.annual_periods(company_facts, "revenue")[-max_years:]
    out: List[ComparabilityEvidence] = []
    for earlier, later in zip(annual, annual[1:]):
        if not earlier.value:
            continue
        change = (later.value - earlier.value) / abs(earlier.value)
        if abs(change) < _REVENUE_DISCONTINUITY_THRESHOLD:
            continue
        out.append(ComparabilityEvidence(
            kind=BreakKind.REVENUE_DISCONTINUITY,
            period=later.end,
            detail=(f"Reported revenue moved {change:+.1%} between the fiscal year ending "
                    f"{earlier.end} ({earlier.value:,.0f}) and the one ending {later.end} "
                    f"({later.value:,.0f})."),
            magnitude=round(change, 4)))
    return out


def _completed_transactions(submissions: Optional[dict], since: Optional[str], limit: int = 6
                            ) -> List[ComparabilityEvidence]:
    if not submissions:
        return []
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    items = recent.get("items") or []
    dates = recent.get("filingDate") or []
    out: List[ComparabilityEvidence] = []
    for index, form in enumerate(forms):
        if form != "8-K":
            continue
        item_text = (items[index] if index < len(items) else "") or ""
        if not any(code in item_text for code in _TRANSACTION_ITEMS):
            continue
        filed = dates[index] if index < len(dates) else ""
        if since and filed and filed < since:
            continue
        out.append(ComparabilityEvidence(
            kind=BreakKind.COMPLETED_ACQUISITION_OR_DISPOSITION,
            period=filed,
            detail=(f"An 8-K filed {filed} reports item(s) {item_text.strip()} — a completed "
                    "acquisition or disposition of assets."),
            source="sec_submissions"))
        if len(out) >= limit:
            break
    return out


def assess_historical_comparability(company_facts: dict,
                                    submissions: Optional[dict] = None,
                                    history_window_start: Optional[str] = None,
                                    history_years: int = 5
                                    ) -> ComparabilityAssessment:
    """Classify how far this company's reported history describes it today.

    The classification is EVIDENCE-FIRST, and that ordering is the whole
    design. A revenue discontinuity on its own says nothing: NVIDIA's revenue
    rose 114% in FY2025 and 65% in FY2026 selling more of the same product to
    the same customers, and downweighting that history would discard the only
    real information there is about the company. What makes history
    non-comparable is the ISSUER stating that the business changed —
    discontinued-operations facts it tagged itself, a restatement of a
    headline line onto a continuing-operations basis, a completed
    acquisition or disposition reported on an 8-K.

    So:

        issuer-tagged business change INSIDE the history window
                                                    -> STRUCTURAL_BREAK
        such evidence only OUTSIDE the window, or a completed transaction
        with no restatement                         -> PARTIALLY_COMPARABLE
        neither                                     -> COMPARABLE
        too little history to look at               -> UNKNOWN

    "Inside the window" matters because the window is what the CAGR is
    computed over. AT&T tagged discontinued operations through the periods
    ending 2021-06-30 to 2022-09-30 (DirecTV, then WarnerMedia); a five-year
    CAGR run in 2026 spans exactly those years, which is why its -7.1%
    measures a company that no longer exists.

    UNKNOWN is returned rather than COMPARABLE when there is too little
    history: "we did not look far enough back" and "we looked and it is
    clean" are different claims, and only one of them supports leaning on a
    long-period growth rate.
    """
    assessment = ComparabilityAssessment()
    annual = pf.annual_periods(company_facts, "revenue")
    if len(annual) < 3:
        assessment.status = HistoricalComparability.UNKNOWN
        assessment.summary = (
            f"Only {len(annual)} annual revenue period(s) are available, which is too little "
            "history to judge whether it describes the company as it is now.")
        return assessment

    window = annual[-(history_years + 1):] if history_years else annual
    window_start = history_window_start or (window[0].start or window[0].end)
    window_end = window[-1].end or ""

    # Everything the issuer said about its own structure, anywhere in its
    # filed history, then partitioned by whether it lands in the window.
    all_discontinued = _discontinued_operations_periods(company_facts, since=None)
    in_window = [e for e in all_discontinued
                 if e.period and window_start <= e.period <= window_end]
    before_window = [e for e in all_discontinued if e.period and e.period < window_start]
    restatement = _continuing_operations_restatement(company_facts)
    transactions = _completed_transactions(submissions, window_start)
    discontinuities = _revenue_discontinuities(company_facts,
                                               max_years=(history_years + 1) or 6)

    evidence: List[ComparabilityEvidence] = []
    # Report at most a few discontinued-operations periods: an issuer tags
    # the concept on every interim filing for years, and twenty near-identical
    # entries bury the finding rather than supporting it.
    evidence.extend(_summarize_discontinued(in_window or before_window))
    if restatement is not None:
        evidence.append(restatement)
    evidence.extend(transactions)
    evidence.extend(discontinuities)
    assessment.evidence = evidence
    assessment.affected_periods = tuple(sorted(
        {e.period for e in (in_window or before_window) if e.period}))

    issuer_change_in_window = bool(in_window) or restatement is not None

    if issuer_change_in_window:
        assessment.status = HistoricalComparability.STRUCTURAL_BREAK
        latest_affected = max((e.period or "" for e in in_window), default="")
        after = [p.end for p in annual if p.end and p.end > latest_affected]
        assessment.comparable_from = after[0] if after else None
        pieces = []
        if in_window:
            pieces.append(
                f"The issuer tagged discontinued operations for {len(in_window)} period(s) "
                f"between {min(e.period for e in in_window)} and "
                f"{max(e.period for e in in_window)}, inside the {history_years}-year window a "
                "historical growth rate is computed over.")
        if restatement is not None:
            pieces.append(restatement.detail)
        if discontinuities:
            pieces.append(discontinuities[0].detail)
        assessment.summary = (
            " ".join(pieces)
            + " A long-period growth rate spanning these years measures a company that no "
              "longer exists in that form, so it receives reduced weight in forward "
              "assumptions and is reported as backward-looking context rather than as a "
              "forecast anchor."
            + (f" Reported periods from {assessment.comparable_from} onward are on the "
               "current basis." if assessment.comparable_from else ""))
    elif before_window or transactions:
        assessment.status = HistoricalComparability.PARTIALLY_COMPARABLE
        pieces = []
        if before_window:
            pieces.append(
                f"The issuer tagged discontinued operations through "
                f"{max(e.period for e in before_window)}, which is BEFORE the "
                f"{history_years}-year window a historical growth rate is computed over.")
        if transactions:
            pieces.append(f"{len(transactions)} completed acquisition or disposition of assets "
                          f"was reported on an 8-K since {window_start}.")
        assessment.summary = (
            " ".join(pieces)
            + " The history used for forward assumptions is not itself restated, so it is "
              "still used — with this noted alongside it.")
    else:
        assessment.status = HistoricalComparability.COMPARABLE
        assessment.summary = (
            "No discontinued operations, continuing-operations restatement or completed "
            f"acquisition/disposition was found inside the {history_years}-year window a "
            "historical growth rate is computed over, so the reported history is treated as "
            "describing the company as it is now."
            + (f" Revenue did move sharply within the window ({discontinuities[-1].detail}) "
               "but with no filing evidence of a business change, that is this company's own "
               "reported growth rather than a break in comparability."
               if discontinuities else ""))
    return assessment


def _summarize_discontinued(entries: Sequence[ComparabilityEvidence]
                            ) -> List[ComparabilityEvidence]:
    """First and last only. An issuer tags the concept on every interim
    filing for years; listing all of them buries the finding."""
    if not entries:
        return []
    if len(entries) <= 2:
        return list(entries)
    return [entries[0], entries[-1]]


# ---------------------------------------------------------------------------
# Section 18 — material events AFTER the balance-sheet date
# ---------------------------------------------------------------------------

# 8-K items that report an event capable of changing the capital structure or
# the asset base between the last filed balance sheet and the valuation date.
#   1.01  Entry into a Material Definitive Agreement
#   2.01  Completion of Acquisition or Disposition of Assets
#   2.03  Creation of a Direct Financial Obligation
#   2.04  Triggering Events That Accelerate a Direct Financial Obligation
#   3.02  Unregistered Sales of Equity Securities
#   8.01  Other Events  (only when the description names a capital action)
_POST_BALANCE_SHEET_ITEMS = {
    "1.01": "entry into a material definitive agreement",
    "2.01": "completion of an acquisition or disposition of assets",
    "2.03": "creation of a direct financial obligation",
    "2.04": "a triggering event accelerating a direct financial obligation",
    "3.02": "an unregistered sale of equity securities",
}

# Registration/prospectus forms that mean a security was actually offered.
# Section 23/25: Form 144 is notice of a proposed sale of restricted stock by
# an affiliate. It is included so the event can be REPORTED and correctly
# typed as an insider secondary sale -- not so it can be counted as dilution.
_CAPITAL_MARKETS_FORMS = ("424B2", "424B3", "424B5", "424B7", "FWP", "144")


# ---------------------------------------------------------------------------
# Phase H.9, sections 23-28 — TYPED post-balance-sheet events
# ---------------------------------------------------------------------------
#
# THE BUG THIS REPLACES. Every post-balance-sheet securities filing was
# described as "a securities offering prospectus supplement" and treated
# alike. On a live issuer that produced a dilution warning built from:
#
#     424B5 / FWP     an investment-grade DEBT offering
#     8-K item 2.03   creation of a direct financial obligation -- also debt
#     Form 144        an INSIDER selling shares he already owns
#
# None of those issues a single new share. A Form 144 in particular moves
# existing stock between holders: the issuer's share count is unchanged, no
# capital is raised, and there is no dilution of any kind. Reporting them as
# potential equity dilution is not a conservative approximation; it is a
# different fact about the company.


class PostBalanceSheetEventType:
    """Section 23."""

    ISSUER_EQUITY_ISSUANCE = "ISSUER_EQUITY_ISSUANCE"
    ISSUER_DEBT_ISSUANCE = "ISSUER_DEBT_ISSUANCE"
    CONVERTIBLE_ISSUANCE = "CONVERTIBLE_ISSUANCE"
    WARRANT_ISSUANCE = "WARRANT_ISSUANCE"
    INSIDER_SECONDARY_SALE = "INSIDER_SECONDARY_SALE"
    SHAREHOLDER_SECONDARY_SALE = "SHAREHOLDER_SECONDARY_SALE"
    SHARE_REPURCHASE = "SHARE_REPURCHASE"
    ACQUISITION = "ACQUISITION"
    DIVESTITURE = "DIVESTITURE"
    REFINANCING = "REFINANCING"
    SPECIAL_DIVIDEND = "SPECIAL_DIVIDEND"
    OTHER_MATERIAL_FINANCING = "OTHER_MATERIAL_FINANCING"
    UNKNOWN = "UNKNOWN"
    ALL = (ISSUER_EQUITY_ISSUANCE, ISSUER_DEBT_ISSUANCE, CONVERTIBLE_ISSUANCE,
           WARRANT_ISSUANCE, INSIDER_SECONDARY_SALE, SHAREHOLDER_SECONDARY_SALE,
           SHARE_REPURCHASE, ACQUISITION, DIVESTITURE, REFINANCING,
           SPECIAL_DIVIDEND, OTHER_MATERIAL_FINANCING, UNKNOWN)


# Section 24. Deterministic impact per type. Every flag answers a question the
# valuation actually asks, and nothing is inferred from the presence of
# securities language.
#
#   affects_share_count   does the ISSUER's share count change?
#   affects_debt          does reported debt change?
#   affects_cash          does the issuer receive or pay cash?
#   potential_dilution    can an existing holder's stake be reduced?
_EVENT_IMPACT = {
    PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE: {
        "affects_share_count": True, "affects_debt": False, "affects_cash": True,
        "potential_dilution": True, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": False, "requires_reassessment": True},
    PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE: {
        "affects_share_count": False, "affects_debt": True, "affects_cash": True,
        "potential_dilution": False, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": True, "requires_reassessment": True},
    PostBalanceSheetEventType.CONVERTIBLE_ISSUANCE: {
        "affects_share_count": False, "affects_debt": True, "affects_cash": True,
        # Section 27: debt today, potential equity later. Both, not either.
        "potential_dilution": True, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": True, "requires_reassessment": True},
    PostBalanceSheetEventType.WARRANT_ISSUANCE: {
        "affects_share_count": False, "affects_debt": False, "affects_cash": False,
        "potential_dilution": True, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": False, "requires_reassessment": False},
    PostBalanceSheetEventType.INSIDER_SECONDARY_SALE: {
        # Section 25: existing shares changing hands. The issuer neither
        # issues stock nor receives cash.
        "affects_share_count": False, "affects_debt": False, "affects_cash": False,
        "potential_dilution": False, "affects_equity_bridge": False,
        "affects_enterprise_value_bridge": False, "requires_reassessment": False},
    PostBalanceSheetEventType.SHAREHOLDER_SECONDARY_SALE: {
        "affects_share_count": False, "affects_debt": False, "affects_cash": False,
        "potential_dilution": False, "affects_equity_bridge": False,
        "affects_enterprise_value_bridge": False, "requires_reassessment": False},
    PostBalanceSheetEventType.SHARE_REPURCHASE: {
        "affects_share_count": True, "affects_debt": False, "affects_cash": True,
        "potential_dilution": False, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": False, "requires_reassessment": False},
    PostBalanceSheetEventType.ACQUISITION: {
        "affects_share_count": False, "affects_debt": True, "affects_cash": True,
        "potential_dilution": False, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": True, "requires_reassessment": True},
    PostBalanceSheetEventType.DIVESTITURE: {
        "affects_share_count": False, "affects_debt": False, "affects_cash": True,
        "potential_dilution": False, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": True, "requires_reassessment": True},
    PostBalanceSheetEventType.REFINANCING: {
        "affects_share_count": False, "affects_debt": True, "affects_cash": True,
        "potential_dilution": False, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": True, "requires_reassessment": False},
    PostBalanceSheetEventType.SPECIAL_DIVIDEND: {
        "affects_share_count": False, "affects_debt": False, "affects_cash": True,
        "potential_dilution": False, "affects_equity_bridge": True,
        "affects_enterprise_value_bridge": False, "requires_reassessment": True},
    PostBalanceSheetEventType.OTHER_MATERIAL_FINANCING: {
        "affects_share_count": False, "affects_debt": False, "affects_cash": False,
        "potential_dilution": False, "affects_equity_bridge": False,
        "affects_enterprise_value_bridge": False, "requires_reassessment": True},
    # Section 28: an unidentifiable event asserts NOTHING. In particular it
    # does not assert dilution.
    PostBalanceSheetEventType.UNKNOWN: {
        "affects_share_count": False, "affects_debt": False, "affects_cash": False,
        "potential_dilution": False, "affects_equity_bridge": False,
        "affects_enterprise_value_bridge": False, "requires_reassessment": False},
}

_DEBT_LANGUAGE = re.compile(
    r"(?i)\b(?:notes?|bonds?|debenture|senior\s+(?:unsecured|secured)|"
    r"term\s+loan|credit\s+facility|indenture|fixed\s+rate|floating\s+rate|"
    r"coupon|maturing|due\s+20\d{2})\b")
_EQUITY_LANGUAGE = re.compile(
    r"(?i)\b(?:common\s+stock|ordinary\s+shares?|equity\s+offering|"
    r"depositary\s+shares?|preferred\s+stock)\b")
_CONVERTIBLE_LANGUAGE = re.compile(r"(?i)\bconvertible\b")
_WARRANT_LANGUAGE = re.compile(r"(?i)\bwarrants?\b")

# 8-K item codes that identify an event type on their own.
_ITEM_EVENT_TYPES = (
    ("2.03", PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE),
    ("2.01", PostBalanceSheetEventType.ACQUISITION),
    ("3.02", PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE),
)

# Forms whose very existence names the event.
_FORM_EVENT_TYPES = {
    # A Form 144 is notice of a proposed sale of RESTRICTED stock by an
    # affiliate -- existing shares, no issuance, no proceeds to the issuer.
    "144": PostBalanceSheetEventType.INSIDER_SECONDARY_SALE,
    "4": PostBalanceSheetEventType.INSIDER_SECONDARY_SALE,
}


def classify_security_event(form: str, items: str, description: str = "",
                            document_text: str = "") -> Tuple[str, float]:
    """(event type, confidence) for one post-balance-sheet filing.

    Deterministic and evidence-bound. When the form and its language do not
    identify a type, the answer is UNKNOWN -- section 28 -- rather than a
    guess that happens to be the most alarming option.
    """
    form = (form or "").upper()
    items = items or ""
    haystack = " ".join(filter(None, (description, document_text)))

    if form in _FORM_EVENT_TYPES:
        return _FORM_EVENT_TYPES[form], 0.95
    for code, event_type in _ITEM_EVENT_TYPES:
        if code in items:
            # An item 2.03 that also mentions convertible language is a
            # convertible, which carries BOTH debt and future-dilution
            # consequences.
            if event_type == PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE \
                    and _CONVERTIBLE_LANGUAGE.search(haystack):
                return PostBalanceSheetEventType.CONVERTIBLE_ISSUANCE, 0.8
            return event_type, 0.85

    if form.startswith("424") or form == "FWP":
        # A prospectus supplement is a securities offering of SOME kind. Which
        # kind is decided by the security described, never assumed.
        if _CONVERTIBLE_LANGUAGE.search(haystack):
            return PostBalanceSheetEventType.CONVERTIBLE_ISSUANCE, 0.8
        if _WARRANT_LANGUAGE.search(haystack):
            return PostBalanceSheetEventType.WARRANT_ISSUANCE, 0.7
        if _DEBT_LANGUAGE.search(haystack) and not _EQUITY_LANGUAGE.search(haystack):
            return PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE, 0.75
        if _EQUITY_LANGUAGE.search(haystack) and not _DEBT_LANGUAGE.search(haystack):
            return PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE, 0.75
        return PostBalanceSheetEventType.UNKNOWN, 0.3
    return PostBalanceSheetEventType.UNKNOWN, 0.2


def event_impact(event_type: str) -> dict:
    """Section 24's deterministic impact flags for one event type."""
    return dict(_EVENT_IMPACT.get(event_type, _EVENT_IMPACT[
        PostBalanceSheetEventType.UNKNOWN]))


_EVENT_DESCRIPTIONS = {
    PostBalanceSheetEventType.ISSUER_EQUITY_ISSUANCE:
        "an issuance of equity by the company",
    PostBalanceSheetEventType.ISSUER_DEBT_ISSUANCE:
        "an issuance of debt by the company",
    PostBalanceSheetEventType.CONVERTIBLE_ISSUANCE:
        "an issuance of convertible securities by the company",
    PostBalanceSheetEventType.WARRANT_ISSUANCE: "an issuance of warrants",
    PostBalanceSheetEventType.INSIDER_SECONDARY_SALE:
        "a sale of existing shares by an insider, which does not change the company's "
        "share count and raises no capital for it",
    PostBalanceSheetEventType.SHAREHOLDER_SECONDARY_SALE:
        "a sale of existing shares by a shareholder, which does not change the company's "
        "share count",
    PostBalanceSheetEventType.SHARE_REPURCHASE: "a repurchase of shares by the company",
    PostBalanceSheetEventType.ACQUISITION: "a completed acquisition",
    PostBalanceSheetEventType.DIVESTITURE: "a completed disposition of assets",
    PostBalanceSheetEventType.REFINANCING: "a refinancing",
    PostBalanceSheetEventType.SPECIAL_DIVIDEND: "a special distribution",
    PostBalanceSheetEventType.OTHER_MATERIAL_FINANCING: "a material financing event",
    PostBalanceSheetEventType.UNKNOWN:
        "a securities filing whose type could not be determined from its form and item "
        "codes; no share-count or dilution consequence is claimed for it",
}


def describe_event(event_type: str) -> str:
    return _EVENT_DESCRIPTIONS.get(event_type, _EVENT_DESCRIPTIONS[
        PostBalanceSheetEventType.UNKNOWN])


@dataclass
class PostBalanceSheetEvent:
    """A filed event dated after the balance sheet the valuation bridges to."""

    filed: str
    form: str
    items: str
    description: str
    accession: str = ""
    event_type: str = PostBalanceSheetEventType.UNKNOWN
    confidence: float = 0.0
    impact: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"filed": self.filed, "form": self.form, "items": self.items,
                "description": self.description, "accession": self.accession,
                "event_type": self.event_type, "confidence": self.confidence,
                "impact": dict(self.impact)}

    @property
    def implies_dilution(self) -> bool:
        return bool(self.impact.get("potential_dilution"))


def find_post_balance_sheet_events(submissions: Optional[dict],
                                   balance_sheet_date: Optional[str],
                                   valuation_date: Optional[str] = None,
                                   limit: int = 8) -> List[PostBalanceSheetEvent]:
    """Filings made AFTER the balance-sheet date that could have changed it.

    Section 18: a balance sheet can be the latest one filed and still be
    economically stale. AT&T's 2026-06-30 balance sheet is the newest there
    is, but a debt-funded transaction closing after it would leave the equity
    bridge subtracting a capital structure the company no longer has.

    This function only REPORTS. Nothing here adjusts an accounting balance —
    a deterministic transaction adjustment is not available from an 8-K item
    code, and inventing one would be exactly the fabrication this project
    exists to prevent. The events are surfaced to the assumption builder, the
    RiskReviewer and the FinalInvestmentSynthesizer so a reader can weigh
    them.
    """
    if not submissions or not balance_sheet_date:
        return []
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    items = recent.get("items") or []
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []

    out: List[PostBalanceSheetEvent] = []
    for index, form in enumerate(forms):
        filed = dates[index] if index < len(dates) else ""
        if not filed or filed <= balance_sheet_date:
            continue
        if valuation_date and filed > valuation_date:
            continue
        item_text = (items[index] if index < len(items) else "") or ""
        accession = accessions[index] if index < len(accessions) else ""

        if form == "8-K":
            matched = [label for code, label in _POST_BALANCE_SHEET_ITEMS.items()
                       if code in item_text]
            if not matched:
                continue
            description = "; ".join(matched)
            event_type, confidence = classify_security_event(
                form, item_text, description)
            out.append(PostBalanceSheetEvent(
                filed=filed, form=form, items=item_text.strip(),
                description=f"{describe_event(event_type)} ({description})",
                accession=accession, event_type=event_type, confidence=confidence,
                impact=event_impact(event_type)))
        elif form in _CAPITAL_MARKETS_FORMS:
            event_type, confidence = classify_security_event(form, "", "")
            out.append(PostBalanceSheetEvent(
                filed=filed, form=form, items="",
                description=describe_event(event_type),
                accession=accession, event_type=event_type, confidence=confidence,
                impact=event_impact(event_type)))
        if len(out) >= limit:
            break
    return out
