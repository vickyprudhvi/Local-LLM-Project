"""Phase H.7 — reporting frameworks: us-gaap AND ifrs-full.

THE BUG THIS EXISTS TO FIX (live JBS N.V., valuation run 2026-08-19)
====================================================================
A full analysis produced this:

    latest annual period      : None
    latest quarterly period   : None
    revenue                   : unavailable
    every balance-sheet field : None
    DCF                       : not available -- "the latest annual revenue
                                was not reported"

The issuer reports. It filed a 20-F and forty-six 6-Ks. What it does not do
is report under `us-gaap`: its companyfacts payload contains exactly one
taxonomy, `ifrs-full`, with 287 concepts. Every lookup in this project went
to `facts["us-gaap"]`, found nothing, and reported the company as having
published no financials at all -- a far stronger and more wrong claim than
"this project cannot read this issuer's reporting framework".

The same gap hid behind the form names. `ANNUAL_FORMS` was ("10-K",
"10-K/A") and `QUARTERLY_FORMS` was ("10-Q", "10-Q/A"), so a foreign private
issuer's annual report (20-F) and interim report (6-K) were invisible even
where the concepts would have resolved.

WHAT THIS MODULE DOES
=====================
It answers one question -- "which taxonomy is this issuer reporting in, and
what are its concept names for the fields we read?" -- and nothing else. The
normalized field names are unchanged, so every consumer downstream
(finance/freshness.py, finance/ttm.py, finance/net_debt.py, the DCF input
builder) reads `revenue` and `long_term_debt` exactly as before and never
learns which framework produced them.

THE RULE ABOUT MIXING: a series is built from ONE taxonomy. IFRS operating
profit and us-gaap operating income are not the same measure, and an issuer
that changed frameworks mid-history has two series rather than one long one.
`PeriodFact.taxonomy` carries the framework through so a mixed sum is
detectable rather than invisible.

Nothing here is issuer-specific. The taxonomy is read from the payload; the
concept lists are reviewed mappings for a reporting framework, exactly as
finance/xbrl_mapping.py's are for us-gaap.
"""

from typing import Dict, List, Optional, Tuple

US_GAAP = "us-gaap"
IFRS = "ifrs-full"
SUPPORTED_TAXONOMIES = (US_GAAP, IFRS)

# Annual and interim report forms BY ISSUER TYPE. Section 7: freshness is
# decided from actual financial periods, but form names still say which
# filings to look in, and a domestic-only list makes a foreign private
# issuer's entire history unreachable.
#
#   10-K / 10-Q      domestic registrant
#   20-F             foreign private issuer annual report
#   40-F             Canadian MJDS annual report
#   6-K              foreign private issuer interim/current report, which
#                    carries the interim financial statements many FPIs file
#                    instead of a 10-Q
ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")
INTERIM_FORMS = ("10-Q", "10-Q/A", "6-K", "6-K/A")
ALL_REPORT_FORMS = ANNUAL_FORMS + INTERIM_FORMS

# Forms that can carry an earnings release or interim results for guidance
# discovery (section 10). 8-K item 2.02 is the domestic path and stays
# item-gated; a 6-K has no item taxonomy at all, so it is included wholesale
# and the extractor's own precision rules -- a range or a stated tolerance,
# inside a forward-looking clause, attached to a named metric -- do the
# filtering that the item code would otherwise do.
EARNINGS_MATERIAL_FORMS = ("8-K", "6-K")


# ---------------------------------------------------------------------------
# IFRS concept mapping
# ---------------------------------------------------------------------------
#
# Same contract as finance/xbrl_mapping.py::CONCEPT_MAP -- normalized field
# name -> (is_instant, candidate concepts in precedence order) -- so the two
# frameworks are interchangeable to every caller.
#
# Reviewed against IFRS Taxonomy element names as they appear in real
# 20-F/6-K companyfacts payloads. Where IFRS has no equivalent of a us-gaap
# line the field is simply ABSENT rather than approximated: inventing an
# "operating income" from a measure the issuer did not report would be a
# guess wearing a normalized field name, which is the failure mode this
# whole project is built against.
IFRS_CONCEPT_MAP: Dict[str, Tuple[bool, Tuple[str, ...]]] = {
    "revenue": (False, (
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromSaleOfGoods",
    )),
    "operating_income": (False, (
        "ProfitLossFromOperatingActivities",
        "OperatingIncomeLoss",
    )),
    "net_income": (False, (
        "ProfitLossAttributableToOwnersOfParent",
        "ProfitLoss",
    )),
    "gross_profit": (False, ("GrossProfit",)),
    "cash_and_cash_equivalents": (True, (
        "CashAndCashEquivalents",
        "Cash",
    )),
    "short_term_investments": (True, (
        "OtherCurrentFinancialAssets",
        "CurrentFinancialAssetsAtFairValueThroughProfitOrLoss",
    )),
    "assets": (True, ("Assets",)),
    "current_assets": (True, ("CurrentAssets",)),
    "liabilities": (True, ("Liabilities",)),
    "current_liabilities": (True, ("CurrentLiabilities",)),
    "stockholders_equity": (True, (
        "EquityAttributableToOwnersOfParent",
        "Equity",
    )),
    "minority_interest": (True, ("NoncontrollingInterests",)),
    "short_term_debt": (True, (
        "ShorttermBorrowings",
        "CurrentPortionOfLongtermBorrowings",
    )),
    "current_portion_of_long_term_debt": (True, (
        "CurrentPortionOfLongtermBorrowings",
    )),
    "long_term_debt": (True, (
        "NoncurrentPortionOfNoncurrentBorrowings",
        "LongtermBorrowings",
        "Borrowings",
    )),
    "lease_liabilities": (True, ("LeaseLiabilities",)),
    "operating_cash_flow": (False, (
        "CashFlowsFromUsedInOperatingActivities",
    )),
    "capital_expenditure": (False, (
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    )),
    "depreciation_and_amortization": (False, (
        "DepreciationAndAmortisationExpense",
        "DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss",
    )),
    "interest_expense": (False, (
        "InterestExpense",
        "FinanceCosts",
    )),
    "income_tax_expense": (False, ("IncomeTaxExpenseContinuingOperations",)),
    "diluted_shares": (False, (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "AdjustedWeightedAverageShares",
    )),
    "diluted_eps": (False, ("DilutedEarningsLossPerShare",)),
}

# Taxonomy elements that are NOT financial statements and must never be
# counted when deciding which framework an issuer reports in.
_NON_FINANCIAL_TAXONOMIES = frozenset({"dei", "ecd", "srt", "ffd", "country", "invest"})


def detect_taxonomy(company_facts: dict) -> Optional[str]:
    """Which reporting framework this issuer's facts are in.

    Chooses the framework with the MOST concepts when a payload carries both
    -- an issuer transitioning between filing statuses genuinely can --
    rather than preferring us-gaap by default. Preferring one would silently
    pick the stub side of a transition, which is the same class of error as
    picking the stale side of a period.
    """
    facts = (company_facts or {}).get("facts") or {}
    present = [(name, len(facts.get(name) or {})) for name in SUPPORTED_TAXONOMIES
               if facts.get(name)]
    if not present:
        return None
    return max(present, key=lambda pair: pair[1])[0]


def available_taxonomies(company_facts: dict) -> List[str]:
    facts = (company_facts or {}).get("facts") or {}
    return [name for name in SUPPORTED_TAXONOMIES if facts.get(name)]


def concept_map_for(taxonomy: Optional[str]) -> Dict[str, Tuple[bool, Tuple[str, ...]]]:
    """The reviewed field -> concept mapping for one framework."""
    if taxonomy == IFRS:
        return IFRS_CONCEPT_MAP
    from finance.xbrl_mapping import CONCEPT_MAP
    return CONCEPT_MAP


def concept_map_for_facts(company_facts: dict) -> Tuple[Optional[str], Dict]:
    """(taxonomy, mapping) for one companyfacts payload."""
    taxonomy = detect_taxonomy(company_facts)
    return taxonomy, concept_map_for(taxonomy)


def unsupported_taxonomy_reason(company_facts: dict) -> Optional[str]:
    """A plain statement of why nothing could be read, or None.

    This exists so a report can say "this issuer reports under a framework
    this project does not read" instead of "this company reported no
    financials". Those are not close to the same claim, and the second one
    is what was published.
    """
    facts = (company_facts or {}).get("facts") or {}
    if not facts:
        return "The SEC returned no tagged financial facts for this issuer."
    if detect_taxonomy(company_facts) is not None:
        return None
    frameworks = sorted(name for name in facts if name not in _NON_FINANCIAL_TAXONOMIES)
    if not frameworks:
        return ("The SEC returned only document metadata for this issuer, with no tagged "
                "financial statements.")
    return (f"This issuer reports under {', '.join(frameworks)}, for which this project has "
            f"no reviewed concept mapping. Only {', '.join(SUPPORTED_TAXONOMIES)} are read.")


def reporting_currency_note(company_facts: dict) -> Optional[str]:
    """A note when the issuer reports in a currency this project cannot value.

    finance/xbrl_mapping.py::_candidate_facts deliberately refuses to read a
    non-USD series into the numeric path -- there is no FX conversion here,
    and a EUR revenue series feeding an equity bridge that ends in a USD
    price per share is the currency version of the share-basis error. That
    refusal must not look like "the company reported nothing", so the
    situation is stated instead.
    """
    facts = (company_facts or {}).get("facts") or {}
    taxonomy = detect_taxonomy(company_facts)
    if not taxonomy:
        return None
    # Decided from the concepts the VALUATION actually reads, not from a scan
    # of everything the issuer tags.
    #
    # Two earlier versions of this check were fooled. The first counted a
    # `shares` unit as evidence of usability -- but every filer reports share
    # counts in `shares` whatever currency its statements use. The second
    # counted any `USD` unit, and a real euro-reporting issuer carried three:
    # a foreign-exchange derivative notional, a purchase commitment and a
    # tax-benefit lapse. None of them is a statement line, and 548 of its
    # concepts were in euros. Incidental disclosures cannot establish the
    # currency of the accounts.
    #
    # The mapping FOR THIS FRAMEWORK is the list of things the numeric path
    # tries to read. If none of those is available in US dollars while some
    # are available in another currency, the statements are not in dollars.
    # Taken through `concept_map_for` rather than reaching straight for the
    # us-gaap map, because an IFRS filer tags `Revenue`, not `Revenues`, and
    # a us-gaap-only list would find nothing to judge and fall silent -- the
    # same silence this whole check exists to end.
    wanted = set()
    for _prefer_latest, concepts in concept_map_for(taxonomy).values():
        wanted.update(concepts)

    currencies = set()
    usable = False
    for name, entry in (facts.get(taxonomy) or {}).items():
        if name not in wanted:
            continue
        for unit_name, rows in ((entry or {}).get("units") or {}).items():
            if not isinstance(rows, list) or not rows:
                continue
            if unit_name == "USD":
                usable = True
            elif len(unit_name) == 3 and unit_name.isalpha():
                currencies.add(unit_name)
    if usable or not currencies:
        return None
    return (f"This issuer reports in {', '.join(sorted(currencies))}. This project has no "
            "currency conversion, so its financial statements are not read into a valuation "
            "that ends in a US-dollar price per share.")
