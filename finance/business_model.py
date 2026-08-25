"""Phase H.10, sections 16-18 — is a standard FCFF DCF the right model here?

A live broker-dealer reported $566M of operating cash flow against $51M of
operating income and $25M of net income. Operating cash flow less capital
expenditure was computed correctly, labelled free cash flow, and the DCF
suitability check said SUITABLE. Nothing in the pipeline knew that for a
broker-dealer most of that cash is customer money moving through customer
accounts -- it belongs to the customers, it reverses, and it is not cash the
owners of the business can take out.

The failure is not that the arithmetic was wrong. It is that the model has
an implicit assumption -- operating cash flow less capital expenditure
approximates what the owners earn -- which holds for an industrial or
software company and does not hold for a bank, a broker, or an insurer. The
assumption was never written down, so it could never be checked.

This module writes it down and checks it.

Classification comes from the issuer's OWN filing metadata: the SIC code the
SEC assigns on its submissions record, corroborated by the presence of
concepts that only a financial institution reports. Both are properties of
the company's filings, so no ticker, name or third-party sector label is
consulted -- which matters, because the vendor sector field for that same
broker-dealer read "Technology / Software - Application".
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional


class CashFlowValuationProfile:
    STANDARD_OPERATING_COMPANY = "STANDARD_OPERATING_COMPANY"
    FINANCIAL_INSTITUTION = "FINANCIAL_INSTITUTION"
    BROKER_DEALER = "BROKER_DEALER"
    BANK = "BANK"
    INSURER = "INSURER"
    REIT_OR_SPECIALIZED = "REIT_OR_SPECIALIZED"
    OTHER_SPECIALIZED = "OTHER_SPECIALIZED"
    UNKNOWN = "UNKNOWN"

    ALL = (STANDARD_OPERATING_COMPANY, FINANCIAL_INSTITUTION, BROKER_DEALER,
           BANK, INSURER, REIT_OR_SPECIALIZED, OTHER_SPECIALIZED, UNKNOWN)


class FcffSuitability:
    SUITABLE = "SUITABLE"
    LIMITED = "LIMITED"
    NOT_SUITABLE = "NOT_SUITABLE"


DCF_CASH_FLOW_NOT_STANDARD_FCFF = "DCF_CASH_FLOW_NOT_STANDARD_FCFF"


# SEC SIC ranges. These are the Commission's own division of registrants and
# are public, stable and issuer-independent -- the same code identifies every
# broker-dealer that files, which is precisely what makes this general rather
# than another special case.
#
# Ranges are (low, high, profile), checked in order.
_SIC_RANGES = (
    (6020, 6079, CashFlowValuationProfile.BANK),                # depository institutions
    (6080, 6099, CashFlowValuationProfile.BANK),                # foreign bank branches, functions
    (6100, 6199, CashFlowValuationProfile.FINANCIAL_INSTITUTION),  # nondepository credit
    (6200, 6299, CashFlowValuationProfile.BROKER_DEALER),       # brokers, dealers, exchanges
    (6300, 6411, CashFlowValuationProfile.INSURER),             # insurance carriers and agents
    (6500, 6599, CashFlowValuationProfile.REIT_OR_SPECIALIZED),  # real estate
    (6798, 6798, CashFlowValuationProfile.REIT_OR_SPECIALIZED),  # REITs
    (6700, 6799, CashFlowValuationProfile.OTHER_SPECIALIZED),   # holding/investment offices
)

# us-gaap concepts that essentially only a financial institution reports.
# Each is corroborating evidence, not a classifier on its own: a single
# incidental tag should not reclassify a manufacturer, so a minimum number
# must be present before these move the verdict by themselves.
_BROKER_DEALER_CONCEPTS = frozenset({
    "PayablesToCustomers",
    "PayablesToBrokerDealersAndClearingOrganizations",
    "ReceivablesFromCustomers",
    "ReceivablesFromBrokerDealersAndClearingOrganizations",
    "CashAndSecuritiesSegregatedUnderFederalAndOtherRegulations",
    "SegregatedCashAndSecurities",
    "SecuritiesBorrowed",
    "SecuritiesLoaned",
    "FinancialInstrumentsOwnedAtFairValue",
    "BrokerageCommissionsRevenue",
})

_BANK_CONCEPTS = frozenset({
    "Deposits",
    "DepositsDomestic",
    "InterestAndDividendIncomeOperating",
    "InterestExpenseDeposits",
    "LoansAndLeasesReceivableNetReportedAmount",
    "ProvisionForLoanLeaseAndOtherLosses",
    "FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell",
})

_INSURER_CONCEPTS = frozenset({
    "PolicyholderBenefitsAndClaimsIncurredNet",
    "LiabilityForFuturePolicyBenefits",
    "PremiumsEarnedNet",
    "DeferredPolicyAcquisitionCosts",
    "UnpaidPolicyClaimsAndClaimsAdjustmentExpense",
})

_CONCEPT_PROFILES = (
    (_BROKER_DEALER_CONCEPTS, CashFlowValuationProfile.BROKER_DEALER),
    (_BANK_CONCEPTS, CashFlowValuationProfile.BANK),
    (_INSURER_CONCEPTS, CashFlowValuationProfile.INSURER),
)

# How many corroborating concepts must appear before filing content alone
# reclassifies a company whose SIC code says otherwise or is missing. Two,
# because one tag can appear incidentally (a manufacturer with a captive
# finance arm reports some receivables concepts) while a pattern of them
# describes the business.
_CONCEPT_EVIDENCE_THRESHOLD = 2

# Profiles for which operating cash flow less capital expenditure is not
# owner free cash flow.
_SPECIALIZED = frozenset({
    CashFlowValuationProfile.BANK,
    CashFlowValuationProfile.BROKER_DEALER,
    CashFlowValuationProfile.INSURER,
    CashFlowValuationProfile.FINANCIAL_INSTITUTION,
})


@dataclass
class BusinessModelClassification:
    """What kind of business this is, and how that was decided.

    `reasons` exists because a classification that changes a valuation
    outcome has to be auditable -- a reader who disagrees needs to see that
    it came from the SEC's own SIC code and the company's own tags, not from
    a guess.
    """

    profile: str = CashFlowValuationProfile.UNKNOWN
    standard_fcff_suitability: str = FcffSuitability.SUITABLE
    sic: Optional[str] = None
    sic_description: Optional[str] = None
    corroborating_concepts: tuple = ()
    reasons: list = field(default_factory=list)
    findings: list = field(default_factory=list)

    @property
    def is_specialized(self) -> bool:
        return self.profile in _SPECIALIZED

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "standard_fcff_suitability": self.standard_fcff_suitability,
            "sic": self.sic,
            "sic_description": self.sic_description,
            "corroborating_concepts": list(self.corroborating_concepts),
            "reasons": list(self.reasons),
            "findings": list(self.findings),
        }


def _profile_from_sic(sic: Optional[str]) -> Optional[str]:
    if not sic:
        return None
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return None
    for low, high, profile in _SIC_RANGES:
        if low <= code <= high:
            return profile
    return CashFlowValuationProfile.STANDARD_OPERATING_COMPANY


def _concepts_present(company_facts: Optional[dict]) -> frozenset:
    """Which us-gaap concepts this issuer actually reports.

    Reads only the concept NAMES, never the values -- classification must
    not depend on how large a number is, only on which things the company
    reports at all.
    """
    if not isinstance(company_facts, dict):
        return frozenset()
    facts = company_facts.get("facts")
    if not isinstance(facts, dict):
        return frozenset()
    names = set()
    for taxonomy in ("us-gaap", "ifrs-full"):
        block = facts.get(taxonomy)
        if isinstance(block, dict):
            names.update(block.keys())
    return frozenset(names)


def classify_business_model(submissions: Optional[dict] = None,
                            company_facts: Optional[dict] = None
                            ) -> BusinessModelClassification:
    """Decide the cash-flow valuation profile from filing metadata alone.

    Two independent signals, in this order:

      1. the SIC code on the issuer's SEC submissions record, and
      2. the presence of concepts only a financial institution reports.

    The second can confirm the first, supply a verdict when the first is
    missing, or -- when a company files a clear pattern of financial-
    institution concepts while its SIC code says otherwise -- override it.
    A SIC code is assigned once and can lag what a company has become; what
    it tags in this year's filing cannot.
    """
    result = BusinessModelClassification()

    sic = None
    if isinstance(submissions, dict):
        sic = submissions.get("sic")
        result.sic = str(sic) if sic else None
        result.sic_description = submissions.get("sicDescription")

    sic_profile = _profile_from_sic(sic)

    present = _concepts_present(company_facts)
    concept_profile = None
    matched: tuple = ()
    for concepts, profile in _CONCEPT_PROFILES:
        hits = tuple(sorted(present & concepts))
        if len(hits) > len(matched):
            matched, concept_profile = hits, profile
    result.corroborating_concepts = matched

    if sic_profile and sic_profile != CashFlowValuationProfile.STANDARD_OPERATING_COMPANY:
        result.profile = sic_profile
        result.reasons.append(
            f"SEC SIC {result.sic} ({result.sic_description or 'no description'}) places this "
            f"issuer in {sic_profile}.")
        if matched:
            result.reasons.append(
                f"Corroborated by {len(matched)} reported concept(s) specific to that model: "
                f"{', '.join(matched[:4])}.")
    elif len(matched) >= _CONCEPT_EVIDENCE_THRESHOLD and concept_profile:
        result.profile = concept_profile
        result.reasons.append(
            f"The issuer reports {len(matched)} concept(s) specific to {concept_profile} "
            f"({', '.join(matched[:4])})"
            + (f", though its SEC SIC code is {result.sic} "
               f"({result.sic_description or 'no description'})." if result.sic else "."))
    elif sic_profile == CashFlowValuationProfile.STANDARD_OPERATING_COMPANY:
        result.profile = sic_profile
        result.reasons.append(
            f"SEC SIC {result.sic} ({result.sic_description or 'no description'}) is outside "
            "the financial-institution ranges, and the issuer reports no pattern of "
            "financial-institution concepts.")
    else:
        result.profile = CashFlowValuationProfile.UNKNOWN
        result.reasons.append(
            "No SEC SIC code was available and the issuer reports no pattern of concepts "
            "specific to any specialized business model.")

    result.standard_fcff_suitability = _suitability_for(result.profile)

    if result.standard_fcff_suitability != FcffSuitability.SUITABLE:
        result.findings.append({
            "code": DCF_CASH_FLOW_NOT_STANDARD_FCFF,
            "severity": "warning" if result.standard_fcff_suitability == FcffSuitability.LIMITED
                        else "error",
            "message": describe_cash_flow_limitation(result),
            "profile": result.profile,
            "sic": result.sic,
        })
    return result


def _suitability_for(profile: str) -> str:
    if profile in _SPECIALIZED:
        return FcffSuitability.NOT_SUITABLE
    if profile in (CashFlowValuationProfile.REIT_OR_SPECIALIZED,
                   CashFlowValuationProfile.OTHER_SPECIALIZED):
        return FcffSuitability.LIMITED
    if profile == CashFlowValuationProfile.UNKNOWN:
        # Not knowing is not a reason to refuse. The standard model stays in
        # use and the uncertainty is reported -- refusing every company whose
        # SIC lookup failed would make the tool useless on exactly the
        # smaller issuers where the metadata is thinnest.
        return FcffSuitability.SUITABLE
    return FcffSuitability.SUITABLE


def describe_cash_flow_limitation(classification: BusinessModelClassification) -> str:
    """Why operating cash flow less capital expenditure is not owner cash here."""
    profile = classification.profile
    if profile == CashFlowValuationProfile.BROKER_DEALER:
        detail = ("customer cash, segregated balances and receivables from and payables to "
                  "customers and clearing organisations move through operating cash flow, so "
                  "it largely measures customer money in transit rather than cash the business "
                  "generates for its owners")
    elif profile == CashFlowValuationProfile.BANK:
        detail = ("deposit-taking and lending are the business itself, so changes in deposits "
                  "and loans dominate operating cash flow rather than sitting outside it")
    elif profile == CashFlowValuationProfile.INSURER:
        detail = ("premiums are collected long before claims are paid, so operating cash flow "
                  "reflects the timing of the insurance cycle rather than period earnings")
    elif profile == CashFlowValuationProfile.FINANCIAL_INSTITUTION:
        detail = ("lending and funding flows run through operating cash flow, which therefore "
                  "does not measure cash available to the owners")
    else:
        detail = ("this business model's operating cash flow does not correspond to cash "
                  "available to the owners in the way the standard model assumes")
    return (
        f"Operating cash flow less capital expenditure is not owner free cash flow for a "
        f"{profile.replace('_', ' ').lower()}: {detail}. The figure remains arithmetically "
        f"correct and is reported as such, but it is not used as a discounted-cash-flow "
        f"input. Basis: {' '.join(classification.reasons)}")


def cash_flow_definition_note(classification: BusinessModelClassification) -> Optional[str]:
    """Section 18: label the economic definition rather than suppressing it.

    The number is still computed and still shown -- a reader is entitled to
    operating cash flow less capital expenditure for any company. What
    changes is that it is not called owner free cash flow and is not fed to
    the valuation model.
    """
    if classification.standard_fcff_suitability == FcffSuitability.SUITABLE:
        return None
    return (
        "operating cash flow less capital expenditure, which for this business model is not "
        "equivalent to owner free cash flow")


def may_enter_standard_fcff(classification: Optional[BusinessModelClassification]) -> bool:
    """Section 19's gate, as one question with one answer."""
    if classification is None:
        return True
    return classification.standard_fcff_suitability != FcffSuitability.NOT_SUITABLE
