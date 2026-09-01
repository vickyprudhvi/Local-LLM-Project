"""Phase H.10, sections 16-19 — is the standard model the right model?

A live broker-dealer reported $566M of operating cash flow against $51M of
operating income. Operating cash flow less capital expenditure was computed,
labelled free cash flow, and DCF suitability returned SUITABLE. The number
was arithmetically right and economically meaningless: for a broker, most of
that cash is customer money in transit.

These tests fix the classification to the issuer's OWN filing metadata --
the SEC's SIC code and the concepts the company reports -- and assert that a
specialized business cannot reach the standard model. No ticker, name or
vendor sector label appears in any production path exercised here; the
vendor sector for that same broker-dealer read "Technology / Software".
"""

import pytest

from finance import business_model as B


P = B.CashFlowValuationProfile
S = B.FcffSuitability


def _facts(*concepts):
    return {"facts": {"us-gaap": {name: {} for name in concepts}}}


def _subs(sic, description="", **kw):
    out = {"sic": sic, "sicDescription": description}
    out.update(kw)
    return out


# ---------------------------------------------------------------------------
# The SIC matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sic, profile", [
    ("6021", P.BANK),
    ("6022", P.BANK),
    ("6199", P.FINANCIAL_INSTITUTION),
    ("6211", P.BROKER_DEALER),          # security brokers and dealers
    ("6221", P.BROKER_DEALER),
    ("6311", P.INSURER),
    ("6411", P.INSURER),
    ("6798", P.REIT_OR_SPECIALIZED),
    ("6500", P.REIT_OR_SPECIALIZED),
    ("2834", P.STANDARD_OPERATING_COMPANY),   # pharmaceutical preparations
    ("7372", P.STANDARD_OPERATING_COMPANY),   # prepackaged software
    ("3674", P.STANDARD_OPERATING_COMPANY),   # semiconductors
    ("5411", P.STANDARD_OPERATING_COMPANY),   # grocery stores
])
def test_sic_classification_matrix(sic, profile):
    result = B.classify_business_model(submissions=_subs(sic))
    assert result.profile == profile


@pytest.mark.parametrize("profile, suitability", [
    (P.BANK, S.NOT_SUITABLE),
    (P.BROKER_DEALER, S.NOT_SUITABLE),
    (P.INSURER, S.NOT_SUITABLE),
    (P.FINANCIAL_INSTITUTION, S.NOT_SUITABLE),
    (P.REIT_OR_SPECIALIZED, S.LIMITED),
    (P.OTHER_SPECIALIZED, S.LIMITED),
    (P.STANDARD_OPERATING_COMPANY, S.SUITABLE),
    (P.UNKNOWN, S.SUITABLE),
])
def test_suitability_follows_the_profile(profile, suitability):
    assert B._suitability_for(profile) == suitability  # noqa: SLF001


# ---------------------------------------------------------------------------
# Section 41 — the forbidden path
# ---------------------------------------------------------------------------

def test_a_broker_dealer_cannot_enter_the_standard_model():
    result = B.classify_business_model(submissions=_subs(
        "6211", "Security Brokers, Dealers & Flotation Companies"))
    assert result.profile == P.BROKER_DEALER
    assert result.standard_fcff_suitability == S.NOT_SUITABLE
    assert B.may_enter_standard_fcff(result) is False
    assert result.findings
    assert result.findings[0]["code"] == B.DCF_CASH_FLOW_NOT_STANDARD_FCFF


def test_an_ordinary_operating_company_is_unaffected():
    """The contract must not cost every normal company its valuation."""
    result = B.classify_business_model(submissions=_subs("3674", "Semiconductors"))
    assert result.profile == P.STANDARD_OPERATING_COMPANY
    assert B.may_enter_standard_fcff(result) is True
    assert result.findings == []


def test_a_missing_sic_does_not_block_the_model():
    """Not knowing is not a reason to refuse.

    Rejecting every issuer whose metadata lookup failed would disable the
    tool on exactly the smaller companies where metadata is thinnest.
    """
    result = B.classify_business_model(submissions=None, company_facts=None)
    assert result.profile == P.UNKNOWN
    assert B.may_enter_standard_fcff(result) is True


# ---------------------------------------------------------------------------
# Classification from what the issuer actually files
# ---------------------------------------------------------------------------

def test_filed_concepts_classify_when_the_sic_code_is_absent():
    result = B.classify_business_model(
        submissions=None,
        company_facts=_facts("PayablesToCustomers", "SecuritiesBorrowed",
                             "CashAndSecuritiesSegregatedUnderFederalAndOtherRegulations"))
    assert result.profile == P.BROKER_DEALER
    assert len(result.corroborating_concepts) >= 2


def test_filed_concepts_override_a_stale_sic_code():
    """A SIC code is assigned once; what a company tags this year is current."""
    result = B.classify_business_model(
        submissions=_subs("7372", "Prepackaged Software"),
        company_facts=_facts("Deposits", "InterestExpenseDeposits",
                             "LoansAndLeasesReceivableNetReportedAmount"))
    assert result.profile == P.BANK
    assert B.may_enter_standard_fcff(result) is False


def test_one_incidental_concept_does_not_reclassify_a_manufacturer():
    """A captive finance arm is not a bank.

    The threshold exists because a single tag appears for all sorts of
    reasons; a PATTERN of them describes the business.
    """
    result = B.classify_business_model(
        submissions=_subs("3711", "Motor Vehicles"),
        company_facts=_facts("ReceivablesFromCustomers", "Revenues", "OperatingIncomeLoss"))
    assert result.profile == P.STANDARD_OPERATING_COMPANY
    assert B.may_enter_standard_fcff(result) is True


def test_the_classification_says_why():
    result = B.classify_business_model(submissions=_subs(
        "6211", "Security Brokers, Dealers & Flotation Companies"))
    assert result.reasons
    assert "6211" in " ".join(result.reasons)


def test_the_limitation_explains_the_economics_not_just_the_label():
    result = B.classify_business_model(submissions=_subs("6211"))
    text = B.describe_cash_flow_limitation(result)
    assert "customer" in text.lower()
    assert "arithmetically correct" in text


def test_the_figure_is_relabelled_rather_than_suppressed():
    """Section 18: the number is still reported, under an honest name."""
    broker = B.classify_business_model(submissions=_subs("6211"))
    note = B.cash_flow_definition_note(broker)
    assert note and "not equivalent to owner free cash flow" in note

    ordinary = B.classify_business_model(submissions=_subs("3674"))
    assert B.cash_flow_definition_note(ordinary) is None


def test_classification_reads_concept_names_and_never_values():
    """Which things a company reports, not how large they are.

    A classifier that looked at magnitudes would drift with the business
    cycle; one that looks at the tag set does not.
    """
    small = B.classify_business_model(
        submissions=None, company_facts=_facts("Deposits", "InterestExpenseDeposits"))
    large = B.classify_business_model(
        submissions=None, company_facts=_facts("Deposits", "InterestExpenseDeposits"))
    assert small.profile == large.profile == P.BANK


def test_ifrs_filers_are_classified_from_their_own_taxonomy_block():
    result = B.classify_business_model(
        submissions=_subs("6211", "Security Brokers"),
        company_facts={"facts": {"ifrs-full": {"Revenue": {}}}})
    assert result.profile == P.BROKER_DEALER


# ---------------------------------------------------------------------------
# A positive non-financial SIC code is not overridden by concept evidence
# ---------------------------------------------------------------------------
#
# Found by live verification of this phase's wiring. A large telecom (SEC SIC
# 4813, "Telephone Communications") was classified BANK because it reports
# `LoansAndLeasesReceivableNetReportedAmount` and
# `ProvisionForLoanLeaseAndOtherLosses` for its device-payment and equipment-
# lease receivables. The valuation was then refused as
# NOT_APPLICABLE_FOR_BUSINESS_MODEL for a company a discounted cash flow fits
# perfectly well.
#
# The general invariant that was missing: concept evidence may CONFIRM a
# classification, or SUPPLY one where the SIC code is absent or is itself
# financial. It may not OVERRIDE a code that positively places the issuer in
# a non-financial division. The module's rationale for allowing an override
# -- "a SIC code is assigned once and can lag what a company has become" --
# describes an issuer whose code is missing or already financial; it does not
# describe one the Commission has placed in Telephone Communications.
#
# This is the same case the module's own comment already anticipated ("a
# manufacturer with a captive finance arm reports some receivables
# concepts") -- the threshold of two was simply too low to tell the two
# apart -- and no count can, because the question is not how MANY financial
# concepts a captive finance arm reports but WHICH. Only a bank takes
# deposits; anyone with a finance arm books loans receivable.

def _submissions(sic, description):
    return {"sic": sic, "sicDescription": description}


def _facts(*concepts):
    return {"facts": {"us-gaap": {name: {"units": {}} for name in concepts}}}


def test_a_captive_finance_arm_does_not_make_a_telecom_a_bank():
    from finance.business_model import CashFlowValuationProfile, classify_business_model

    result = classify_business_model(
        _submissions("4813", "Telephone Communications (No Radiotelephone)"),
        _facts("LoansAndLeasesReceivableNetReportedAmount",
               "ProvisionForLoanLeaseAndOtherLosses", "Revenues", "Assets"))
    assert result.profile == CashFlowValuationProfile.STANDARD_OPERATING_COMPANY
    assert result.standard_fcff_suitability == "SUITABLE"


def test_the_financial_concepts_are_still_reported_as_context():
    """Refusing the override is not the same as ignoring the evidence. The
    concepts stay on the classification, and the reason says what was seen
    and why it did not decide the verdict."""
    from finance.business_model import classify_business_model

    result = classify_business_model(
        _submissions("4813", "Telephone Communications (No Radiotelephone)"),
        _facts("LoansAndLeasesReceivableNetReportedAmount",
               "ProvisionForLoanLeaseAndOtherLosses"))
    assert set(result.corroborating_concepts) == {
        "LoansAndLeasesReceivableNetReportedAmount",
        "ProvisionForLoanLeaseAndOtherLosses"}
    assert any("4813" in reason for reason in result.reasons)


def test_concept_evidence_still_classifies_when_no_sic_code_exists():
    """The override exists for a real case and must survive: an issuer whose
    submissions record carries no SIC code at all is classified by what it
    files."""
    from finance.business_model import CashFlowValuationProfile, classify_business_model

    result = classify_business_model(
        {}, _facts("PayablesToCustomers", "ReceivablesFromCustomers", "SecuritiesBorrowed"))
    assert result.profile == CashFlowValuationProfile.BROKER_DEALER


def test_concept_evidence_still_refines_a_financial_sic_code():
    """Within financial services the SIC code can genuinely lag: a
    nondepository-credit code (6199) on an issuer filing customer payables
    and segregated cash is a broker-dealer, and the concepts say so."""
    from finance.business_model import CashFlowValuationProfile, classify_business_model

    result = classify_business_model(
        _submissions("6199", "Finance Services"),
        _facts("PayablesToCustomers", "ReceivablesFromCustomers",
               "CashAndSecuritiesSegregatedUnderFederalAndOtherRegulations"))
    assert result.profile in (CashFlowValuationProfile.BROKER_DEALER,
                              CashFlowValuationProfile.FINANCIAL_INSTITUTION)
    assert result.standard_fcff_suitability == "NOT_SUITABLE"


def test_a_non_financial_sic_still_yields_to_a_definitive_filing_pattern():
    """The rule is about WHICH concepts, not about ignoring evidence.

    An issuer whose SIC is non-financial but which reports DEPOSITS is not a
    manufacturer with a finance arm; it is a bank whose code is wrong. Only a
    bank takes deposits, and no amount of captive financing produces that
    line -- which is why the test is a concept identity and not a count.
    """
    from finance.business_model import CashFlowValuationProfile, classify_business_model

    result = classify_business_model(
        _submissions("4813", "Telephone Communications (No Radiotelephone)"),
        _facts("Deposits", "InterestExpenseDeposits",
               "InterestAndDividendIncomeOperating",
               "FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell",
               "ProvisionForLoanLeaseAndOtherLosses"))
    assert result.profile == CashFlowValuationProfile.BANK
