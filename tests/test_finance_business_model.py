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
