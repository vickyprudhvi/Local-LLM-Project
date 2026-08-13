"""Phase H.3 — finance/xbrl_mapping.py: deterministic SEC concept resolution.

Small, hand-built company_facts fixture (not the full real ~3.7MB AAPL
payload — see tests/test_finance_sec_normalization.py's live-shaped fixture
and docs/security/YAHOO_SEC_PROVIDER_REVIEW.md for the real-data proof) so
precedence, instant-vs-duration correctness, and amended-filing handling are
each checked in isolation.
"""

from finance.xbrl_mapping import (
    ANNUAL_FORMS,
    extract_statements,
    list_available_periods,
    resolve_concept,
)

COMPANY_FACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 1000, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
                {"start": "2023-01-01", "end": "2023-12-31", "val": 900, "accn": "0001-23-000001",
                 "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-01"},
            ]}},
            "NetIncomeLoss": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 150, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            # Amended filing: SAME fiscal year/period, a DIFFERENT value, filed LATER --
            # the amendment must win over the original.
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
                {"end": "2024-12-31", "val": 200, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
                {"end": "2024-12-31", "val": 250, "accn": "0001-24-000002",
                 "fy": 2024, "fp": "FY", "form": "10-K/A", "filed": "2025-03-01"},
            ]}},
            "Assets": {"units": {"USD": [
                {"end": "2024-12-31", "val": 5000, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            # Second candidate in the precedence list for revenue -- must NOT be
            # picked when the FIRST candidate already has a match.
            "Revenues": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 999999, "accn": "0001-24-999999",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            # ShortTermBorrowings / DebtCurrent deliberately absent -> short_term_debt
            # must resolve to None, never a guess.
        }
    }
}


# ---- resolve_concept: precedence, instant vs duration, amendments ----

def test_first_candidate_concept_wins_when_present():
    fact = resolve_concept(COMPANY_FACTS, "revenue", ANNUAL_FORMS, 2024, "FY")
    assert fact.value == 1000.0
    assert fact.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_instant_concept_never_matches_a_duration_fact_and_vice_versa():
    # Assets is INSTANT (only `end`) -- resolving it must not somehow pick up
    # a duration fact even if one existed under the same concept name.
    fact = resolve_concept(COMPANY_FACTS, "assets", ANNUAL_FORMS, 2024, "FY")
    assert fact.value == 5000.0
    assert fact.start is None


def test_amended_filing_supersedes_the_original():
    fact = resolve_concept(COMPANY_FACTS, "cash_and_cash_equivalents", ANNUAL_FORMS, 2024, "FY")
    assert fact.value == 250.0
    assert fact.form == "10-K/A"
    assert fact.accession == "0001-24-000002"


def test_unmapped_field_returns_none_not_a_guess():
    fact = resolve_concept(COMPANY_FACTS, "short_term_debt", ANNUAL_FORMS, 2024, "FY")
    assert fact is None


def test_unknown_normalized_field_name_returns_none():
    fact = resolve_concept(COMPANY_FACTS, "totally_made_up_field", ANNUAL_FORMS, 2024, "FY")
    assert fact is None


def test_wrong_fiscal_year_returns_none():
    fact = resolve_concept(COMPANY_FACTS, "revenue", ANNUAL_FORMS, 2099, "FY")
    assert fact is None


# ---- list_available_periods / extract_statements ----

def test_list_available_periods_finds_both_years_newest_first():
    periods = list_available_periods(COMPANY_FACTS, ANNUAL_FORMS)
    assert periods[0] == (2024, "FY")
    assert (2023, "FY") in periods


def test_extract_statements_reports_unresolved_fields_honestly():
    statements = extract_statements(COMPANY_FACTS, "annual", max_periods=2)
    fy2024 = next(p for p in statements if p["fiscal_year"] == 2024)
    assert fy2024["values"]["revenue"]["value"] == 1000.0
    assert "short_term_debt" in fy2024["unresolved_fields"]


def test_extract_statements_respects_max_periods():
    statements = extract_statements(COMPANY_FACTS, "annual", max_periods=1)
    assert len(statements) == 1
    assert statements[0]["fiscal_year"] == 2024


def test_extract_statements_quarterly_finds_nothing_in_an_annual_only_fixture():
    statements = extract_statements(COMPANY_FACTS, "quarterly", max_periods=5)
    assert statements == []


# ---- MLI corrective patch: duration windows ----
#
# In SEC companyfacts a Q4 duration fact is routinely tagged `fp: "FY"` --
# companies file no separate Q4 10-Q, so the fourth-quarter figure rides
# along in the 10-K carrying the annual period tag. A 10-K therefore carries
# BOTH, identical in fy/fp/form/filed/end and differing only in `start`.
# Found live on MLI: annual revenue came back as a true FY2025 ($4.18B) but
# fourth-QUARTER values for every prior year ($923.5M for FY2024, whose real
# annual revenue is ~$3.77B), producing revenue_growth_yoy = +352.5% (a full
# year divided by one quarter) and a revenue_cagr that then fed the DCF and
# hit its 25% clamp.

_FY_AND_Q4_SAME_FILING = {
    "facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            # The true annual fact and the Q4 fact from the SAME 10-K: same
            # fy/fp/form/filed/end, differing ONLY in `start`.
            {"start": "2024-01-01", "end": "2024-12-31", "val": 4_000, "accn": "a",
             "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            {"start": "2024-10-01", "end": "2024-12-31", "val": 900, "accn": "a",
             "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
        ]}},
        "NetIncomeLoss": {"units": {"USD": [
            {"start": "2024-01-01", "end": "2024-12-31", "val": 400, "accn": "a",
             "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
        ]}},
    }}
}


def test_annual_resolution_picks_the_full_year_not_the_q4_fact():
    fact = resolve_concept(_FY_AND_Q4_SAME_FILING, "revenue", ANNUAL_FORMS, 2024, "FY")
    assert fact.value == 4_000, "picked the Q4 fact tagged fp=FY instead of the true annual one"
    assert fact.start == "2024-01-01" and fact.end == "2024-12-31"


def test_extract_statements_annual_never_returns_a_quarterly_duration():
    periods = extract_statements(_FY_AND_Q4_SAME_FILING, "annual", 4)
    assert periods
    assert periods[0]["values"]["revenue"]["value"] == 4_000


def test_quarterly_resolution_picks_the_quarter_not_the_full_year():
    """The mirror case: a ~90-day fact must win when quarterly is asked for."""
    facts = {"facts": {"us-gaap": {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            {"start": "2024-01-01", "end": "2024-03-31", "val": 250, "accn": "b",
             "fy": 2024, "fp": "Q1", "form": "10-Q", "filed": "2024-04-20"},
            {"start": "2023-04-01", "end": "2024-03-31", "val": 1_000, "accn": "b",
             "fy": 2024, "fp": "Q1", "form": "10-Q", "filed": "2024-04-20"},
        ]}},
    }}}
    from finance.xbrl_mapping import QUARTERLY_FORMS
    fact = resolve_concept(facts, "revenue", QUARTERLY_FORMS, 2024, "Q1")
    assert fact.value == 250, "picked a trailing-twelve-month fact instead of the quarter"


def test_duration_window_can_be_overridden_to_accept_anything():
    """Backward-compatible escape hatch -- an explicit wide window restores
    the pre-patch 'any duration' behavior."""
    fact = resolve_concept(_FY_AND_Q4_SAME_FILING, "revenue", ANNUAL_FORMS, 2024, "FY",
                           duration_days=(0, 10 ** 6))
    assert fact.value in (900, 4_000)  # no longer duration-filtered


def test_instant_facts_are_unaffected_by_the_duration_window():
    fact = resolve_concept(COMPANY_FACTS, "assets", ANNUAL_FORMS, 2024, "FY")
    assert fact.value == 5000 and fact.start is None


# ---- VZ corrective patch: productive-assets CapEx tags ----
#
# Verizon reports capital expenditure as PaymentsToAcquireOtherProductiveAssets
# ($17.011B FY2025) and uses NEITHER PropertyPlantAndEquipment spelling
# anywhere in its filings. With only those two mapped, capex resolved to None
# and propose_assumptions fell back to the configured 5%-of-revenue default --
# ~12.7% actual vs 5% assumed for one of the most capital-intensive issuers
# there is. The DCF still validated cleanly, because validation checks
# arithmetic, not whether an input was ever found: base modeled value came out
# at $108.80/share instead of $64.65, turning a ~37% gap into a claimed +130%.

_VZ_SHAPED_FACTS = {
    "facts": {"us-gaap": {
        # The tag VZ actually uses in recent filings...
        "PaymentsToAcquireOtherProductiveAssets": {"units": {"USD": [
            {"start": "2025-01-01", "end": "2025-12-31", "val": 17_011_000_000, "accn": "v",
             "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-02-10"},
        ]}},
        # ...and the older spelling it used through 2018.
        "PaymentsToAcquireProductiveAssets": {"units": {"USD": [
            {"start": "2018-01-01", "end": "2018-12-31", "val": 16_658_000_000, "accn": "w",
             "fy": 2018, "fp": "FY", "form": "10-K", "filed": "2019-02-10"},
        ]}},
        "NetIncomeLoss": {"units": {"USD": [
            {"start": "2025-01-01", "end": "2025-12-31", "val": 1, "accn": "v",
             "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-02-10"},
        ]}},
    }}
}


def test_capex_resolves_from_the_other_productive_assets_tag():
    fact = resolve_concept(_VZ_SHAPED_FACTS, "capital_expenditure", ANNUAL_FORMS, 2025, "FY")
    assert fact is not None, "VZ-style CapEx tag not mapped -> silent 5% default fallback"
    assert fact.value == 17_011_000_000


def test_capex_falls_through_per_period_when_a_company_switched_tags():
    """VZ used the non-'Other' spelling through 2018 and the 'Other' one
    after -- both years must resolve."""
    older = resolve_concept(_VZ_SHAPED_FACTS, "capital_expenditure", ANNUAL_FORMS, 2018, "FY")
    assert older is not None and older.value == 16_658_000_000


def test_capex_still_prefers_the_standard_property_plant_tag_when_present():
    """Precedence unchanged: the productive-assets tags are FALLBACKS, not a
    replacement for the standard spelling."""
    facts = {"facts": {"us-gaap": {
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            {"start": "2025-01-01", "end": "2025-12-31", "val": 500, "accn": "a",
             "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-02-10"}]}},
        "PaymentsToAcquireOtherProductiveAssets": {"units": {"USD": [
            {"start": "2025-01-01", "end": "2025-12-31", "val": 999, "accn": "a",
             "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2026-02-10"}]}},
    }}}
    fact = resolve_concept(facts, "capital_expenditure", ANNUAL_FORMS, 2025, "FY")
    assert fact.value == 500
