"""Phase H.3 — finance/sec_normalization.py: SEC XBRL facts normalized into
the EXACT SAME shape finance/normalization.py already produces from Alpha
Vantage (FinancialPeriod/NormalizedStatements) -- this is the seam that lets
finance/metrics.py and _dcf_inputs_from_facts work unchanged regardless of
provider. See docs/security/YAHOO_SEC_PROVIDER_REVIEW.md and this session's
live AAPL verification for the real-data proof; this file uses a small
hand-built fixture to check the mapping/derivation logic in isolation.
"""

from finance.normalization import FinancialPeriod, NormalizedStatements
from finance.sec_normalization import normalize_sec_statements

COMPANY_FACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 1000, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "NetIncomeLoss": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 150, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
                {"end": "2024-12-31", "val": 200, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "ShortTermInvestments": {"units": {"USD": [
                {"end": "2024-12-31", "val": 50, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "LongTermDebtCurrent": {"units": {"USD": [
                {"end": "2024-12-31", "val": 30, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "LongTermDebtNoncurrent": {"units": {"USD": [
                {"end": "2024-12-31", "val": 70, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "Assets": {"units": {"USD": [
                {"end": "2024-12-31", "val": 5000, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "StockholdersEquity": {"units": {"USD": [
                {"end": "2024-12-31", "val": 2000, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 100, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 300, "accn": "0001-24-000001",
                 "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
            ]}},
            # ShortTermBorrowings/DebtCurrent deliberately absent.
        }
    }
}

PROVENANCE = {"origin": "provider", "cache_status": "fresh", "stale": False}


def test_returns_a_normalized_statements_instance_with_matching_contract():
    statements, _fact_prov = normalize_sec_statements(COMPANY_FACTS, "TEST", PROVENANCE,
                                                       max_annual=5, max_quarterly=8)
    assert isinstance(statements, NormalizedStatements)
    assert set(statements.annual) == {"income_statement", "balance_sheet", "cash_flow"}
    period = statements.annual["balance_sheet"][0]
    assert isinstance(period, FinancialPeriod)
    assert isinstance(period.values, dict)
    assert all(isinstance(v, float) for v in period.values.values())  # plain floats, no rich objects


def test_field_names_are_renamed_to_match_alpha_vantages_exact_keys():
    statements, _fact_prov = normalize_sec_statements(COMPANY_FACTS, "TEST", PROVENANCE)
    bs_values = statements.annual["balance_sheet"][0].values
    # xbrl "assets" -> AV's "total_assets"; "stockholders_equity" -> "shareholder_equity"
    assert bs_values["total_assets"] == 5000.0
    assert bs_values["shareholder_equity"] == 2000.0
    assert bs_values["cash_and_cash_equivalents"] == 200.0
    assert bs_values["short_term_investments"] == 50.0
    # xbrl "diluted_shares" -> AV's "shares_outstanding" (the fallback slot
    # _dcf_inputs_from_facts already reads from the balance sheet)
    assert bs_values["shares_outstanding"] == 100.0


def test_derived_aggregates_use_the_same_policy_as_alpha_vantage():
    statements, _fact_prov = normalize_sec_statements(COMPANY_FACTS, "TEST", PROVENANCE)
    bs_values = statements.annual["balance_sheet"][0].values
    assert bs_values["cash_and_short_term_investments"] == 250.0  # 200 + 50
    # short_term_debt has no resolved XBRL fact -> the key is ABSENT entirely
    # (matching xbrl_mapping.extract_statements' own unresolved-field
    # contract), not present with a None value -- excluded from the sum via
    # .get(), never zero-filled.
    assert "short_term_debt" not in bs_values
    assert bs_values["total_debt"] == 100.0  # 30 (current) + 70 (long-term) only


def test_income_and_cash_flow_values_land_in_the_right_statement():
    statements, _fact_prov = normalize_sec_statements(COMPANY_FACTS, "TEST", PROVENANCE)
    inc_values = statements.annual["income_statement"][0].values
    cf_values = statements.annual["cash_flow"][0].values
    assert inc_values["revenue"] == 1000.0
    assert inc_values["net_income"] == 150.0
    assert "total_assets" not in inc_values  # balance-sheet fields never leak into income_statement
    assert cf_values["operating_cash_flow"] == 300.0


def test_fact_provenance_carries_the_exact_concept_and_accession():
    _statements, fact_prov = normalize_sec_statements(COMPANY_FACTS, "TEST", PROVENANCE)
    entry = fact_prov["sec.annual.balance_sheet.0.total_assets"]
    assert entry["concept"] == "Assets"
    assert entry["accession_number"] == "0001-24-000001"
    assert entry["form"] == "10-K"


def test_provenance_dict_is_attached_to_every_statement_type():
    statements, _fact_prov = normalize_sec_statements(COMPANY_FACTS, "TEST", PROVENANCE)
    for stmt in ("income_statement", "balance_sheet", "cash_flow"):
        assert statements.provenance[stmt]["origin"] == "provider"


def test_max_annual_bounds_the_period_count():
    statements, _fact_prov = normalize_sec_statements(COMPANY_FACTS, "TEST", PROVENANCE, max_annual=0)
    assert statements.annual["income_statement"] == []
