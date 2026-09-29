"""Phase H.3 — SEC XBRL facts -> the SAME normalized statement shape
finance/normalization.py already produces from Alpha Vantage.

This is the seam that lets finance/metrics.py, finance/dcf.py, and every
existing test that reads `facts["statements"]["annual"]["balance_sheet"][0]
["values"]["total_debt"]` keep working COMPLETELY UNCHANGED regardless of
which provider actually supplied the data — only this file knows SEC's shape
exists at all.

finance.xbrl_mapping.extract_statements() already resolved which concept won
for each field and enforced instant-vs-duration correctness; this file's only
job is renaming those normalized field names to match Alpha Vantage's
existing key names 1:1 (e.g. xbrl "assets" -> "total_assets", "stockholders_
equity" -> "shareholder_equity") and reusing the SAME
_derive_balance_sheet_aggregates() Alpha Vantage's own normalization already
uses for cash_and_short_term_investments / total_debt, so the debt/cash
policy is identical no matter the source.
"""

from typing import Dict, Tuple

from finance.normalization import (
    FinancialPeriod,
    NormalizedStatements,
    PeriodType,
    ValueBasis,
    _derive_balance_sheet_aggregates,
    _normalize_cash_flow_signs,
)
from finance.xbrl_mapping import extract_statements

# xbrl_mapping field name -> (which of the three statements it belongs to,
# the exact key name finance/normalization.py's Alpha-Vantage path already
# uses for the same concept). Anything not listed here (e.g. diluted_eps) is
# carried through UNCHANGED as an additive, harmless extra key.
_FIELD_TO_STATEMENT_AND_KEY = {
    "revenue": ("income_statement", "revenue"),
    "operating_income": ("income_statement", "operating_income"),
    "net_income": ("income_statement", "net_income"),
    "cash_and_cash_equivalents": ("balance_sheet", "cash_and_cash_equivalents"),
    "short_term_investments": ("balance_sheet", "short_term_investments"),
    "assets": ("balance_sheet", "total_assets"),
    "current_assets": ("balance_sheet", "current_assets"),
    "liabilities": ("balance_sheet", "total_liabilities"),
    "current_liabilities": ("balance_sheet", "current_liabilities"),
    "stockholders_equity": ("balance_sheet", "shareholder_equity"),
    "short_term_debt": ("balance_sheet", "short_term_debt"),
    "current_portion_of_long_term_debt": ("balance_sheet", "current_portion_of_long_term_debt"),
    "long_term_debt": ("balance_sheet", "long_term_debt"),
    "diluted_shares": ("balance_sheet", "shares_outstanding"),
    "operating_cash_flow": ("cash_flow", "operating_cash_flow"),
    "capital_expenditure": ("cash_flow", "capital_expenditure"),
    "depreciation_and_amortization": ("cash_flow", "depreciation_amortization"),
}

_STATEMENTS = ("income_statement", "balance_sheet", "cash_flow")


def normalize_sec_statements(company_facts: dict, symbol: str, sec_provenance: dict,
                             max_annual: int = 5, max_quarterly: int = 8
                             ) -> Tuple[NormalizedStatements, Dict[str, dict]]:
    """Returns (NormalizedStatements matching Alpha Vantage's exact contract,
    a SEPARATE per-fact provenance dict keyed by
    "sec.<annual|quarterly>.<statement>.<period index>.<field>" ->
    {concept, accession_number, fiscal_year, fiscal_period, form, filed} --
    additive detail for evidence-ID citation; nothing in NormalizedStatements
    itself changes shape to carry it, so `.values` stays plain floats exactly
    like the Alpha Vantage path.
    """
    result = NormalizedStatements(symbol=symbol, currency="USD")
    fact_provenance: Dict[str, dict] = {}

    for period_type, max_n, extract_kind in (
        (PeriodType.ANNUAL, max_annual, "annual"),
        (PeriodType.QUARTERLY, max_quarterly, "quarterly"),
    ):
        raw_periods = extract_statements(company_facts, extract_kind, max_n)
        buckets = {stmt: [] for stmt in _STATEMENTS}

        for index, period in enumerate(raw_periods):
            per_statement_values = {stmt: {} for stmt in _STATEMENTS}
            end_date = None
            for field_name, resolved in period["values"].items():
                mapping = _FIELD_TO_STATEMENT_AND_KEY.get(field_name)
                stmt, key = mapping if mapping else ("income_statement", field_name)
                per_statement_values[stmt][key] = resolved["value"]
                if resolved.get("end") and end_date is None:
                    end_date = resolved["end"]
                fact_provenance[f"sec.{extract_kind}.{stmt}.{index}.{key}"] = {
                    "concept": resolved["concept"],
                    "accession_number": resolved["accession_number"],
                    "fiscal_year": resolved["fiscal_year"],
                    "fiscal_period": resolved["fiscal_period"],
                    "form": resolved["form"],
                    "filed": resolved["filed"],
                }
            # Reuse Alpha Vantage's own aggregation policy verbatim -- the
            # SAME "sum only the known components, never guess a missing
            # one" total_debt/cash_and_short_term_investments logic, so a
            # DCF fed from SEC data applies an IDENTICAL debt/cash policy to
            # one fed from Alpha Vantage.
            _derive_balance_sheet_aggregates(per_statement_values["balance_sheet"])
            _normalize_cash_flow_signs(per_statement_values["cash_flow"])

            fiscal_date = end_date or f"FY{period['fiscal_year']}{period['fiscal_period']}"
            for stmt in _STATEMENTS:
                buckets[stmt].append(FinancialPeriod(
                    fiscal_date=fiscal_date, period_type=period_type, currency="USD",
                    values=per_statement_values[stmt], dataset_id="sec_company_facts",
                    basis=ValueBasis.REPORTED,
                ))

        target = result.annual if period_type == PeriodType.ANNUAL else result.quarterly
        for stmt in _STATEMENTS:
            target[stmt] = buckets[stmt]

    for stmt in _STATEMENTS:
        result.provenance[stmt] = dict(sec_provenance)

    return result, fact_provenance
