"""Phase H.3 — finance/reconciliation.py: the one REAL cross-provider overlap
this project's architecture actually produces (Yahoo basic shares outstanding
vs. SEC diluted-weighted-average shares) -- see the module docstring for why
this is the only overlap worth reconciling given each capability routes to
exactly one provider.
"""

from finance.reconciliation import reconcile_facts


def _facts(yahoo_shares, sec_shares, fiscal_date="2025-09-27"):
    return {
        "overview": {"shares_outstanding": yahoo_shares},
        "statements": {"annual": {"balance_sheet": [
            {"dataset_id": "sec_company_facts", "fiscal_date": fiscal_date,
             "values": {"shares_outstanding": sec_shares} if sec_shares is not None else {}},
        ]}},
    }


def test_no_warning_when_share_counts_agree_closely():
    warnings = reconcile_facts(_facts(15_000_000_000, 15_004_697_000))
    assert warnings == []


def test_warning_when_share_counts_diverge_materially():
    warnings = reconcile_facts(_facts(14_594_180_000, 15_004_697_000))
    assert len(warnings) == 1
    assert "MARKET_CAP_SHARE_COUNT_MISMATCH" in warnings[0]
    assert "14,594,180,000" in warnings[0]
    assert "15,004,697,000" in warnings[0]


def test_no_warning_when_yahoo_share_count_is_missing():
    warnings = reconcile_facts(_facts(None, 15_004_697_000))
    assert warnings == []


def test_no_warning_when_statements_are_not_sec_sourced():
    """Alpha-Vantage-sourced balance sheets have dataset_id="income_statement"
    et al, not "sec_company_facts" -- this overlap only exists when SEC
    actually supplied the fundamentals."""
    facts = _facts(14_594_180_000, 20_000_000_000)
    facts["statements"]["annual"]["balance_sheet"][0]["dataset_id"] = "balance_sheet"
    warnings = reconcile_facts(facts)
    assert warnings == []


def test_no_warning_when_balance_sheet_is_empty():
    facts = {"overview": {"shares_outstanding": 100}, "statements": {"annual": {"balance_sheet": []}}}
    assert reconcile_facts(facts) == []


def test_handles_completely_empty_facts_gracefully():
    assert reconcile_facts({}) == []
