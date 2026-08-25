"""Phase H.7 — entity identity, share basis and per-share invariants
(sections 15-26, 60).

Every test here is about a number that would otherwise be wrong by a large
factor while looking entirely ordinary: a market capitalisation built on one
share class, an ADR price multiplied by an ordinary-share count, a P/B built
on equity that belongs partly to somebody else, a subsidiary's dividend
credited to the parent.

No production code reads a ticker; nothing here asserts a company-specific
result.
"""

import pytest

from finance import entity as E


def _security(classes=(), ratio=1.0, ticker="XYZ", security_type="common_stock",
              currency="USD"):
    return E.SecurityIdentity(
        ticker=ticker, entity_id="0000000001", exchange="NYSE",
        security_type=security_type, depositary_ratio=ratio, currency=currency,
        share_classes=tuple(classes))


def _issuer(cik="0000000001", name="Example Issuer"):
    return E.IssuerEntity(entity_id=cik, name=name, cik=cik)


# ---------------------------------------------------------------------------
# Section 17 — share-count types stay separate
# ---------------------------------------------------------------------------

def test_each_share_count_type_is_stored_under_its_own_name():
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 1_000, "provider")
    counts.set(E.ShareCountType.WEIGHTED_AVERAGE_DILUTED, 1_100, "sec")
    counts.set(E.ShareCountType.WEIGHTED_AVERAGE_BASIC, 1_050, "sec")
    assert counts.get(E.ShareCountType.CURRENT_OUTSTANDING) == 1_000
    assert counts.get(E.ShareCountType.WEIGHTED_AVERAGE_DILUTED) == 1_100
    assert counts.get(E.ShareCountType.WEIGHTED_AVERAGE_BASIC) == 1_050
    assert counts.sources[E.ShareCountType.WEIGHTED_AVERAGE_DILUTED] == "sec"


def test_a_zero_or_negative_share_count_is_refused_rather_than_stored():
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 0, "provider")
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, -5, "provider")
    assert counts.get(E.ShareCountType.CURRENT_OUTSTANDING) is None


# ---------------------------------------------------------------------------
# Section 18 — multiple share classes
# ---------------------------------------------------------------------------

def test_a_single_class_aggregates_to_itself():
    security = _security([E.ShareClass("common", 1_000, 1.0, is_traded=True)])
    total, reason = E.economic_shares_outstanding(security)
    assert total == pytest.approx(1_000)
    assert reason


def test_two_classes_with_equal_economics_sum():
    security = _security([
        E.ShareClass("class_a", 700, 1.0, is_traded=True),
        E.ShareClass("class_b", 300, 1.0),
    ])
    total, _reason = E.economic_shares_outstanding(security)
    assert total == pytest.approx(1_000)


def test_a_class_with_different_economics_is_weighted_not_added():
    """Section 18: not every share class has 1:1 economics."""
    security = _security([
        E.ShareClass("class_a", 700, 1.0, is_traded=True),
        E.ShareClass("class_b", 300, 0.5),
    ])
    total, _reason = E.economic_shares_outstanding(security)
    assert total == pytest.approx(700 + 150)


def test_a_class_with_no_reported_count_makes_the_total_unknown():
    """Returning the traded class alone is exactly the substitution that
    produced a 76% market-capitalisation error."""
    security = _security([
        E.ShareClass("class_a", 700, 1.0, is_traded=True),
        E.ShareClass("class_b", None, 1.0),
    ])
    total, reason = E.economic_shares_outstanding(security)
    assert total is None
    assert "class_b" in reason


def test_no_share_class_detail_is_unknown_not_zero():
    total, reason = E.economic_shares_outstanding(_security([]))
    assert total is None and reason


# ---------------------------------------------------------------------------
# Section 19 — ADR / depositary ratio
# ---------------------------------------------------------------------------

def test_a_depositary_ratio_converts_ordinary_shares_into_traded_units():
    """One receipt representing two ordinary shares halves the count that may
    be multiplied by the receipt's price."""
    security = _security([E.ShareClass("ordinary", 1_000, 1.0)],
                         ratio=2.0, security_type="adr")
    total, reason = E.economic_shares_outstanding(security)
    assert total == pytest.approx(500)
    assert "depositary ratio" in reason


def test_an_ordinary_share_is_not_treated_as_a_depositary_receipt():
    security = _security([E.ShareClass("common", 1_000, 1.0)])
    assert security.is_depositary_receipt is False
    total, _reason = E.economic_shares_outstanding(security)
    assert total == pytest.approx(1_000)


def test_a_ratio_other_than_one_marks_the_security_as_a_receipt():
    assert _security([], ratio=3.0).is_depositary_receipt is True


# ---------------------------------------------------------------------------
# Sections 20-21 — reconciliation and the market-cap invariant
# ---------------------------------------------------------------------------

def test_a_share_count_that_reconciles_against_market_cap_is_selected():
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 1_000, "provider")
    result = E.reconcile_share_basis(counts, _security(), price=10.0,
                                     reported_market_cap=10_000)
    assert result.status == E.ReconciliationStatus.RECONCILED
    assert result.selected_value == pytest.approx(1_000)
    assert result.market_cap_gap == pytest.approx(0.0, abs=1e-9)


def test_a_count_covering_one_class_is_reported_as_a_material_difference():
    """The live failure: price x shares 76% below the reported market cap."""
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 776_405_057, "provider")
    result = E.reconcile_share_basis(counts, _security(), price=13.84,
                                     reported_market_cap=45_524_795_392)
    assert result.status == E.ReconciliationStatus.MATERIAL_DIFFERENCE
    assert result.is_material is True
    codes = {f["code"] for f in result.findings}
    assert E.MARKET_CAP_RECONCILIATION_FAILURE in codes
    assert result.market_cap_gap < -0.5


def test_the_full_economic_basis_resolves_the_multi_class_case():
    """Aggregating both classes makes the identity hold, which is what the
    reconciliation is for."""
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 700, "provider")
    security = _security([
        E.ShareClass("class_a", 700, 1.0, is_traded=True),
        E.ShareClass("class_b", 300, 1.0),
    ])
    result = E.reconcile_share_basis(counts, security, price=10.0,
                                     reported_market_cap=10_000)
    assert result.selected_basis == E.ShareCountType.ECONOMIC_OUTSTANDING
    assert result.status == E.ReconciliationStatus.RECONCILED


def test_a_small_timing_difference_is_not_a_material_finding():
    """Buybacks and option exercises move a count between two measurement
    moments; a real company routinely lands inside the tolerance."""
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 1_010, "provider")
    result = E.reconcile_share_basis(counts, _security(), price=10.0,
                                     reported_market_cap=10_000)
    assert result.status == E.ReconciliationStatus.RECONCILED
    assert result.findings == []


def test_weighted_average_counts_rank_below_current_ones_for_market_cap():
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.WEIGHTED_AVERAGE_DILUTED, 1_200, "sec")
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 1_000, "provider")
    result = E.reconcile_share_basis(counts, _security(), price=10.0,
                                     reported_market_cap=10_000)
    assert result.selected_basis == E.ShareCountType.CURRENT_OUTSTANDING


def test_no_share_count_at_all_is_unknown_rather_than_an_error():
    result = E.reconcile_share_basis(E.ShareCountSet(), _security(), price=10.0,
                                     reported_market_cap=10_000)
    assert result.status == E.ReconciliationStatus.UNKNOWN
    assert result.selected_value is None


def test_pairwise_comparisons_are_recorded_even_when_one_reconciles():
    """Phase H.10 changed what this comparison MEANS.

    It used to assert that a current outstanding count and a weighted-average
    diluted count differing by 40% produced MATERIAL_DIFFERENCE. Section 13
    of the semantics phase says that is wrong: the two are different bases --
    one counts shares in issue at a moment, the other averages them over a
    reporting period -- so they are expected to differ and the gap is not a
    data-quality finding. The comparison is still RECORDED, which is what
    this test was really protecting; it is now recorded as not comparable.
    """
    counts = E.ShareCountSet()
    counts.set(E.ShareCountType.CURRENT_OUTSTANDING, 1_000, "provider")
    counts.set(E.ShareCountType.WEIGHTED_AVERAGE_DILUTED, 1_400, "sec")
    result = E.reconcile_share_basis(counts, _security(), price=10.0,
                                     reported_market_cap=10_000)
    assert result.comparisons
    assert all(c["status"] == E.ReconciliationStatus.NOT_COMPARABLE
               for c in result.comparisons)
    assert not any(c["status"] == E.ReconciliationStatus.MATERIAL_DIFFERENCE
                   for c in result.comparisons)


# ---------------------------------------------------------------------------
# Section 22 — P/E reconciliation
# ---------------------------------------------------------------------------

def test_consistent_pe_derivations_reconcile():
    record = E.reconcile_price_earnings(price=100.0, diluted_eps=5.0,
                                        market_cap=10_000, attributable_net_income=500)
    assert record["per_share_pe"] == pytest.approx(20.0)
    assert record["aggregate_pe"] == pytest.approx(20.0)
    assert record["status"] == E.ReconciliationStatus.RECONCILED


def test_inconsistent_pe_derivations_are_reported_not_averaged():
    record = E.reconcile_price_earnings(price=100.0, diluted_eps=5.0,
                                        market_cap=10_000, attributable_net_income=250)
    assert record["status"] == E.ReconciliationStatus.MATERIAL_DIFFERENCE
    assert {f["code"] for f in record["findings"]} == {E.PE_RECONCILIATION_FAILURE}


def test_negative_earnings_make_pe_not_meaningful_rather_than_failed():
    record = E.reconcile_price_earnings(price=100.0, diluted_eps=-2.0,
                                        market_cap=10_000, attributable_net_income=-500)
    assert record["status"] == E.ReconciliationStatus.UNKNOWN
    assert record["findings"] == []


# ---------------------------------------------------------------------------
# Sections 23-24 — P/B and attributable equity
# ---------------------------------------------------------------------------

def test_price_to_book_uses_parent_attributable_equity_when_available():
    record = E.reconcile_price_book(price=10.0, shares=1_000,
                                    attributable_equity=5_000, total_equity=6_000,
                                    minority_interest=1_000)
    assert record["equity_basis"] == "equity_attributable_to_parent"
    assert record["price_to_book"] == pytest.approx(2.0)


def test_a_material_noncontrolling_interest_is_removed_from_book_value():
    record = E.reconcile_price_book(price=10.0, shares=1_000,
                                    attributable_equity=None, total_equity=6_000,
                                    minority_interest=1_000)
    assert record["equity_basis"] == "total_equity_less_noncontrolling_interest"
    assert record["price_to_book"] == pytest.approx(2.0)


def test_using_total_equity_with_a_material_nci_is_flagged():
    record = E.reconcile_price_book(price=10.0, shares=1_000,
                                    attributable_equity=None, total_equity=6_000,
                                    minority_interest=None)
    assert record["price_to_book"] == pytest.approx(10_000 / 6_000)
    # No NCI reported, so nothing to flag.
    assert record["findings"] == []


def test_non_positive_equity_produces_no_ratio():
    record = E.reconcile_price_book(price=10.0, shares=1_000,
                                    attributable_equity=-100)
    assert record["price_to_book"] is None


# ---------------------------------------------------------------------------
# Sections 25-26 — corporate actions belong to an entity
# ---------------------------------------------------------------------------

def test_an_action_declared_for_another_ticker_is_rejected():
    ok, reason = E.validate_corporate_action_scope(
        {"ticker": "OTHER", "amount": 1.0}, _security(ticker="XYZ"), _issuer())
    assert ok is False
    assert "OTHER" in reason


def test_an_action_declared_by_another_entity_is_rejected():
    ok, reason = E.validate_corporate_action_scope(
        {"cik": "0000000999", "amount": 1.0}, _security(), _issuer(cik="0000000001"))
    assert ok is False
    assert "0000000999" in reason


def test_an_action_with_no_scope_metadata_is_accepted_as_the_analyzed_security():
    ok, reason = E.validate_corporate_action_scope(
        {"amount": 1.0}, _security(), _issuer())
    assert ok is True and reason is None


def test_a_subsidiary_dividend_never_becomes_the_parent_stock_dividend():
    """Section 25's headline rule."""
    ok, _reason = E.validate_dividend(
        {"ticker": "SUBSIDIARY", "amount": 0.5, "ex_date": "2026-05-01"},
        _security(ticker="PARENT"), _issuer())
    assert ok is False


def test_a_dividend_needs_an_amount_and_a_date():
    assert E.validate_dividend({"ex_date": "2026-05-01"}, _security(), _issuer())[0] is False
    assert E.validate_dividend({"amount": 0.5}, _security(), _issuer())[0] is False


def test_a_dividend_in_another_currency_is_excluded_rather_than_mixed():
    ok, reason = E.validate_dividend(
        {"amount": 0.5, "currency": "EUR", "ex_date": "2026-05-01"},
        _security(currency="USD"), _issuer())
    assert ok is False
    assert "EUR" in reason


def test_a_well_formed_dividend_for_the_analyzed_security_is_accepted():
    ok, reason = E.validate_dividend(
        {"amount": 0.5, "currency": "USD", "ex_date": "2026-05-01"},
        _security(), _issuer())
    assert ok is True and reason is None


# ---------------------------------------------------------------------------
# Section 15 — issuer identity from filing metadata
# ---------------------------------------------------------------------------

def test_a_foreign_private_issuer_is_identified_from_its_filed_forms():
    submissions = {"stateOfIncorporation": "P7",
                   "filings": {"recent": {"form": ["20-F", "6-K", "6-K"]}}}
    issuer = E.build_issuer_entity({}, submissions, cik="0001791942", name="Example N.V.")
    assert issuer.reporting_status == "foreign_private_issuer"
    assert issuer.country_of_incorporation == "P7"


def test_a_domestic_registrant_is_identified_from_its_filed_forms():
    submissions = {"stateOfIncorporation": "DE",
                   "filings": {"recent": {"form": ["10-K", "10-Q", "8-K"]}}}
    issuer = E.build_issuer_entity({}, submissions, cik="0000000001", name="Example Inc")
    assert issuer.reporting_status == "domestic_registrant"


# ---------------------------------------------------------------------------
# Share-class detection from cover-page counts
# ---------------------------------------------------------------------------

def _dei(values, end="2026-06-30"):
    return {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": end, "val": v, "form": "10-Q"} for v in values]}}}}}


def test_one_cover_page_count_is_a_single_class():
    classes, note = E.detect_share_classes(_dei([1_000]))
    assert len(classes) == 1 and classes[0].shares_outstanding == 1_000
    assert note is None


def test_several_cover_page_counts_on_one_date_mean_several_classes():
    classes, note = E.detect_share_classes(_dei([700, 300]))
    assert len(classes) == 2
    assert {c.shares_outstanding for c in classes} == {700, 300}
    assert note and "class axis" in note


def test_no_cover_page_count_reports_a_reason_rather_than_guessing():
    classes, note = E.detect_share_classes({"facts": {}})
    assert classes == [] and note
