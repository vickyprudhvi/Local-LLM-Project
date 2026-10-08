"""Phase H.32 -- borrowing capacity is not funded debt.

Live-traced gap: a post-balance-sheet credit facility was rendered as
"additional debt issuance" even though the filing established new borrowing
capacity and explicitly stated no loans were outstanding. `classify_security_
event`'s TYPE output is unchanged (an item-2.03 facility is still debt-shaped,
exactly as the gated V2 reader's own ground truth keeps a committed-but-
undrawn facility typed ISSUER_DEBT_ISSUANCE) -- what changes is that
`describe_event`/`event_impact` now require their own evidence that funds
actually moved before asserting funded debt, rather than inferring it from
the event TYPE or the mere existence of an item-2.03 filing.
"""

import pytest

from finance import structural_breaks as SB

_T = SB.PostBalanceSheetEventType


def _submissions(rows):
    """rows: (form, items, filed[, description])."""
    return {"filings": {"recent": {
        "form": [r[0] for r in rows],
        "items": [r[1] for r in rows],
        "filingDate": [r[2] for r in rows],
        "accessionNumber": [f"0000-26-{i}" for i, _ in enumerate(rows)],
    }}}


# ---------------------------------------------------------------------------
# A. $20B delayed-draw term-loan facility, undrawn -> capacity only, no
#    funded-debt impact.
# ---------------------------------------------------------------------------

def test_A_undrawn_delayed_draw_term_loan_is_capacity_only():
    description = ("the Company entered into a $20.0 billion delayed-draw "
                   "term loan facility; no amounts have been drawn under "
                   "the facility as of the date of this filing")
    funded = SB._detect_funded_evidence(description)
    assert funded is False
    assert SB.event_impact(_T.ISSUER_DEBT_ISSUANCE, funded)["affects_debt"] is False
    text = SB.describe_event(_T.ISSUER_DEBT_ISSUANCE, funded)
    assert "issuance of debt" not in text
    assert "established" in text


# ---------------------------------------------------------------------------
# B. $8B revolver, no loans outstanding -> not debt issuance.
# ---------------------------------------------------------------------------

def test_B_revolver_with_no_loans_outstanding_is_not_debt_issuance():
    description = ("the Company entered into an $8.0 billion revolving "
                   "credit facility; no loans were outstanding under the "
                   "facility immediately after giving effect to this filing")
    funded = SB._detect_funded_evidence(description)
    assert funded is False
    impact = SB.event_impact(_T.ISSUER_DEBT_ISSUANCE, funded)
    assert impact["affects_debt"] is False
    assert impact["affects_cash"] is False
    assert "issuance of debt" not in SB.describe_event(_T.ISSUER_DEBT_ISSUANCE, funded)


# ---------------------------------------------------------------------------
# C. facility with $2B explicitly drawn -> funded borrowing.
# ---------------------------------------------------------------------------

def test_C_facility_with_explicit_draw_is_funded_borrowing():
    description = ("the Company entered into a term loan agreement and drew "
                   "$2.0 billion under the facility, receiving net proceeds "
                   "of approximately $1.98 billion")
    funded = SB._detect_funded_evidence(description)
    assert funded is True
    impact = SB.event_impact(_T.ISSUER_DEBT_ISSUANCE, funded)
    assert impact["affects_debt"] is True
    assert impact["affects_cash"] is True
    assert "issuance of debt" in SB.describe_event(_T.ISSUER_DEBT_ISSUANCE, funded)


# ---------------------------------------------------------------------------
# D. notes issued with a stated principal amount -> debt issuance (the
#    positive case: real funded-issuance language must still be recognized).
# ---------------------------------------------------------------------------

def test_D_notes_issued_with_principal_amount_is_debt_issuance():
    description = ("the Company issued $500 million aggregate principal "
                   "amount of senior notes due 2033")
    funded = SB._detect_funded_evidence(description)
    assert funded is True
    impact = SB.event_impact(_T.ISSUER_DEBT_ISSUANCE, funded)
    assert impact["affects_debt"] is True
    assert "issuance of debt" in SB.describe_event(_T.ISSUER_DEBT_ISSUANCE, funded)


# ---------------------------------------------------------------------------
# E. facility entered into + "no current plan to draw" -> capacity event
#    only.
# ---------------------------------------------------------------------------

def test_E_facility_with_no_current_plan_to_draw_is_capacity_only():
    description = ("the Company entered into a new credit facility; the "
                   "Company has no current plan to draw on the facility")
    funded = SB._detect_funded_evidence(description)
    assert funded is False
    assert SB.event_impact(_T.ISSUER_DEBT_ISSUANCE, funded)["affects_debt"] is False
    assert "issuance of debt" not in SB.describe_event(_T.ISSUER_DEBT_ISSUANCE, funded)


# ---------------------------------------------------------------------------
# F. research prose cannot call A/B/E "debt issuance" -- checked directly
#    against `describe_event`'s rendered text for each.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("description", [
    "the Company entered into a $20.0 billion delayed-draw term loan "
    "facility; no amounts have been drawn under the facility as of the "
    "date of this filing",
    "the Company entered into an $8.0 billion revolving credit facility; "
    "no loans were outstanding under the facility",
    "the Company entered into a new credit facility; the Company has no "
    "current plan to draw on the facility",
])
def test_F_capacity_only_prose_never_asserts_debt_issuance_or_increase(description):
    funded = SB._detect_funded_evidence(description)
    text = SB.describe_event(_T.ISSUER_DEBT_ISSUANCE, funded)
    assert "issuance of debt" not in text
    assert "increased debt" not in text
    assert "increase debt" not in text


# ---------------------------------------------------------------------------
# End-to-end through `find_post_balance_sheet_events`: a bare item-2.03
# filing (the function's own real-world shape -- it never reads filing
# text) never claims funded debt, since the SEC's own short item-code label
# establishes nothing about whether a dollar moved.
# ---------------------------------------------------------------------------

def test_bare_item_2_03_through_find_post_balance_sheet_events_is_capacity_only():
    events = SB.find_post_balance_sheet_events(
        _submissions([("8-K", "2.03", "2026-08-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    assert len(events) == 1
    event = events[0]
    assert event.event_type == _T.ISSUER_DEBT_ISSUANCE
    assert event.funded is None
    assert event.impact["affects_debt"] is False
    assert "issuance of debt" not in event.description
    # The readiness-limitation sentence still surfaces the event for review
    # (section 24's `requires_reassessment` is independent of funded status).
    assert event.impact["requires_reassessment"] is True


def test_undrawn_facility_never_increases_debt_impact_even_when_dilution_checked():
    """Section 7's capital-structure-effect rule, end to end: an event this
    path cannot confirm as funded must never carry `affects_debt=True`,
    regardless of how confidently the item code identified the instrument
    TYPE.
    """
    events = SB.find_post_balance_sheet_events(
        _submissions([("8-K", "2.03", "2026-08-01")]),
        balance_sheet_date="2026-06-30", valuation_date="2026-08-17")
    event = events[0]
    assert event.impact["potential_dilution"] is False
    assert event.impact["affects_debt"] is False
