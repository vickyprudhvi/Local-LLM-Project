"""The Finance Extraction benchmark: cases, expected data, and provenance.

WHAT THE EXPECTED DATA IS, AND WHAT IT IS NOT

Two kinds of case, labelled, because the difference matters when reading a
score:

  REAL      the release text captured in this repo's own ticker fixtures.
            The expected statements were read out of that text by hand for
            this benchmark. They are ground truth about what the document
            says; they are not an issuer's own summary and not a vendor feed.

  SYNTHETIC text written for this benchmark, so the expected data is exact by
            construction. Used for classes the captured fixtures do not cover
            and for defect classes a real release rarely contains cleanly.

`expected_guidance` is deliberately PARTIAL on several real cases: it names
the statements verified by reading, not every statement in the document. A
partial expected set can prove PRECISION (nothing invented) and can prove
recall of what it names; it cannot prove total recall, and the scorer reports
those separately rather than blending them into one number.

The ticker names here are benchmark DATA. They appear in no production path,
which `test_finance_extraction_benchmark.py` asserts.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_FIXTURES = os.path.dirname(os.path.abspath(__file__))


class CaseKind:
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"


@dataclass(frozen=True)
class ExpectedStatement:
    """One statement a correct extractor must produce, or must not."""

    metric: str
    target_period: str
    unit: str
    low: Optional[float] = None
    high: Optional[float] = None
    basis: Optional[str] = None
    # The revision the release performs on this statement. Set only where the
    # document states it plainly enough that a reader has no excuse -- an
    # unset value means "not part of this case's ground truth", not
    # "INITIATE", so that action accuracy is scored on statements whose action
    # was actually verified rather than on a default nobody checked.
    action: Optional[str] = None
    note: str = ""

    @property
    def key(self) -> Tuple[str, str]:
        return (self.metric, self.target_period)


@dataclass(frozen=True)
class BenchmarkCase:
    """One document, its class, and what a correct read produces."""

    case_id: str
    kind: str
    classes: Tuple[str, ...]
    fixture: Optional[str] = None          # ticker fixture file, REAL cases
    document_key: Optional[str] = None     # exhibit inside that fixture
    text: Optional[str] = None             # SYNTHETIC cases carry their own
    filed: str = "2026-01-01"
    fiscal_year_hint: Optional[int] = None
    expected_guidance: Tuple[ExpectedStatement, ...] = ()
    forbidden_guidance: Tuple[Tuple[str, str], ...] = ()
    expected_complete: bool = False        # is expected_guidance exhaustive?
    expected_latest_period: Optional[str] = None
    notes: str = ""

    def document_text(self) -> str:
        if self.text is not None:
            return self.text
        from finance import guidance as gm

        path = os.path.join(_FIXTURES, self.fixture)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        raw = (payload.get("sec") or {}).get("filing_documents") or {}
        return gm.html_to_text(raw[self.document_key])


# ---------------------------------------------------------------------------
# REAL cases — expected statements read from the captured release text
# ---------------------------------------------------------------------------

_REAL: List[BenchmarkCase] = [
    # -- semiconductor / high growth / multi-horizon -----------------------
    #
    # "Revenue is expected to be $91.0 billion, plus or minus 2%. ... GAAP and
    # non-GAAP gross margins are expected to be 74.9% and 75.0% ... GAAP and
    # non-GAAP operating expenses are expected to be approximately $8.5
    # billion and $8.3 billion ... For the full year fiscal 2027 ... tax
    # rates to be between 16.0% and 18.0%"
    #
    # Two horizons in one release, which is why this case is here.
    BenchmarkCase(
        case_id="semiconductor-q1fy27", kind=CaseKind.REAL,
        classes=("semiconductor", "high_growth", "multi_horizon", "guidance_heavy"),
        fixture="nvda_regression.json",
        document_key=None, filed="2026-05-27", fiscal_year_hint=2027,
        expected_guidance=(
            ExpectedStatement("revenue", "Q2 FY2027", "currency", 89.18, 92.82,
                              note="$91.0 billion plus or minus 2%"),
            ExpectedStatement("adjusted_gross_margin", "Q2 FY2027", "ratio",
                              0.745, 0.755, basis="adjusted",
                              note="non-GAAP 75.0% plus or minus 50 bps"),
            ExpectedStatement("tax_rate", "FY2027", "ratio", 0.16, 0.18,
                              note="full-year fiscal 2027, a SECOND horizon"),
        ),
        notes="Outlook names a quarter and a full year in one block."),

    # -- retail, non-calendar fiscal year, component-vs-consolidated -------
    #
    # "The Company expects inside same-store sales to increase 2% to 5% ...
    # The Company expects EBITDA to increase 8% to 10%"
    #
    # INSIDE SAME-STORE SALES IS NOT CONSOLIDATED REVENUE. This case exists
    # to measure whether an extractor promotes a component to the total --
    # the AT&T failure class, in a different issuer's vocabulary.
    BenchmarkCase(
        case_id="retail-fy27-outlook", kind=CaseKind.REAL,
        classes=("mature_profitable", "retail", "non_calendar_fiscal_year",
                 "component_vs_consolidated"),
        fixture="casy_regression.json", document_key=None,
        filed="2026-06-09", fiscal_year_hint=2027,
        expected_guidance=(
            ExpectedStatement("ebitda_growth", "FY2027", "ratio", 0.08, 0.10),
        ),
        forbidden_guidance=(("revenue_growth", "FY2027"),),
        notes="Same-store sales guidance must not become consolidated revenue "
              "growth."),

    # -- industrial, a full-year outlook table ----------------------------
    BenchmarkCase(
        case_id="industrial-fy26-outlook", kind=CaseKind.REAL,
        classes=("mature_profitable", "industrial", "guidance_table"),
        fixture="aos_regression.json", document_key=None,
        filed="2026-07-30", fiscal_year_hint=2026,
        expected_guidance=(
            ExpectedStatement("revenue_growth", "FY2026", "ratio", 0.02, 0.03),
            ExpectedStatement("earnings_per_share", "FY2026", "currency_per_share",
                              3.60, 3.75),
            ExpectedStatement("adjusted_earnings_per_share", "FY2026",
                              "currency_per_share", 3.70, 3.85, basis="adjusted"),
        ),
        notes="Sales growth 2-3%; GAAP and adjusted EPS stated separately."),

    # -- telecom, multi-year framework beside a full-year outlook ----------
    BenchmarkCase(
        case_id="telecom-reiterated-fy26", kind=CaseKind.REAL,
        classes=("telecom", "guidance_heavy", "multi_year_framework",
                 "reaffirmation"),
        fixture="t_regression.json", document_key=None,
        filed="2026-07-23", fiscal_year_hint=2026,
        expected_guidance=(
            ExpectedStatement("free_cash_flow", "FY2026", "currency",
                              action="REAFFIRMED", note="$18B+"),
        ),
        notes="Reiterates full-year guidance; a multi-year framework runs "
              "alongside it and must not be read as a named-year outlook."),

    # -- healthcare, a dense full-year guidance paragraph ------------------
    BenchmarkCase(
        case_id="healthcare-fy26-guidance", kind=CaseKind.REAL,
        classes=("healthcare", "guidance_heavy"),
        fixture="mrk_regression.json", document_key=None,
        filed="2026-07-29", fiscal_year_hint=2026,
        expected_guidance=(
            ExpectedStatement("revenue", "FY2026", "currency"),
            ExpectedStatement("adjusted_earnings_per_share", "FY2026",
                              "currency_per_share", basis="adjusted"),
            ExpectedStatement("tax_rate", "FY2026", "ratio"),
        ),
        notes="Values intentionally unpinned; this case measures metric "
              "identity and target period, not magnitudes."),

    # -- loss-making, capital-expenditure-led outlook ----------------------
    BenchmarkCase(
        case_id="loss-making-capex", kind=CaseKind.REAL,
        classes=("loss_making", "automotive", "capex_led"),
        fixture="rivn_regression.json", document_key=None,
        filed="2026-08-05", fiscal_year_hint=2026,
        expected_guidance=(),
        notes="Expected set deliberately empty: this case measures FALSE "
              "POSITIVES on a release whose forward statements this taxonomy "
              "has no identity for."),

    # -- foreign private issuer -------------------------------------------
    BenchmarkCase(
        case_id="foreign-private-issuer", kind=CaseKind.REAL,
        classes=("foreign_private_issuer",),
        fixture="jbs_regression.json", document_key=None,
        filed="2026-08-13", fiscal_year_hint=2026,
        expected_guidance=(),
        notes="Same purpose: a release with no extractable named-period "
              "guidance. Anything produced here is a false positive."),
]


def _resolve_document_keys() -> None:
    """Pin each REAL case to the newest exhibit in its fixture.

    Done once at import rather than written into each case, so a fixture
    re-capture does not silently point a case at a document that moved.
    """
    for index, case in enumerate(_REAL):
        if case.document_key is not None or not case.fixture:
            continue
        path = os.path.join(_FIXTURES, case.fixture)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        raw = (payload.get("sec") or {}).get("filing_documents") or {}
        exhibits = sorted(k for k in raw if "index" not in k)
        if not exhibits:
            continue
        _REAL[index] = BenchmarkCase(
            **{**case.__dict__, "document_key": exhibits[-1]})


_resolve_document_keys()


# ---------------------------------------------------------------------------
# SYNTHETIC cases — exact by construction
# ---------------------------------------------------------------------------

def _synthetic() -> List[BenchmarkCase]:
    from tests.fixtures.earnings_release import EARNINGS_RELEASE_HTML
    from tests.fixtures.multi_horizon_release import MULTI_HORIZON_RELEASE
    from finance import guidance as gm

    return [
        BenchmarkCase(
            case_id="synthetic-multi-horizon", kind=CaseKind.SYNTHETIC,
            classes=("software", "multi_horizon", "guidance_heavy"),
            text=MULTI_HORIZON_RELEASE, filed="2027-02-20", fiscal_year_hint=2027,
            expected_guidance=(
                ExpectedStatement("revenue_growth", "Q1 FY2027", "ratio", 0.27, 0.29),
                ExpectedStatement("revenue", "FY2027", "currency", 118.0, 120.0),
                ExpectedStatement("adjusted_earnings_per_share", "Q1 FY2027",
                                  "currency_per_share", 1.60, 1.70, basis="adjusted"),
                ExpectedStatement("adjusted_earnings_per_share", "FY2027",
                                  "currency_per_share", 7.10, 7.40, basis="adjusted"),
            ),
            expected_complete=True,
            notes="Two horizons, and one metric stated for both."),

        BenchmarkCase(
            case_id="synthetic-historical-tables", kind=CaseKind.SYNTHETIC,
            classes=("table_contamination", "earnings_release_before_10k"),
            text=gm.html_to_text(EARNINGS_RELEASE_HTML),
            filed="2026-07-28", fiscal_year_hint=2026,
            expected_guidance=(
                ExpectedStatement("revenue", "Q3 FY2026", "currency", 1260.0, 1300.0),
                ExpectedStatement("adjusted_operating_margin", "Q3 FY2026", "ratio",
                                  0.20, 0.22, basis="adjusted"),
                ExpectedStatement("adjusted_ebitda_margin", "Q3 FY2026", "ratio",
                                  0.28, 0.30, basis="adjusted"),
            ),
            forbidden_guidance=(("share_count", "Q2 FY2026"),
                                ("operating_income", "Q3 FY2026"),
                                ("adjusted_ebitda", "Q3 FY2026")),
            expected_complete=True,
            notes="Three historical tables above three outlook sentences. The "
                  "forbidden entries are the contaminations each table invites."),

        BenchmarkCase(
            case_id="synthetic-margin-not-amount", kind=CaseKind.SYNTHETIC,
            classes=("unit_identity",),
            text=("Financial Outlook\n\nThird quarter fiscal year 2027 non-GAAP "
                  "operating income guidance of approximately 67 percent of "
                  "projected revenue. Third quarter fiscal year 2027 Adjusted "
                  "EBITDA guidance of approximately 68 percent of projected "
                  "revenue.\n"),
            filed="2026-06-03", fiscal_year_hint=2027,
            expected_guidance=(
                ExpectedStatement("adjusted_operating_margin", "Q3 FY2027", "ratio",
                                  0.67, 0.67, basis="adjusted"),
                ExpectedStatement("adjusted_ebitda_margin", "Q3 FY2027", "ratio",
                                  0.68, 0.68, basis="adjusted"),
            ),
            forbidden_guidance=(("operating_income", "Q3 FY2027"),
                                ("adjusted_ebitda", "Q3 FY2027")),
            expected_complete=True,
            notes="A percentage OF REVENUE is a margin. The forbidden entries "
                  "are the same figures read as amounts."),

        BenchmarkCase(
            case_id="synthetic-reaffirmation", kind=CaseKind.SYNTHETIC,
            classes=("economic_dedup", "reaffirmation"),
            text=("Financial Outlook\n\nFor fiscal year 2027, we confirm our prior "
                  "revenue guidance of $90 billion.\n"),
            filed="2026-06-10", fiscal_year_hint=2027,
            expected_guidance=(
                ExpectedStatement("revenue", "FY2027", "currency", 90.0, 90.0,
                                  action="REAFFIRMED"),
            ),
            expected_complete=True,
            notes="A reaffirmation is one statement, not a second signal."),

        BenchmarkCase(
            case_id="synthetic-analyst-estimate", kind=CaseKind.SYNTHETIC,
            classes=("evidence_grounding",),
            text=("Financial Outlook\n\nAnalysts surveyed by a data provider expect "
                  "fiscal year 2027 revenue of $92 billion. The Company does not "
                  "comment on third-party estimates.\n"),
            filed="2026-06-10", fiscal_year_hint=2027,
            expected_guidance=(),
            forbidden_guidance=(("revenue", "FY2027"),),
            expected_complete=True,
            notes="An analyst estimate is not management guidance."),
    ]


def all_cases() -> List[BenchmarkCase]:
    cases = [c for c in _REAL if c.document_key]
    try:
        cases += _synthetic()
    except Exception:                                     # noqa: BLE001
        pass
    return cases


def classes_covered() -> List[str]:
    seen = []
    for case in all_cases():
        for name in case.classes:
            if name not in seen:
                seen.append(name)
    return sorted(seen)
