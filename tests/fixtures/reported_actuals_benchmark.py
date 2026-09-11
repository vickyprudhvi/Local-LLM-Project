"""The Reported Actuals Source Integration benchmark: cases and ground truth.

WHERE THE GROUND TRUTH COMES FROM

From the DOCUMENT, and it was written when the document was written. Every
expected figure below is the same module-level constant the fixture text is
generated from -- `Q4["revenue"]` is both what the release prints and what a
correct reader must return -- so the expected data cannot drift away from the
text, and it was never read off an extractor's output.

The forbidden figures are the same: a guided revenue, an analyst consensus, a
nine-month total, an adjusted operating income. Each is a real number printed
in the same document, each is a plausible thing for a careless reader to
return, and none of them is the reported GAAP line it would be mistaken for.
That is what makes the precision measurement mean something: the wrong answers
are IN the documents, not absent from them.

WHAT A CASE MEASURES

    discovery      would this filing be recognised as a reported-actual source
    document       which exhibit inside it carries the results
    extraction     which facts come out, with which period, unit and currency
    candidate      what the resolver is finally offered

A case may exercise any subset. `expected_facts` is exhaustive for the primary
period on every case that names it, so precision and recall are both real
numbers rather than one of each.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tests.fixtures import reported_actuals_releases as R


MILLIONS = R.MILLIONS


class Hard:
    """Section 25's counters. Zero, each proved by a control case."""

    GUIDANCE_AS_ACTUAL = "GUIDANCE_AS_ACTUAL"
    ANALYST_ESTIMATE_AS_ACTUAL = "ANALYST_ESTIMATE_AS_ACTUAL"
    WRONG_PERIOD_ACCEPTED = "WRONG_PERIOD_ACCEPTED"
    WRONG_UNIT_ACCEPTED = "WRONG_UNIT_ACCEPTED"
    WRONG_CURRENCY_ACCEPTED = "WRONG_CURRENCY_ACCEPTED"
    YTD_AS_QUARTER_ACCEPTED = "YTD_AS_QUARTER_ACCEPTED"
    UNRELATED_8K_AS_EARNINGS_ACTUAL = "UNRELATED_8K_AS_EARNINGS_ACTUAL"
    SILENT_SAME_PERIOD_VALUE_CONFLICT = "SILENT_SAME_PERIOD_VALUE_CONFLICT"
    UNSUPPORTED_ACCEPTED_FACT = "UNSUPPORTED_ACCEPTED_FACT"

    ALL = (GUIDANCE_AS_ACTUAL, ANALYST_ESTIMATE_AS_ACTUAL,
           WRONG_PERIOD_ACCEPTED, WRONG_UNIT_ACCEPTED, WRONG_CURRENCY_ACCEPTED,
           YTD_AS_QUARTER_ACCEPTED, UNRELATED_8K_AS_EARNINGS_ACTUAL,
           SILENT_SAME_PERIOD_VALUE_CONFLICT, UNSUPPORTED_ACCEPTED_FACT)


class FailureClass:
    """Section 31's vocabulary. The only names a failure gets."""

    SOURCE_DISCOVERY = "SOURCE_DISCOVERY"
    DOCUMENT_SELECTION = "DOCUMENT_SELECTION"
    ACTUAL_SECTION_SELECTION = "ACTUAL_SECTION_SELECTION"
    TABLE_STRUCTURE = "TABLE_STRUCTURE"
    METRIC_IDENTITY = "METRIC_IDENTITY"
    PERIOD_RESOLUTION = "PERIOD_RESOLUTION"
    UNIT_SCALE = "UNIT_SCALE"
    CURRENCY = "CURRENCY"
    ACTUAL_GUIDANCE_SEPARATION = "ACTUAL_GUIDANCE_SEPARATION"
    STATEMENT_COMPLETENESS = "STATEMENT_COMPLETENESS"
    SAME_PERIOD_RECONCILIATION = "SAME_PERIOD_RECONCILIATION"
    TTM_INTEGRATION = "TTM_INTEGRATION"

    ALL = (SOURCE_DISCOVERY, DOCUMENT_SELECTION, ACTUAL_SECTION_SELECTION,
           TABLE_STRUCTURE, METRIC_IDENTITY, PERIOD_RESOLUTION, UNIT_SCALE,
           CURRENCY, ACTUAL_GUIDANCE_SEPARATION, STATEMENT_COMPLETENESS,
           SAME_PERIOD_RECONCILIATION, TTM_INTEGRATION)


def _scaled(values: Dict[str, float]) -> Dict[str, float]:
    return {name: value * MILLIONS for name, value in values.items()}


# The full Q4 statement set, as a correct reader must return it: every figure
# the release prints for the quarter ended 31 July 2026, in dollars.
Q4_FACTS = _scaled(R.Q4)

# The same release's FULL-YEAR figures. Real, printed, and NOT the quarter's.
FY_FACTS = _scaled(R.FY)


@dataclass(frozen=True)
class SourceCase:
    """One document, and what a correct read of it produces."""

    case_id: str
    case_class: str
    description: str
    document: str
    form: str = "8-K"
    filed: str = "2026-09-02"
    accession: str = "0000000000-26-000001"
    document_id: str = "exhibit991.htm"

    # -- discovery ---------------------------------------------------------
    filing_index_html: Optional[str] = None
    expected_document_id: Optional[str] = None
    submissions_row: Optional[dict] = None
    expected_is_source: Optional[bool] = None

    # -- extraction --------------------------------------------------------
    expected_period_end: Optional[str] = None
    expected_period_type: Optional[str] = None
    expected_completeness: Optional[str] = None
    expected_currency: Optional[str] = None
    expected_scale: Optional[str] = None
    # field -> value, EXHAUSTIVE for the expected period.
    expected_facts: Dict[str, float] = field(default_factory=dict)
    # values that appear in the document and must never be accepted as a
    # reported GAAP fact for the expected period.
    forbidden_values: Tuple[Tuple[str, float, str], ...] = ()
    expected_candidate_count: Optional[int] = None
    expects_no_candidate: bool = False

    positive_controls: Tuple[str, ...] = ()
    notes: str = ""


ALL_CASES: Tuple[SourceCase, ...] = (

    # -- A -----------------------------------------------------------------
    SourceCase(
        case_id="A-complete-q4-release", case_class="A",
        description="Complete Q4/FY earnings release, filed before the 10-K.",
        document=R.COMPLETE_Q4_RELEASE,
        filing_index_html=R.EARNINGS_INDEX,
        expected_document_id="exhibit991.htm",
        submissions_row=R.filing("8-K", "0000000000-26-000001", "2026-09-02",
                                 items="2.02,9.01", report_date="2026-09-02",
                                 description="PRESS RELEASE",
                                 document="exhibit991.htm"),
        expected_is_source=True,
        expected_period_end=R.Q4_END_ISO, expected_period_type="QUARTER",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="millions",
        expected_facts=Q4_FACTS,
        forbidden_values=(
            ("revenue", FY_FACTS["revenue"], "the full year, not the quarter"),
            ("operating_income", R.ADJUSTED_OPERATING_INCOME,
             "the adjusted figure, not the GAAP line"),
            ("net_income", R.ADJUSTED_NET_INCOME, "adjusted net income"),
        ),
        notes="The case the whole phase exists for: the resolver could never "
              "see this document."),

    # -- B -----------------------------------------------------------------
    SourceCase(
        case_id="B-headline-only-release", case_class="B",
        description="An income statement and nothing else.",
        document=R.HEADLINE_ONLY_RELEASE,
        filing_index_html=R.EARNINGS_INDEX,
        expected_document_id="exhibit991.htm",
        expected_period_end=R.Q4_END_ISO, expected_period_type="QUARTER",
        expected_completeness="HEADLINE_ONLY", expected_currency="USD",
        expected_scale="millions",
        expected_facts={"revenue": Q4_FACTS["revenue"],
                        "net_income": Q4_FACTS["net_income"]},
        notes="A candidate exists and cannot carry the state. Completeness is "
              "the existing gate; nothing new decides it."),

    SourceCase(
        case_id="B2-prose-only-release", case_class="B",
        description="Two numbers in a sentence and no statement at all.",
        document=R.PROSE_ONLY_RELEASE,
        expects_no_candidate=True,
        notes="Prose is not a financial statement and this layer reads none. "
              "Nothing is the correct output."),

    # -- C -----------------------------------------------------------------
    SourceCase(
        case_id="C-periodic-filing-agrees", case_class="C",
        description="The 10-K for the same period, stating the same figures.",
        document=R.PERIODIC_FILING_MATCHING, form="10-K", filed="2026-10-15",
        accession="0000000000-26-000009",
        expected_period_end=R.Q4_END_ISO, expected_period_type="QUARTER",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="millions",
        expected_facts=Q4_FACTS,
        notes="One economic period, two sources, no disagreement."),

    # -- D -----------------------------------------------------------------
    SourceCase(
        case_id="D-periodic-filing-conflicts", case_class="D",
        description="The 10-K restates the quarter's revenue by 4.5%.",
        document=R.PERIODIC_FILING_CONFLICTING, form="10-K", filed="2026-10-15",
        accession="0000000000-26-000009",
        expected_period_end=R.Q4_END_ISO, expected_period_type="QUARTER",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="millions",
        expected_facts={**Q4_FACTS, "revenue": R.CONFLICTING_REVENUE},
        positive_controls=(Hard.SILENT_SAME_PERIOD_VALUE_CONFLICT,),
        notes="Far outside a rounding move. The conflict must be RECORDED, and "
              "neither this layer nor the release decides which figure wins."),

    # -- E -----------------------------------------------------------------
    SourceCase(
        case_id="E-actuals-and-outlook", case_class="E",
        description="Reported results and a forward outlook in one document.",
        document=R.RELEASE_WITH_OUTLOOK,
        expected_period_end=R.Q4_END_ISO, expected_period_type="QUARTER",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="millions",
        expected_facts=Q4_FACTS,
        forbidden_values=(
            ("revenue", R.GUIDED_Q1_REVENUE, "next quarter's guidance"),
            ("revenue", R.GUIDED_FY_REVENUE, "next year's guidance"),
        ),
        positive_controls=(Hard.GUIDANCE_AS_ACTUAL,),
        notes="The outlook table has the same shape and the same row names as "
              "the income statement. Only its caption differs, which is why "
              "the caption is what decides."),

    # -- F -----------------------------------------------------------------
    SourceCase(
        case_id="F-unrelated-8k", case_class="F",
        description="An 8-K about a leadership change, with an EX-99.1.",
        document=R.LEADERSHIP_RELEASE,
        filing_index_html=R.LEADERSHIP_INDEX,
        submissions_row=R.filing("8-K", "0000000000-26-000004", "2026-08-20",
                                 items="5.02", report_date="2026-08-20",
                                 description="PRESS RELEASE - LEADERSHIP APPOINTMENT",
                                 document="exhibit991.htm"),
        expected_is_source=False, expected_document_id=None,
        expects_no_candidate=True,
        positive_controls=(Hard.UNRELATED_8K_AS_EARNINGS_ACTUAL,),
        notes="It is an 8-K, it has an EX-99.1, it is a press release, and it "
              "names a dollar figure. A form allowlist takes it."),

    # -- G -----------------------------------------------------------------
    SourceCase(
        case_id="G-analyst-estimates-beside-results", case_class="G",
        description="A consensus table printed next to the reported results.",
        document=R.RELEASE_WITH_ANALYST_TABLE,
        expected_period_end=R.Q4_END_ISO, expected_period_type="QUARTER",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="millions",
        expected_facts=Q4_FACTS,
        forbidden_values=(
            ("revenue", R.ANALYST_REVENUE, "a third party's estimate"),
        ),
        positive_controls=(Hard.ANALYST_ESTIMATE_AS_ACTUAL,),
        notes="Same period, same row names, same units. Not the company's "
              "report of what happened."),

    # -- H -----------------------------------------------------------------
    SourceCase(
        case_id="H-foreign-private-issuer", case_class="H",
        description="IFRS half-year results filed on a 6-K.",
        document=R.FOREIGN_ISSUER_RELEASE, form="6-K", filed="2026-08-05",
        accession="0000000000-26-000007",
        submissions_row=R.filing("6-K", "0000000000-26-000007", "2026-08-05",
                                 report_date="2026-06-30",
                                 description="INTERIM RESULTS FOR THE HALF YEAR",
                                 document="results6k.htm"),
        expected_is_source=True,
        expected_period_end=R.FOREIGN_HALF_YEAR_END, expected_period_type="YTD_6M",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="millions",
        expected_facts={"revenue": R.FOREIGN_REVENUE,
                        "operating_income": 486.0 * MILLIONS,
                        "net_income": 372.0 * MILLIONS,
                        "cash_and_cash_equivalents": 1_180.0 * MILLIONS,
                        "assets": 11_400.0 * MILLIONS,
                        "stockholders_equity": 5_120.0 * MILLIONS,
                        "operating_cash_flow": 705.0 * MILLIONS},
        notes="A 6-K carries no item numbers, so the description is the only "
              "metadata evidence. The half year is this issuer's reported "
              "period and the candidate says so."),

    SourceCase(
        case_id="H2-euro-reporting-issuer", case_class="H",
        description="A release whose statements are in euros.",
        document=R.EURO_RELEASE, form="6-K", filed="2027-02-10",
        accession="0000000000-27-000002",
        expected_period_end="2026-12-31", expected_period_type="ANNUAL",
        expected_currency="EUR", expected_scale="millions",
        expected_completeness="COMPLETE",
        expected_facts={"revenue": R.EURO_REVENUE,
                        "operating_income": 540.0 * MILLIONS,
                        "cash_and_cash_equivalents": 980.0 * MILLIONS,
                        "assets": 14_200.0 * MILLIONS,
                        "stockholders_equity": 6_400.0 * MILLIONS,
                        "operating_cash_flow": 812.0 * MILLIONS},
        positive_controls=(Hard.WRONG_CURRENCY_ACCEPTED,),
        notes="Every figure is a euro figure. Recording one as a dollar is a "
              "silent error of roughly fifteen percent that no downstream "
              "check would ever catch."),

    # -- I -----------------------------------------------------------------
    SourceCase(
        case_id="I-thousands-scale", case_class="I",
        description="The same statements printed in thousands.",
        document=R.THOUSANDS_RELEASE,
        expected_period_end=R.Q4_END_ISO, expected_period_type="QUARTER",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="thousands",
        expected_facts={"revenue": Q4_FACTS["revenue"],
                        "operating_income": Q4_FACTS["operating_income"],
                        "net_income": 236_000 * 1_000.0,
                        "cash_and_cash_equivalents":
                            Q4_FACTS["cash_and_cash_equivalents"],
                        "assets": Q4_FACTS["assets"],
                        "stockholders_equity": Q4_FACTS["stockholders_equity"],
                        "operating_cash_flow": Q4_FACTS["operating_cash_flow"]},
        forbidden_values=(
            ("revenue", 1_450_000.0, "the printed digits, unscaled"),
            ("revenue", 1_450_000.0 * MILLIONS, "read as millions"),
        ),
        positive_controls=(Hard.WRONG_UNIT_ACCEPTED,),
        notes="Identical digits to case A's balance sheet in one place and a "
              "thousand-fold different meaning."),

    SourceCase(
        case_id="I2-no-scale-stated", case_class="I",
        description="A statement table that names no scale anywhere.",
        document=R.NO_SCALE_RELEASE,
        expects_no_candidate=True,
        positive_controls=(Hard.WRONG_UNIT_ACCEPTED,),
        notes="'1,450.0' is a plausible figure in every scale there is. "
              "Refusing it is the only honest answer."),

    # -- J -----------------------------------------------------------------
    SourceCase(
        case_id="J-quarter-and-ytd-columns", case_class="J",
        description="A quarter column and a nine-month column, one period end.",
        document=R.QUARTER_AND_YTD_RELEASE,
        expected_period_end=R.Q3_END_ISO, expected_period_type="QUARTER",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="millions",
        expected_facts={"revenue": R.Q3_QUARTER_REVENUE,
                        "operating_income": 268.0 * MILLIONS,
                        "net_income": 209.0 * MILLIONS,
                        "cash_and_cash_equivalents": 740.0 * MILLIONS,
                        "assets": 5_980.0 * MILLIONS,
                        "stockholders_equity": 2_760.0 * MILLIONS,
                        "operating_cash_flow": 331.0 * MILLIONS},
        forbidden_values=(
            ("revenue", R.Q3_YTD_REVENUE, "nine months read as three"),
            ("operating_cash_flow", 988.0 * MILLIONS, "nine months of cash flow"),
        ),
        expected_candidate_count=1,
        positive_controls=(Hard.YTD_AS_QUARTER_ACCEPTED,
                           Hard.WRONG_PERIOD_ACCEPTED),
        notes="Both columns say 'April 30, 2026'. Only the band above them "
              "says which is three months and which is nine."),

    # -- K -----------------------------------------------------------------
    SourceCase(
        case_id="K-instant-and-flow-in-one-release", case_class="K",
        description="A balance sheet and a cash-flow statement together.",
        document=R.COMPLETE_Q4_RELEASE,
        expected_period_end=R.Q4_END_ISO, expected_period_type="QUARTER",
        expected_completeness="COMPLETE", expected_currency="USD",
        expected_scale="millions",
        expected_facts=Q4_FACTS,
        positive_controls=(Hard.WRONG_PERIOD_ACCEPTED,),
        notes="The balance sheet's columns carry no duration band and its "
              "figures are instants; the cash-flow statement's carry one and "
              "its figures are flows. Same dates, different kind of fact."),
)


def cases() -> List[SourceCase]:
    return list(ALL_CASES)


def classes_covered() -> List[str]:
    return sorted({case.case_class for case in ALL_CASES})


def controls_covered() -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {name: [] for name in Hard.ALL}
    for case in ALL_CASES:
        for name in case.positive_controls:
            found[name].append(case.case_id)
    # Every accepted fact on every case is checked against the document text,
    # so this requirement is controlled by the whole set rather than by one.
    found[Hard.UNSUPPORTED_ACCEPTED_FACT] = [c.case_id for c in ALL_CASES
                                             if c.expected_facts]
    return found


# ---------------------------------------------------------------------------
# The discovery-only cases (section 24's precision and recall)
# ---------------------------------------------------------------------------
#
# One submissions payload holding the mix a real issuer files: two earnings
# 8-Ks, a 6-K carrying interim results, and four filings that are not reported
# results however much they look like one.

DISCOVERY_SUBMISSIONS = R.submissions(
    R.filing("8-K", "acc-earnings-1", "2026-09-02", items="2.02,9.01",
             report_date="2026-09-02", description="PRESS RELEASE",
             document="exhibit991.htm"),
    R.filing("8-K", "acc-leadership", "2026-08-20", items="5.02",
             description="PRESS RELEASE - LEADERSHIP APPOINTMENT",
             document="exhibit991.htm"),
    R.filing("8-K", "acc-debt", "2026-08-11", items="1.01,2.03",
             description="INDENTURE", document="ex41.htm"),
    R.filing("8-K", "acc-acquisition", "2026-07-14", items="8.01",
             description="PRESS RELEASE - ACQUISITION ANNOUNCEMENT",
             document="ex991.htm"),
    R.filing("6-K", "acc-interim", "2026-08-05", report_date="2026-06-30",
             description="INTERIM RESULTS FOR THE HALF YEAR ENDED 30 JUNE 2026",
             document="results6k.htm"),
    R.filing("6-K", "acc-agm", "2026-06-02",
             description="NOTICE OF ANNUAL GENERAL MEETING",
             document="agm.htm"),
    R.filing("8-K", "acc-earnings-2", "2026-06-03", items="2.02,9.01",
             report_date="2026-06-03", description="PRESS RELEASE",
             document="exhibit991.htm"),
    R.filing("10-Q", "acc-10q", "2026-06-05", report_date="2026-04-30",
             description="FORM 10-Q", document="form10q.htm"),
)

# Which of the above are genuinely earnings-result sources. Decided from what
# each filing IS, not from what the classifier says about it.
DISCOVERY_TRUTH = {
    "acc-earnings-1": True,
    "acc-leadership": False,
    "acc-debt": False,
    "acc-acquisition": False,
    "acc-interim": True,
    "acc-agm": False,
    "acc-earnings-2": True,
    # The 10-Q is a reported-actual source, and the CompanyFacts path already
    # reads it. `carriers_only` keeps it out of this layer's scope, so it is
    # excluded from the discovery denominator rather than counted as a miss.
    "acc-10q": None,
}

# The document-selection cases: an index where the conventional slot holds the
# wrong document, and one with no exhibit at all.
DOCUMENT_SELECTION_CASES = (
    ("conventional", R.EARNINGS_INDEX, "exhibit991.htm"),
    ("inverted", R.INVERTED_INDEX, "earnings992.htm"),
    ("leadership", R.LEADERSHIP_INDEX, None),
    ("no-exhibit", R.NO_EXHIBIT_INDEX, None),
)
