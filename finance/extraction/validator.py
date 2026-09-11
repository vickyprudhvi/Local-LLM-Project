"""The acceptance boundary. Nothing the model says gets in without passing.

This is where V2's guarantee lives. The model is allowed to be wrong; it is
not allowed to be believed. Every candidate is checked against the source
text it claims to come from and against the same semantic rules the rest of
the system uses, and a candidate that fails any check is REFUSED rather than
repaired -- a repaired candidate is a value this layer invented.

The checks, in the order a reader would want them explained:

     1 evidence grounding        the sentence is in the document
     2 value grounding           the number is in the sentence
     3 metric identity           a name the taxonomy knows
     4 unit compatibility        the unit matches the metric's kind
     5 denominator compatibility a ratio names what it is a ratio of
     6 basis                     GAAP / non-GAAP, as stated
     7 prospective semantics     a forward statement, not a results table
     8 target period             present and parseable
     9 target-period plausibility a release cannot guide a past period
    10 actualization             a period whose results are in is not a forecast
    11 revision action           a reaffirmation is not a second forecast
    12 economic identity         one target, one canonical signal
    13 supersession              the newest statement of a thing wins
    14 forecast-horizon          what it may do to an assumption
    15 confidence                below the floor is refused

None of this is a second semantic engine. Metric identity, units, periods and
horizon eligibility are `finance.guidance` and `finance.semantics`; this
module sequences them and grounds them against the text.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import tools.config as config
from finance import guidance as gm
from finance import semantics as sem
from finance.extraction import metric_grounding
from finance.extraction import tolerance as tolerance_module
from finance.extraction.schema import (
    GuidanceAction,
    GuidanceCandidate,
    REPORTED_RESULTS_MARKER,
    RejectionCode,
    ReportedActualCandidate,
    ValueType,
    domain_period_frequency,
    domain_unit,
    is_percent_unit,
)


# ---------------------------------------------------------------------------
# 1-2. grounding
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Whitespace-insensitive comparison text.

    A model copying a sentence out of an HTML-derived document routinely
    changes runs of whitespace. That is not a fabrication and refusing it
    would reject correct candidates; changing a WORD is a fabrication and is
    still caught.
    """
    return " ".join((text or "").split()).lower()


def evidence_is_grounded(sentence: str, document_text: str) -> bool:
    return bool(sentence.strip()) and _normalise(sentence) in _normalise(document_text)


# `(?:,\d{3})*` with a STAR let the first alternative match "202" of
# "2027" and leave "7" behind, so a four-digit number without a comma was
# never seen whole and any candidate asserting one was refused as
# ungrounded. The group is required, so a plain digit run falls to the
# second alternative and is read as one number.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _numbers_in(text: str) -> List[float]:
    out = []
    for token in _NUMBER.findall(text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


_TOLERANCE_PHRASE = re.compile(
    r"(?i)plus\s*(?:or|/)?\s*minus|\+\s*/\s*-|±|\bwithin\s+(?:a\s+)?range\s+of")


def _matches_any(value: float, present: List[float]) -> bool:
    forms = {value, value * 100.0, value / 100.0}
    return any(any(abs(f - p) <= max(abs(p) * 1e-6, 1e-9) for p in present)
               for f in forms)


def _operands_are_grounded(candidate: GuidanceCandidate,
                           present: List[float]) -> Tuple[bool, str]:
    """For a point-with-tolerance: is every OPERAND in the sentence?

    Three things must be there -- the point, the tolerance, and the phrase
    that says it is a tolerance at all. The last one matters: without it a
    sentence containing any two numbers could be reshaped into a point and a
    width, which is a way of inventing a range out of unrelated figures.
    """
    if candidate.value is None:
        return False, "a tolerance candidate reports no point value"
    if candidate.tolerance_value is None:
        return False, "a tolerance candidate reports no tolerance value"
    if not _matches_any(candidate.value, present):
        return False, (f"the point value {candidate.value} does not appear in "
                       "the cited sentence")
    if not _matches_any(candidate.tolerance_value, present):
        return False, (f"the tolerance {candidate.tolerance_value} does not "
                       "appear in the cited sentence")
    if not _TOLERANCE_PHRASE.search(candidate.source_sentence or ""):
        return False, ("the cited sentence states no tolerance; two numbers "
                       "alone are not a point and a width")
    return True, ""


def value_is_grounded(candidate: GuidanceCandidate) -> Tuple[bool, str]:
    """Is every number the candidate asserts present in its own sentence?

    The check that stops the failure §7 names: a model reporting
    `ADJUSTED_EBITDA = $68B` from a sentence that says "68 percent of
    projected revenue" has the number and the wrong quantity, so this alone
    does not catch it -- the unit and denominator checks below do. What this
    catches is the number that is not there at all.
    """
    present = _numbers_in(candidate.source_sentence)
    if not present:
        return False, "the cited sentence contains no number"

    if candidate.value_type == ValueType.TOLERANCE:
        # A tolerance's ENDPOINTS are arithmetic, not text: "$91.0 billion
        # plus or minus 2%" contains 91.0 and 2, and contains neither 89.18
        # nor 92.82. So the OPERANDS are what must be grounded. This is not a
        # relaxation -- it is checking the numbers the sentence actually
        # asserts. A candidate that instead reports 89.18 as `low` is a RANGE
        # claim and is still held to literal grounding below.
        return _operands_are_grounded(candidate, present)

    low, high = candidate.bounds()
    for asserted in (v for v in (low, high) if v is not None):
        # A percentage may be written as 68 and carried as 0.68; both count.
        forms = {asserted, asserted * 100.0, asserted / 100.0}
        if not any(any(abs(f - p) <= max(abs(p) * 1e-6, 1e-9) for p in present)
                   for f in forms):
            return False, (f"{asserted} does not appear in the cited sentence")
    return True, ""


# ---------------------------------------------------------------------------
# 3. the KIND of quantity the sentence supports
# ---------------------------------------------------------------------------
#
# Grounding a number is not enough. Section 7's worked example is a model
# reporting
#
#     ADJUSTED_EBITDA = $68 billion
#
# from a sentence that says "Adjusted EBITDA guidance of approximately 68
# percent of projected revenue". The number 68 IS in the sentence. Every
# check that asks only "is this number present" passes it, and what reaches
# the canonical layer is a quarterly profit figure twice the size of the
# company's revenue.
#
# So the unit is grounded too: find where the asserted number sits in its own
# sentence and read what the text says it is RIGHT THERE. A percent sign or
# the word percent beside the number means the figure is a ratio, whatever
# the candidate called it; a currency sign or a scale word means an amount.
# A candidate contradicted by its own evidence is refused.

_LOCAL_REACH = 24

_LOCAL_PERCENT = re.compile(r"(?i)^\s*(?:%|percent(?:age)?\b)")
_LOCAL_SCALE = re.compile(r"(?i)^\s*(?:billion|million|bn|mm)\b")


class _Quantity:
    PERCENT = "PERCENT"
    CURRENCY = "CURRENCY"
    UNKNOWN = "UNKNOWN"


def _local_quantity(sentence: str, position: int, token: str) -> str:
    """What the text says the number at `position` IS."""
    after = sentence[position + len(token):position + len(token) + _LOCAL_REACH]
    before = sentence[max(0, position - 2):position]
    if _LOCAL_PERCENT.match(after):
        return _Quantity.PERCENT
    if "$" in before or _LOCAL_SCALE.match(after):
        return _Quantity.CURRENCY
    return _Quantity.UNKNOWN


def _claimed_quantity(candidate: GuidanceCandidate) -> str:
    if candidate.value_type in ValueType.RATIO_SHAPES or is_percent_unit(candidate.unit):
        return _Quantity.PERCENT
    domain, _scale = domain_unit(candidate.unit)
    if domain in (gm.GuidanceUnit.CURRENCY, gm.GuidanceUnit.CURRENCY_PER_SHARE):
        return _Quantity.CURRENCY
    return _Quantity.UNKNOWN


def unit_matches_evidence(candidate: GuidanceCandidate) -> Tuple[bool, str]:
    """Does the cited sentence support the KIND of quantity claimed?"""
    claimed = _claimed_quantity(candidate)
    if claimed == _Quantity.UNKNOWN:
        return True, ""
    sentence = candidate.source_sentence or ""
    low, high = candidate.bounds()
    for asserted in (v for v in (low, high) if v is not None):
        for token in _tokens_for(asserted):
            for match in re.finditer(re.escape(token), sentence):
                stated = _local_quantity(sentence, match.start(), token)
                if stated == _Quantity.UNKNOWN or stated == claimed:
                    continue
                return False, (
                    f"the cited sentence states {token} as a "
                    f"{stated.lower()} figure, and the candidate claims a "
                    f"{claimed.lower()} one")
    return True, ""


def _tokens_for(value: float) -> List[str]:
    """How this number could be written in the text.

    A percentage carried as 0.68 is written 68; an amount carried as 90.0 is
    written 90 or 90.0. Both spellings are tried so the search finds the
    number wherever the writer put the decimal point.
    """
    forms = {value, value * 100.0}
    tokens = []
    for form in forms:
        if abs(form - round(form)) < 1e-9:
            tokens.append(str(int(round(form))))
        tokens.append(("%f" % form).rstrip("0").rstrip("."))
    return [t for t in dict.fromkeys(tokens) if t]


# ---------------------------------------------------------------------------
# 4-5. units and denominators
# ---------------------------------------------------------------------------

_OF_REVENUE = re.compile(
    r"(?i)\b(?:as\s+a\s+percent(?:age)?\s+of|of)\s+"
    r"(?:its\s+|our\s+|the\s+|a\s+)?"
    r"(?:projected\s+|expected\s+|anticipated\s+|estimated\s+|forecast(?:ed)?\s+|"
    r"guided\s+|full[\s-]?year\s+|quarterly\s+|total\s+|consolidated\s+|net\s+|"
    r"worldwide\s+)*(?:revenues?|sales)\b")

_PERCENT_IN_TEXT = re.compile(r"(?i)\bpercent(?:age)?\b|%")

# Metrics whose taxonomy unit is an absolute amount. A ratio proposed for one
# of these must resolve to that metric's ratio identity or be refused --
# `finance.guidance.resolve_percentage_identity` owns that rule and this
# defers to it rather than restating it.
_ABSOLUTE_UNITS = (gm.GuidanceUnit.CURRENCY, gm.GuidanceUnit.CURRENCY_PER_SHARE,
                   gm.GuidanceUnit.SHARES)


def resolve_identity(candidate: GuidanceCandidate) -> Tuple[Optional[str], Optional[str], str]:
    """(metric_id, domain unit, reason) after unit/denominator reconciliation.

    The single place where "what the model called it" becomes "what the
    taxonomy calls it". A percentage under an absolute metric is re-identified
    through the domain's own resolver, and where no identity exists it is
    refused -- never stored under the absolute name.
    """
    name = candidate.metric_id
    if not name or name not in gm._METRIC_BY_NAME:      # noqa: SLF001
        return None, None, f"{name!r} is not in the reviewed metric taxonomy"

    taxonomy_unit = gm._METRIC_BY_NAME[name][1]         # noqa: SLF001
    model_unit, _scale = domain_unit(candidate.unit)
    ratio_shape = candidate.value_type in ValueType.RATIO_SHAPES
    percent_unit = is_percent_unit(candidate.unit)

    if not (ratio_shape or percent_unit):
        if model_unit is not None and taxonomy_unit != model_unit:
            return None, None, (
                f"unit {candidate.unit!r} does not match the {name!r} taxonomy entry "
                f"({taxonomy_unit!r})")
        return name, taxonomy_unit, ""

    # A ratio. If the metric is already a ratio metric, keep it.
    if taxonomy_unit == gm.GuidanceUnit.RATIO:
        return name, gm.GuidanceUnit.RATIO, ""

    # An absolute metric carrying a ratio. The denominator decides what it is.
    if taxonomy_unit in _ABSOLUTE_UNITS:
        basis = _domain_basis(candidate.basis)
        context = " ".join(filter(None, (candidate.source_sentence,
                                         candidate.denominator_metric or "")))
        resolved = gm.resolve_percentage_identity(name, taxonomy_unit, basis, context)
        if resolved is None:
            return None, None, (
                f"{name!r} is measured as an absolute amount and the candidate is a "
                "ratio, and no denominator for it could be established from the cited "
                "sentence")
        return resolved[0], resolved[1], ""

    return name, taxonomy_unit, ""


def _denominator_is_in_the_metric_identity(name: Optional[str]) -> bool:
    """Does this metric's own definition fix what its percentage is of?

    The taxonomy already answers this: a metric whose canonical unit is RATIO
    -- `gross_margin`, `operating_margin`, `tax_rate` -- carries its
    denominator in its identity. There is no competing reading of "gross
    margin of 75%" for a denominator clause to settle.

    `resolve_identity` has always drawn this line (a ratio candidate under a
    RATIO metric is kept; under an ABSOLUTE metric it must resolve through
    `resolve_percentage_identity` or be refused). This consults the same
    source so the two cannot disagree.
    """
    if not name:
        return False
    entry = gm._METRIC_BY_NAME.get(name)                # noqa: SLF001
    return bool(entry) and entry[1] == gm.GuidanceUnit.RATIO


def denominator_is_grounded(candidate: GuidanceCandidate) -> Tuple[bool, str]:
    """A ratio must name a denominator the SENTENCE supports.

    §7's worked example. A candidate claiming a margin whose sentence never
    says what the percentage is of has not been read out of the document; it
    has been guessed.

    WITH ONE EXCEPTION, and it is the rule's own scope rather than a hole in
    it. The spec's "neither stated -> REFUSED" governs an ABSOLUTE metric
    carrying a percentage, where 21% under `operating_income` could be a
    margin or a growth rate. A metric that is canonically a RATIO has no such
    ambiguity -- its name is the denominator statement -- and demanding that
    the sentence restate it contradicts the very rule that denominator is part
    of the IDENTITY.

    Found live: "GAAP and non-GAAP gross margins are expected to be 74.9% and
    75.0%" had both correctly-read, correctly-grounded margins refused.
    """
    if candidate.value_type not in ValueType.RATIO_SHAPES \
            and not is_percent_unit(candidate.unit):
        return True, ""
    sentence = candidate.source_sentence or ""
    if candidate.value_type == ValueType.MARGIN or candidate.denominator_metric:
        if not (_OF_REVENUE.search(sentence)
                or _denominator_is_in_the_metric_identity(candidate.metric_id)):
            return False, ("the candidate is a margin but the cited sentence does not "
                           "state what the percentage is a percentage of")
    # Unchanged, and still load-bearing: a ratio candidate drawn from a
    # sentence stating no percentage at all was not read out of that sentence.
    # This is what refuses a leverage target stated as a multiple.
    if not _PERCENT_IN_TEXT.search(sentence):
        return False, "the candidate is a ratio but the cited sentence states no percentage"
    return True, ""


# ---------------------------------------------------------------------------
# 6-9. basis, prospective semantics, period
# ---------------------------------------------------------------------------

_BASIS_MAP = {
    "GAAP": gm.BASIS_GAAP,
    "NON_GAAP": gm.BASIS_ADJUSTED,
    "ADJUSTED": gm.BASIS_ADJUSTED,
    "COMPANY_DEFINED": gm.BASIS_COMPANY_DEFINED,
    "UNKNOWN": gm.BASIS_GAAP,
}


def _domain_basis(model_basis: Optional[str]) -> str:
    return _BASIS_MAP.get((model_basis or "UNKNOWN").upper(), gm.BASIS_GAAP)


# The same pattern the reader uses to stop absorbing a table run. Defined in
# `schema` so the two layers cannot drift apart about what "reported results"
# looks like -- two definitions of one idea is exactly how the denominator
# defect happened.
_HISTORICAL_TABLE = REPORTED_RESULTS_MARKER

_ANALYST_ESTIMATE = re.compile(
    r"(?i)\banalysts?\b|\bconsensus\b|\bstreet\s+estimate|\bsurvey\s+of\s+analysts\b")

_PROSPECTIVE = re.compile(
    r"(?i)\b(?:expect\w*|guidance|outlook|anticipat\w*|forecast\w*|project\w*|"
    r"target\w*|reaffirm\w*|reiterat\w*|confirm\w*|plans?\s+to|will\s+be|"
    r"withdraw\w*|suspend\w*)\b")


# How far back a row may look for the caption that governs it. A caption
# further away than this is not introducing the row.
_BLOCK_REACH_CHARS = 1500
_BLOCK_FRAGMENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A flattened table row is short; a paragraph is not. The same boundary rule
# `semantic_extractor.select_sections` uses to decide what to SEND is used
# here to decide what a caption GOVERNS, so the two layers agree about where
# a table starts and stops.
_BLOCK_ROW_MAX_WORDS = 12


def governing_caption(sentence: str, document_text: str) -> Optional[str]:
    """The caption a table row sits under, or None.

    Walks backwards from the row through the fragments before it. Everything
    in between must be a table row: prose between a caption and a row means
    the row does not belong to that caption, which is the "proximity is not
    qualification" rule `finance/guidance.py` documents at length after a live
    run published a historical share count as guidance because "expects"
    appeared in a paragraph above the table.

    Returns the first prospective caption reached. Returns None the moment
    anything reported-looking intervenes, so a reported caption can never
    qualify anything.
    """
    if not sentence.strip() or not document_text:
        return None
    flat_document = _normalise(document_text)
    offset = flat_document.find(_normalise(sentence))
    if offset < 0:
        return None

    window = flat_document[max(0, offset - _BLOCK_REACH_CHARS):offset]
    for fragment in reversed(_BLOCK_FRAGMENT_SPLIT.split(window)):
        fragment = fragment.strip()
        if not fragment:
            continue
        # A reported or analyst caption ends the search rather than being
        # skipped over: it governs these rows, and it does not qualify them.
        if _HISTORICAL_TABLE.search(fragment) or _ANALYST_ESTIMATE.search(fragment):
            return None
        if _PROSPECTIVE.search(fragment):
            return fragment
        if len(fragment.split()) > _BLOCK_ROW_MAX_WORDS:
            return None      # prose intervened; the row is not under a caption
    return None


def prospective_semantics(candidate: GuidanceCandidate,
                          document_text: str = "") -> Tuple[bool, str, str]:
    """(ok, code, reason) for "is this a forward statement at all?".

    Two ways a figure may be established as forward-looking, because issuers
    state guidance two ways. A NARRATED figure carries its own qualification
    ("the company expects revenue of..."). A TABULATED figure does not -- a
    row says "Revenues $9.7B - $10.5B", never "we expect revenues" -- and is
    qualified by the caption above it.

    Testing only the sentence meant tabulated guidance could not pass the
    boundary at all, however well it was read. The caption path is added under
    conditions strict enough that it cannot admit history: see
    `governing_caption`.
    """
    sentence = candidate.source_sentence or ""
    if not candidate.prospective:
        return False, RejectionCode.NOT_PROSPECTIVE, "the candidate is not marked prospective"
    if _ANALYST_ESTIMATE.search(sentence):
        return False, RejectionCode.ANALYST_ESTIMATE, (
            "the cited sentence attributes the figure to analysts rather than to "
            "management")
    if _HISTORICAL_TABLE.search(sentence):
        return False, RejectionCode.HISTORICAL_TABLE, (
            "the cited sentence is part of a reported-results table")
    if _PROSPECTIVE.search(sentence):
        return True, "", ""
    if governing_caption(sentence, document_text):
        return True, "", ""
    return False, RejectionCode.NOT_PROSPECTIVE, (
        "neither the cited sentence nor the caption above it establishes the "
        "figure as forward-looking")


_PERIOD_LABEL = re.compile(r"(?i)^(?:(Q[1-4])\s+)?FY\s?(\d{4})$")


def parse_target_period(label: Optional[str]) -> Optional[gm.GuidancePeriod]:
    """A model-supplied period label to the domain's own period object.

    One representation for actual periods, guidance targets and comparison
    periods (§13). A label this cannot parse is ambiguous and is refused
    rather than guessed at.
    """
    if not label:
        return None
    match = _PERIOD_LABEL.match(label.strip())
    if match is None:
        return None
    quarter, year = match.group(1), int(match.group(2))
    if quarter:
        return gm.GuidancePeriod(label=f"{quarter.upper()} FY{year}",
                                 period_type=gm.GuidancePeriodType.QUARTER,
                                 fiscal_year=year, quarter=int(quarter[1]))
    return gm.GuidancePeriod(label=f"FY{year}",
                             period_type=gm.GuidancePeriodType.ANNUAL,
                             fiscal_year=year)


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------

class GuidanceCandidateValidator:
    """One centralized acceptance boundary for extracted guidance."""

    def __init__(self, document_text: str = "", issued_at: Optional[str] = None,
                 reported_actuals: Optional[Sequence[ReportedActualCandidate]] = None,
                 min_confidence: Optional[float] = None):
        self.document_text = document_text or ""
        self.issued_at = issued_at
        self.reported_actuals = list(reported_actuals or [])
        self.min_confidence = (min_confidence if min_confidence is not None
                               else config.finance_extraction_min_confidence())
        # Derivations computed during `validate`, consumed by `_to_metric`.
        # Keyed by candidate identity so a derived endpoint can never be
        # attached to a different candidate than the one it was computed for.
        self._derivations = {}

    # -- one candidate ---------------------------------------------------

    def validate(self, candidate: GuidanceCandidate
                 ) -> Tuple[Optional[gm.GuidanceMetric], Optional[str], str]:
        """(metric, rejection code, reason). Exactly one of the first two."""
        if not candidate.source_sentence.strip():
            return None, RejectionCode.NO_EVIDENCE, (
                "the candidate cites no sentence, so nothing about it can be checked")
        if self.document_text and not evidence_is_grounded(
                candidate.source_sentence, self.document_text):
            return None, RejectionCode.EVIDENCE_NOT_IN_SOURCE, (
                "the cited sentence does not appear in the supplied document section")

        grounded, why = value_is_grounded(candidate)
        if not grounded:
            return None, RejectionCode.VALUE_NOT_IN_EVIDENCE, why

        if candidate.value_type == ValueType.TOLERANCE:
            # Operands are grounded; now the arithmetic, which fails closed on
            # every §6 condition rather than producing a plausible number.
            derivation, why = tolerance_module.derive(candidate)
            if derivation is None:
                return None, RejectionCode.TOLERANCE_NOT_DERIVABLE, why
            self._derivations[id(candidate)] = derivation

        ok, code, why = prospective_semantics(candidate, self.document_text)
        if not ok:
            return None, code, why

        # An INDEPENDENT requirement, and deliberately early: numeric,
        # unit and period grounding all passed on "inside same-store sales to
        # increase 2% to 5%" proposed as consolidated revenue_growth. Nothing
        # asked whether the sentence was about that metric at all.
        grounding = metric_grounding.ground_metric(
            candidate.metric_id, candidate.source_sentence,
            comparison_period=candidate.comparison_period,
            basis=_domain_basis(candidate.basis))
        if not grounding.supported:
            # The result carries its own code: a name the taxonomy does not
            # have is UNKNOWN_METRIC, evidence that does not describe the
            # metric is METRIC_NOT_GROUNDED. Two different failures.
            return None, (grounding.code or RejectionCode.METRIC_NOT_GROUNDED), \
                grounding.reason

        grounded, why = unit_matches_evidence(candidate)
        if not grounded:
            return None, RejectionCode.UNIT_METRIC_MISMATCH, why

        grounded, why = denominator_is_grounded(candidate)
        if not grounded:
            return None, RejectionCode.DENOMINATOR_NOT_IN_EVIDENCE, why

        name, unit, why = resolve_identity(candidate)
        if name is None:
            reject = (RejectionCode.UNKNOWN_METRIC if "taxonomy" in why
                      else RejectionCode.UNIT_METRIC_MISMATCH)
            return None, reject, why

        period = parse_target_period(candidate.target_period)
        if period is None:
            return None, RejectionCode.AMBIGUOUS_TARGET_PERIOD, (
                f"target period {candidate.target_period!r} could not be resolved to a "
                "fiscal period")
        if self.issued_at and not gm._period_is_plausible(period, self.issued_at):  # noqa: SLF001
            return None, RejectionCode.TARGET_PERIOD_IMPLAUSIBLE, (
                f"a release issued {self.issued_at} cannot guide {period.label}")

        if self._period_already_reported(period):
            return None, RejectionCode.TARGET_PERIOD_COMPLETED, (
                f"actual results for {period.label} have been reported, so this "
                "statement describes a completed period rather than an outlook")

        if candidate.confidence < self.min_confidence:
            return None, RejectionCode.LOW_CONFIDENCE, (
                f"confidence {candidate.confidence:.2f} is below the floor "
                f"{self.min_confidence:.2f}")

        return self._to_metric(candidate, name, unit, period), None, ""

    # -- actualization ---------------------------------------------------

    def _period_already_reported(self, period: gm.GuidancePeriod) -> bool:
        """§11: a period whose actuals are in is no longer a forecast."""
        if not self.reported_actuals:
            return False
        for actual in self.reported_actuals:
            if not actual.period_end:
                continue
            if actual.fiscal_year == period.fiscal_year \
                    and period.period_type == gm.GuidancePeriodType.ANNUAL \
                    and (actual.fiscal_period or "").upper() in ("FY", "Q4"):
                return True
        return False

    def _to_metric(self, candidate: GuidanceCandidate, name: str, unit: str,
                   period: gm.GuidancePeriod) -> gm.GuidanceMetric:
        derivation = self._derivations.pop(id(candidate), None)
        if derivation is not None:
            # The endpoints were COMPUTED from grounded operands. They enter
            # here and nowhere else, so there is exactly one path by which a
            # derived number becomes a guidance figure.
            low, high = derivation.derived_low, derivation.derived_high
        else:
            low, high = candidate.bounds()
        if is_percent_unit(candidate.unit) and low is not None and abs(low) > 1.5:
            low, high = low / 100.0, (high / 100.0 if high is not None else None)
        _domain, scale = domain_unit(candidate.unit)
        evidence_id = f"dcf.guidance.{name}.current"
        target_type = gm.classify_target_type(period, None)
        return gm.GuidanceMetric(
            name=name, low=low, high=high if high is not None else low,
            unit=unit, basis=_domain_basis(candidate.basis),
            fiscal_year=period.fiscal_year, evidence_id=evidence_id,
            source_excerpt=candidate.source_sentence,
            scale=scale,
            guidance_id=gm._guidance_id(  # noqa: SLF001
                candidate.document_id or "", candidate.document_id or "",
                name, period.label),
            issued_at=candidate.issued_at or self.issued_at,
            fiscal_period=period.label, period_type=period.period_type,
            target_period_type=target_type,
            forward_kind=gm.classify_forward_information(
                candidate.source_sentence, target_type),
            source_accession=candidate.document_id,
            source_evidence_ids=(evidence_id,),
            bound_type=_bound_type(candidate),
            derivation_type=(derivation.derivation_type if derivation else None),
            derivation_formula=(derivation.formula if derivation else None),
            derivation_operands=(derivation.operand_evidence if derivation else ()),
            status=(gm.GuidanceStatus.WITHDRAWN
                    if candidate.action == GuidanceAction.WITHDRAWN
                    else gm.GuidanceStatus.CURRENT),
            # A table row does not qualify itself, so recording the row as
            # its own prospective evidence would assert something the
            # document never says. Where a caption did the qualifying, the
            # caption is what gets recorded.
            prospective_evidence=(
                candidate.source_sentence
                if _PROSPECTIVE.search(candidate.source_sentence or "")
                else (governing_caption(candidate.source_sentence,
                                        self.document_text)
                      or candidate.source_sentence)))

    # -- a whole set -----------------------------------------------------

    def validate_all(self, candidates: Sequence[GuidanceCandidate]
                     ) -> Tuple[List[gm.GuidanceMetric],
                                List[Tuple[GuidanceCandidate, str, str]]]:
        """Accepted metrics and refusals, with economic identity resolved.

        §9/§12: one economic target yields ONE canonical signal. Two releases
        stating the same FY revenue are one statement; a reaffirmation
        updates the signal and never becomes a second one.
        """
        accepted: List[gm.GuidanceMetric] = []
        rejected: List[Tuple[GuidanceCandidate, str, str]] = []
        by_identity: Dict[tuple, Tuple[GuidanceCandidate, gm.GuidanceMetric]] = {}

        for candidate in candidates:
            metric, code, reason = self.validate(candidate)
            if metric is None:
                rejected.append((candidate, code, reason))
                continue
            identity = gm.guidance_identity(metric)
            existing = by_identity.get(identity)
            if existing is None:
                by_identity[identity] = (candidate, metric)
                continue
            keep = _prefer(existing, (candidate, metric))
            if keep is not existing:
                by_identity[identity] = keep
                rejected.append((existing[0], RejectionCode.DUPLICATE_ECONOMIC_IDENTITY,
                                 "superseded by a newer statement of the same economic "
                                 "target"))
            else:
                rejected.append((candidate, RejectionCode.DUPLICATE_ECONOMIC_IDENTITY,
                                 "an equal or newer statement of the same economic target "
                                 "is already accepted"))

        accepted = [metric for _c, metric in by_identity.values()]
        return accepted, rejected


def _bound_type(candidate: GuidanceCandidate) -> str:
    if candidate.value_type == ValueType.TOLERANCE:
        # NOT `RANGE`. A stated range and a derived one are different claims
        # about what the company published, and collapsing them here would
        # erase that distinction for every consumer downstream.
        return gm.GuidanceBound.POINT_WITH_TOLERANCE
    if candidate.value_type == ValueType.FLOOR:
        return gm.GuidanceBound.AT_LEAST
    low, high = candidate.bounds()
    if low is not None and high is not None and low != high:
        return gm.GuidanceBound.RANGE
    return gm.GuidanceBound.APPROXIMATELY


def _prefer(left: Tuple[GuidanceCandidate, gm.GuidanceMetric],
            right: Tuple[GuidanceCandidate, gm.GuidanceMetric]
            ) -> Tuple[GuidanceCandidate, gm.GuidanceMetric]:
    """Which of two statements of ONE economic target is the current signal.

    Newest issue date wins -- that is what supersession means. A REAFFIRMED
    or REITERATED statement of the same figure updates the date and nothing
    else, which is precisely why it must not be admitted as a second signal.
    """
    left_at = (left[1].issued_at or "", left[0].confidence)
    right_at = (right[1].issued_at or "", right[0].confidence)
    return right if right_at > left_at else left
