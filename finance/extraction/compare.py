"""V1 against V2, for measurement. Not for choosing.

§15 is explicit and the reason is worth stating: compare mode must not
silently prefer V2 because it is new. So this module RECORDS a disagreement
and returns it; it never resolves one. Whether V2 becomes the default is a
decision made from benchmark numbers by a person, not a runtime fallback made
by whichever extractor answered last.

A disagreement is reported per (metric, target period), because that is the
unit a reader can check: "V1 said the FY2027 revenue guide was absent and V2
said $90 billion" is actionable, and "the two returned different numbers of
statements" is not.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from finance import guidance as gm


# What kind of disagreement this is. Grouped so a benchmark can report
# FAILURE CLASSES rather than a stream of individual sentences (§27).
class DisagreementKind:
    ONLY_V1 = "ONLY_V1"                 # V1 found it, V2 did not
    ONLY_V2 = "ONLY_V2"                 # V2 found it, V1 did not
    VALUE = "VALUE"                     # both found it, different numbers
    UNIT = "UNIT"                       # both found it, different units
    METRIC_IDENTITY = "METRIC_IDENTITY"  # same period, different metric
    BASIS = "BASIS"
    AGREE = "AGREE"

    ALL = (ONLY_V1, ONLY_V2, VALUE, UNIT, METRIC_IDENTITY, BASIS, AGREE)


@dataclass(frozen=True)
class Disagreement:
    metric: Optional[str]
    target_period: Optional[str]
    kind: str
    v1: Optional[dict] = None
    v2: Optional[dict] = None
    validator_status: Optional[str] = None
    expected: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"metric": self.metric, "target_period": self.target_period,
                "kind": self.kind, "v1": self.v1, "v2": self.v2,
                "validator_status": self.validator_status,
                "expected": self.expected}


@dataclass
class ComparisonResult:
    """Both answers side by side. Neither is chosen here."""

    v1_count: int = 0
    v2_count: int = 0
    agreements: int = 0
    disagreements: List[Disagreement] = field(default_factory=list)
    v2_rejections: Dict[str, int] = field(default_factory=dict)
    v2_unsupported: int = 0
    notes: List[str] = field(default_factory=list)

    def by_kind(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.disagreements:
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {"v1_count": self.v1_count, "v2_count": self.v2_count,
                "agreements": self.agreements,
                "disagreement_count": len(self.disagreements),
                "disagreements_by_kind": self.by_kind(),
                "disagreements": [d.to_dict() for d in self.disagreements],
                "v2_rejection_codes": dict(self.v2_rejections),
                "v2_unsupported": self.v2_unsupported,
                "notes": list(self.notes)}


def _key(metric) -> Tuple[Optional[str], Optional[str]]:
    """The unit a reader compares on: one metric, one target period."""
    if isinstance(metric, dict):
        return metric.get("name"), (metric.get("target_period")
                                    or metric.get("fiscal_period"))
    return getattr(metric, "name", None), getattr(metric, "fiscal_period", None)


def _summary(metric) -> Optional[dict]:
    if metric is None:
        return None
    if isinstance(metric, dict):
        source = metric
    else:
        source = metric.to_dict()
    return {k: source.get(k) for k in
            ("name", "low", "high", "unit", "scale", "basis", "target_period",
             "fiscal_period", "target_period_type", "bound_type")}


_APPROX = 1e-9


def _values_agree(left: dict, right: dict) -> bool:
    for field_name in ("low", "high"):
        a, b = left.get(field_name), right.get(field_name)
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if abs(a - b) > max(abs(b) * 1e-6, _APPROX):
            return False
    return True


def compare(v1_metrics: Sequence, v2_metrics: Sequence,
            v2_rejections: Optional[Dict[str, int]] = None,
            v2_unsupported: int = 0,
            expected: Optional[Dict[Tuple[str, str], dict]] = None
            ) -> ComparisonResult:
    """Line up two extractions and record where they differ.

    `expected` is the ground-truth fixture when one exists (§15 asks for it
    in the comparison output). Where it is absent the comparison still
    reports the disagreement -- knowing the two layers disagree is useful
    before knowing which is right.
    """
    result = ComparisonResult(v1_count=len(v1_metrics), v2_count=len(v2_metrics),
                              v2_rejections=dict(v2_rejections or {}),
                              v2_unsupported=v2_unsupported)
    left = {_key(m): m for m in v1_metrics}
    right = {_key(m): m for m in v2_metrics}
    truth = expected or {}

    for key in sorted(set(left) | set(right), key=lambda k: (k[0] or "", k[1] or "")):
        metric, period = key
        one, two = left.get(key), right.get(key)
        ground = truth.get(key)
        if one is None:
            result.disagreements.append(Disagreement(
                metric, period, DisagreementKind.ONLY_V2,
                v2=_summary(two), expected=ground))
            continue
        if two is None:
            result.disagreements.append(Disagreement(
                metric, period, DisagreementKind.ONLY_V1,
                v1=_summary(one), expected=ground))
            continue

        a, b = _summary(one), _summary(two)
        if a.get("unit") != b.get("unit"):
            kind = DisagreementKind.UNIT
        elif a.get("basis") != b.get("basis"):
            kind = DisagreementKind.BASIS
        elif not _values_agree(a, b):
            kind = DisagreementKind.VALUE
        else:
            result.agreements += 1
            continue
        result.disagreements.append(Disagreement(
            metric, period, kind, v1=a, v2=b, expected=ground))
    return result


def run_comparison(document_text: str, symbol: str, accession: str, document: str,
                   filed: str, extractor, expected=None,
                   fiscal_year: Optional[int] = None) -> ComparisonResult:
    """Extract one document both ways and compare. Chooses nothing.

    V1 is `finance.guidance.extract_guidance_from_text`, unchanged and still
    the default. V2 is the semantic extractor plus the validator.
    """
    from finance.extraction.semantic_extractor import ExtractionFailure, select_sections
    from finance.extraction.validator import GuidanceCandidateValidator

    v1_release = gm.extract_guidance_from_text(
        document_text, symbol, accession, document, filed,
        expected_fiscal_year=fiscal_year)
    v1_metrics = list(v1_release.all_metrics or v1_release.metrics.values())

    sections = select_sections(document_text)
    result_notes = []
    v2_metrics, rejections, unsupported = [], {}, 0
    try:
        candidates = extractor.extract(sections, issued_at=filed,
                                       document_id=accession)
        validator = GuidanceCandidateValidator(
            document_text=document_text, issued_at=filed)
        v2_metrics, rejected = validator.validate_all(candidates)
        for _candidate, code, _reason in rejected:
            rejections[code] = rejections.get(code, 0) + 1
        from finance.extraction.schema import RejectionCode
        unsupported = sum(1 for _c, code, _r in rejected
                          if code in RejectionCode.UNSUPPORTED)
    except ExtractionFailure as failure:
        # Fail closed. V2 contributes nothing rather than falling back to an
        # unvalidated read, and the reason is recorded (§20).
        result_notes.append(f"V2 extraction failed ({failure.code}): {failure.message}")

    comparison = compare(v1_metrics, v2_metrics, rejections, unsupported, expected)
    comparison.notes.extend(result_notes)
    comparison.notes.append(f"sections read: {len(sections)}")
    return comparison
