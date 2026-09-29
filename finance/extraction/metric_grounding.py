"""Does the cited evidence describe the metric the candidate claims?

THE FAILURE THIS EXISTS FOR

    evidence:  "The Company expects inside same-store sales to increase 2% to 5%"
    candidate: revenue_growth, FY2027, 2%-5%

Every existing check passed. The numbers were in the sentence, the unit was a
ratio, the period resolved, the sentence was prospective. Nothing asked the
only question that mattered: is this sentence ABOUT consolidated revenue?

It is not. Same-store sales is a component measure, and a retailer's total
revenue can move quite differently from it -- new stores, closures,
acquisitions. Accepted under the name `revenue_growth` it becomes a DCF
revenue-growth assumption, because `guidance.may_anchor_revenue_growth` gates
that path by NAME and the name was already wrong by then.

So metric identity gets its own acceptance requirement, independent of the
numeric ones.

WHY THIS IS NOT A NAME-MATCHING CHECK

Companies do not use canonical identifiers. "Net sales", "total revenue" and
"turnover" are all consolidated revenue, and requiring the literal string
`revenue` would refuse most real releases. So the check runs the TAXONOMY'S
OWN pattern for the claimed metric -- the same `_METRIC_PATTERNS` entry V1
matches with, which already encodes every alias the project recognises. There
is no second metric dictionary here, and adding one would be the bug this
module is named after in a different form.

WHAT THE TAXONOMY ALREADY KNEW

Every entry carries a SCOPE: consolidated, service, product, segment,
component. That vocabulary is what distinguishes "cloud revenue growth" from
"revenue growth" and it was already there -- unused by the acceptance
boundary. This module puts it to work.

WHAT IT ADDS

One thing the taxonomy could not express: a narrowing qualifier on a metric
whose canonical scope is consolidated. "Same-store sales" is not a separate
taxonomy entry, and the consolidated `revenue` pattern matches the bare word
"sales" inside it. A qualifier that narrows the subject is therefore detected
directly, and a consolidated claim narrowed by one is refused.

FAIL CLOSED

Where scope cannot be established the answer is refusal. A component read as
the whole company is a wrong number presented as a right one; a refused
statement is only a missing one.
"""

import re
from dataclasses import dataclass
from typing import Optional

from finance import guidance as gm


class MatchType:
    """How the evidence supports the claimed metric, if it does."""

    EXACT_CANONICAL = "EXACT_CANONICAL"
    APPROVED_ALIAS = "APPROVED_ALIAS"
    APPROVED_DERIVATION = "APPROVED_DERIVATION"
    COMPONENT_METRIC = "COMPONENT_METRIC"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"

    ALL = (EXACT_CANONICAL, APPROVED_ALIAS, APPROVED_DERIVATION,
           COMPONENT_METRIC, AMBIGUOUS, UNSUPPORTED)

    SUPPORTING = (EXACT_CANONICAL, APPROVED_ALIAS, APPROVED_DERIVATION)


@dataclass(frozen=True)
class MetricGroundingResult:
    """The deterministic answer, with everything needed to explain it."""

    supported: bool
    canonical_metric: Optional[str]
    evidence_metric: Optional[str]
    scope: Optional[str]
    match_type: str
    reason: str = ""
    code: Optional[str] = None

    def to_dict(self) -> dict:
        return {"supported": self.supported,
                "canonical_metric": self.canonical_metric,
                "evidence_metric": self.evidence_metric,
                "scope": self.scope, "match_type": self.match_type,
                "reason": self.reason, "code": self.code}


# The head nouns a narrowing qualifier attaches to. Only the subjects whose
# consolidated-vs-component distinction actually changes a valuation input --
# this is not a general noun list.
_HEAD_NOUN = r"(?:sales|revenues?|income|earnings|margins?|bookings|billings)"

# Qualifiers that KEEP a metric consolidated. Taken from the taxonomy's own
# patterns, which already spell out the words that do not narrow anything.
_NON_NARROWING = frozenset({
    "consolidated", "total", "worldwide", "net", "gross", "company",
    "companywide", "overall", "aggregate", "combined", "global", "adjusted",
    "non-gaap", "gaap", "reported", "underlying", "core", "organic",
    "annual", "quarterly", "full-year", "fiscal", "diluted", "basic",
    "operating", "free", "comparable-basis",
})

# A qualifier that NARROWS the subject to a part of the company. §4's list,
# plus the scope words the taxonomy already uses. A hyphenated compound
# immediately before the noun ("same-store", "comparable-store") is treated as
# narrowing on its own, which is what generalises this beyond the listed
# words.
_NARROWING_WORDS = frozenset({
    "same-store", "comparable-store", "comparable", "same", "store",
    "cloud", "ai", "datacenter", "data-center", "segment", "segmental",
    "international", "domestic", "regional", "subscription", "recurring",
    "saas", "software", "service", "services", "product", "products",
    "hardware", "licensing", "advertising", "subscriber", "retail",
    "wholesale", "commercial", "consumer", "enterprise", "inside",
    "fuel", "merchandise", "grocery", "pharmacy", "digital", "online",
    "e-commerce", "ecommerce", "mobility", "wireline", "wireless",
    "broadband", "gaming", "automotive", "datacentre",
})

# Verbs of change. The only vocabulary this module adds, and deliberately not
# metric vocabulary: they say a quantity MOVED, which is what makes a level
# sentence evidence for a growth metric.
_CHANGE_VERB = re.compile(
    r"(?i)\b(?:grow\w*|increas\w*|decreas\w*|declin\w*|ris\w*|fall\w*"
    r"|expand\w*|contract\w*|improv\w*|higher|lower|up|down)\b")

_QUALIFIED_NOUN = re.compile(
    r"(?i)([\w][\w'-]*)\s+(?:" + _HEAD_NOUN + r")\b")


def _is_narrowing(word: str) -> bool:
    token = (word or "").strip().lower()
    if not token or token in _NON_NARROWING:
        return False
    if token in _NARROWING_WORDS:
        return True
    # A hyphenated compound directly before the noun is a qualifier by
    # construction: "same-store sales", "own-brand revenue". Bare hyphenated
    # words that are non-narrowing are listed above.
    return "-" in token


def narrowing_qualifier(sentence: str) -> Optional[str]:
    """The first qualifier that narrows a metric noun, if any."""
    for match in _QUALIFIED_NOUN.finditer(sentence or ""):
        if _is_narrowing(match.group(1)):
            return match.group(1)
    return None


def _entry(metric_id: str):
    return gm._METRIC_BY_NAME.get(metric_id)          # noqa: SLF001


def _competing_narrower_metric(sentence: str, claimed_scope: str) -> Optional[tuple]:
    """A taxonomy metric with a NARROWER scope that this sentence describes.

    "cloud revenue growth" matches both the component entry and, through the
    bare noun, the consolidated one. The narrower reading is the right one:
    the sentence named a part.
    """
    for entry in gm._METRIC_PATTERNS:                 # noqa: SLF001
        name, _unit, _ratio, scope, _basis, pattern = entry
        if scope == gm_SCOPE_CONSOLIDATED or scope == claimed_scope:
            continue
        if pattern.search(sentence or ""):
            return entry
    return None


gm_SCOPE_CONSOLIDATED = "consolidated"


def ground_metric(metric_id: Optional[str], sentence: str,
                  comparison_period: Optional[str] = None,
                  basis: Optional[str] = None
                  ) -> MetricGroundingResult:
    """Is this sentence evidence for THIS metric? Deterministically.

    `comparison_period` marks the §8 derivation case: a growth figure stated
    against a named prior period is derived from two grounded levels rather
    than read directly, and must not be refused for lacking a growth phrase.
    """
    if not metric_id:
        return MetricGroundingResult(
            False, None, None, None, MatchType.UNSUPPORTED,
            "the candidate names no metric", "METRIC_NOT_GROUNDED")

    entry = _entry(metric_id)
    if entry is None:
        return MetricGroundingResult(
            False, metric_id, None, None, MatchType.UNSUPPORTED,
            f"{metric_id!r} is not in the reviewed metric taxonomy",
            "UNKNOWN_METRIC")

    _name, _unit, _is_ratio, scope, _basis, pattern = entry
    text = sentence or ""

    direct = bool(pattern.search(text))
    derived_from = None
    derivation_type = None

    if not direct:
        # A ratio metric is often stated as its LEVEL plus what the
        # percentage is of: "non-GAAP operating income to be 20% to 22% of
        # revenue" is adjusted_operating_margin, and the taxonomy's margin
        # pattern does not match it because the sentence never says "margin".
        #
        # The level metric is found by ASKING the taxonomy rather than by
        # stripping a suffix -- `adjusted_operating_margin` minus "_margin"
        # is not a metric, and guessing works for one case and fails for the
        # rest. Every entry whose pattern matches the sentence is offered to
        # `guidance.resolve_percentage_identity`, the project's own rule for
        # what a percentage on a level metric means, and if one resolves to
        # the claimed metric the claim is grounded.
        for level in gm._METRIC_PATTERNS:                  # noqa: SLF001
            level_name, level_unit, _r, _scope, level_basis, level_pattern = level
            if level_name == metric_id or not level_pattern.search(text):
                continue
            resolved = gm.resolve_percentage_identity(
                level_name, level_unit,
                basis or level_basis or gm.BASIS_GAAP, text)
            if resolved and resolved[0] == metric_id:
                direct = True
                derived_from = level_name
                derivation_type = MatchType.APPROVED_DERIVATION
                break

    if not direct and metric_id.endswith("_growth"):
        # A growth metric is grounded by its LEVEL plus a change verb.
        # "Net sales are expected to increase 6%" describes revenue growth as
        # plainly as "revenue growth of 6%" does, and the taxonomy's growth
        # pattern -- written to match V1's own narrower phrasings -- does not
        # cover it. Decomposing rather than widening that pattern keeps one
        # metric dictionary and adds no new metric vocabulary: the only new
        # words are verbs of change.
        level = _entry(metric_id[: -len("_growth")])
        if level is not None and level[5].search(text):
            if _CHANGE_VERB.search(text):
                direct = True
                derived_from = level[0]
                derivation_type = MatchType.APPROVED_ALIAS
            elif comparison_period:
                # §8: a level stated against a named prior period is a valid
                # deterministic derivation, not an ungrounded growth claim.
                direct = True
                derived_from = level[0]
                derivation_type = MatchType.APPROVED_DERIVATION

    if not direct:
        return MetricGroundingResult(
            False, metric_id, None, scope, MatchType.UNSUPPORTED,
            f"the cited sentence does not describe {metric_id!r}",
            "METRIC_NOT_GROUNDED")

    # The pattern matched. If the claim is CONSOLIDATED, make sure the
    # sentence is not talking about a part of the company.
    if scope == gm_SCOPE_CONSOLIDATED:
        narrower = narrowing_qualifier(text)
        if narrower:
            return MetricGroundingResult(
                False, metric_id, None, scope, MatchType.COMPONENT_METRIC,
                f"the evidence describes {narrower!r} {metric_id!r}, which is a "
                "component measure, and the candidate claims the consolidated one",
                "METRIC_NOT_GROUNDED")
        competing = _competing_narrower_metric(text, scope)
        if competing is not None:
            return MetricGroundingResult(
                False, metric_id, competing[0], scope, MatchType.COMPONENT_METRIC,
                f"the evidence describes {competing[0]!r} (scope {competing[3]}), "
                "not the consolidated metric claimed", "METRIC_NOT_GROUNDED")

    # A canonical mention is one whose own name appears; anything else the
    # taxonomy pattern accepted is an approved alias ("net sales", "turnover").
    canonical_words = metric_id.replace("_", " ")
    match_type = derivation_type or (
        MatchType.EXACT_CANONICAL if canonical_words in text.lower()
        else MatchType.APPROVED_ALIAS)
    return MetricGroundingResult(
        True, metric_id, derived_from or metric_id, scope, match_type,
        "the cited sentence describes this metric"
        if derived_from is None else
        f"the sentence describes {derived_from!r} changing, which grounds "
        f"{metric_id!r}")
