"""Phase H.11, sections 16-27 — where growth came from, when the issuer says.

A 15% increase in revenue can be fifteen points of additional volume, or a
commodity pass-through that reverses next year, or an acquisition that will
not repeat once it laps, or a currency move. Those are different businesses
with the same headline number, and a forecast that treats them alike is
guessing. The assumption builder had no way to tell them apart, because
reported growth arrived as a single rate with nothing attached.

This module builds a BRIDGE: the reported rate, plus whatever contributions
the issuer's own filings state, plus an explicit record of how much of the
change remains unexplained. Two rules keep it honest:

  * only contributions the issuer states NUMERICALLY are recorded. A release
    that says growth was "driven by strong pricing" has said something real
    and unquantified; inventing a number for it would be worse than the gap
    it fills. Such statements are noted as qualitative and contribute
    nothing to the arithmetic.

  * the decomposition is never completed by inference. What is not accounted
    for stays in `unexplained_component`, and coverage is reported as
    PARTIAL rather than rounded up to FULL.

Nothing here classifies by sector, and no issuer is named. A company is
"acquisition-heavy" because its own filing said an acquisition contributed
most of the change, or it is UNKNOWN.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


class GrowthComponentType:
    ORGANIC_VOLUME = "ORGANIC_VOLUME"
    PRICING = "PRICING"
    COMMODITY_PRICE_EFFECT = "COMMODITY_PRICE_EFFECT"
    ACQUISITION = "ACQUISITION"
    DIVESTITURE = "DIVESTITURE"
    FX = "FX"
    MIX = "MIX"
    NEW_PRODUCT = "NEW_PRODUCT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

    ALL = (ORGANIC_VOLUME, PRICING, COMMODITY_PRICE_EFFECT, ACQUISITION,
           DIVESTITURE, FX, MIX, NEW_PRODUCT, OTHER, UNKNOWN)


class GrowthCoverage:
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


class GrowthQuality:
    """Section 19 — a coarse verdict, only ever from measured contributions."""

    PREDOMINANTLY_ORGANIC = "PREDOMINANTLY_ORGANIC"
    MIXED = "MIXED"
    ACQUISITION_HEAVY = "ACQUISITION_HEAVY"
    PRICING_HEAVY = "PRICING_HEAVY"
    COMMODITY_PRICE_HEAVY = "COMMODITY_PRICE_HEAVY"
    FX_HEAVY = "FX_HEAVY"
    UNKNOWN = "UNKNOWN"


HEADLINE_GROWTH_COMPOSITION_UNCERTAIN = "HEADLINE_GROWTH_COMPOSITION_UNCERTAIN"
GROWTH_COMPOSITION_UNCERTAIN = "GROWTH_COMPOSITION_UNCERTAIN"

# Section 24: stable evidence ids.
EVIDENCE_ID_PREFIX = "growth.bridge"
COMPONENT_EVIDENCE_IDS = {
    GrowthComponentType.PRICING: f"{EVIDENCE_ID_PREFIX}.pricing",
    GrowthComponentType.ORGANIC_VOLUME: f"{EVIDENCE_ID_PREFIX}.volume",
    GrowthComponentType.ACQUISITION: f"{EVIDENCE_ID_PREFIX}.acquisition",
    GrowthComponentType.FX: f"{EVIDENCE_ID_PREFIX}.fx",
    GrowthComponentType.COMMODITY_PRICE_EFFECT: f"{EVIDENCE_ID_PREFIX}.commodity",
    GrowthComponentType.DIVESTITURE: f"{EVIDENCE_ID_PREFIX}.divestiture",
    GrowthComponentType.MIX: f"{EVIDENCE_ID_PREFIX}.mix",
    GrowthComponentType.NEW_PRODUCT: f"{EVIDENCE_ID_PREFIX}.new_product",
    GrowthComponentType.OTHER: f"{EVIDENCE_ID_PREFIX}.other",
}

# How much of the reported change one component must carry before the coarse
# classification names it. Below this the mix is MIXED, which is the honest
# answer for a business growing several ways at once.
_DOMINANCE = 0.60

# A share of the reported change that must be explained before coverage is
# called FULL rather than PARTIAL.
_FULL_COVERAGE = 0.85

# Above this reported rate, an unexplained composition matters enough to say
# so (section 27). A company growing 3% with no bridge is unremarkable; one
# growing 25% with no bridge is a forecast risk.
MATERIAL_GROWTH_FOR_COMPOSITION = 0.10


@dataclass
class GrowthComponent:
    """One stated contribution to a change in revenue."""

    type: str
    contribution: Optional[float] = None      # in the same units as the rate
    units: str = "percentage_points"
    evidence_ids: tuple = ()
    source_excerpt: str = ""
    qualitative_only: bool = False

    def to_dict(self) -> dict:
        return {
            "type": self.type, "contribution": self.contribution, "units": self.units,
            "evidence_ids": list(self.evidence_ids),
            "source_excerpt": self.source_excerpt[:240],
            "qualitative_only": self.qualitative_only,
        }


@dataclass
class GrowthBridge:
    """Section 17 — the reported rate and what the issuer attributes it to."""

    period: Optional[str] = None
    reported_growth: Optional[float] = None
    components: List[GrowthComponent] = field(default_factory=list)
    unexplained_component: Optional[float] = None
    coverage_status: str = GrowthCoverage.NONE
    quality: str = GrowthQuality.UNKNOWN
    findings: list = field(default_factory=list)

    @property
    def measured(self) -> List[GrowthComponent]:
        return [c for c in self.components
                if c.contribution is not None and not c.qualitative_only]

    def contribution_of(self, component_type: str) -> Optional[float]:
        for component in self.measured:
            if component.type == component_type:
                return component.contribution
        return None

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "reported_growth": self.reported_growth,
            "components": [c.to_dict() for c in self.components],
            "unexplained_component": self.unexplained_component,
            "coverage_status": self.coverage_status,
            "quality": self.quality,
            "findings": [dict(f) for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Section 18 — reading contributions the issuer stated in its own filing
# ---------------------------------------------------------------------------
#
# Narrow by construction. Each pattern requires a NUMBER attached to an
# attribution word in the same clause; prose that merely names a driver is
# captured separately as qualitative. The alternative -- inferring magnitudes
# from adjectives -- is the kind of invention this project refuses everywhere
# else.

_PERCENT = r"(?P<value>\d{1,3}(?:\.\d+)?)\s*(?:%|percentage\s+points?|ppt|bps)"

_ATTRIBUTION_PATTERNS = (
    (GrowthComponentType.ACQUISITION, re.compile(
        r"(?i)acquisitions?\s+(?:of\s+[\w\s]{0,40}?)?"
        r"(?:contributed|added|accounted\s+for|represented)\s+(?:approximately\s+)?"
        + _PERCENT)),
    (GrowthComponentType.ACQUISITION, re.compile(
        r"(?i)" + _PERCENT + r"\s+(?:of\s+(?:the\s+)?(?:growth|increase)\s+)?"
        r"(?:was\s+|were\s+)?(?:attributable\s+to|from|due\s+to)\s+(?:the\s+)?acquisitions?")),
    (GrowthComponentType.PRICING, re.compile(
        r"(?i)(?:higher\s+|net\s+|favou?rable\s+)?pric(?:ing|es?)\s+"
        r"(?:contributed|added|accounted\s+for|increased\s+\w+\s+by)\s+"
        r"(?:approximately\s+)?" + _PERCENT)),
    (GrowthComponentType.ORGANIC_VOLUME, re.compile(
        r"(?i)(?:volume|unit\s+volume|organic\s+(?:growth|volume))s?\s+"
        r"(?:contributed|added|grew|increased)\s+(?:by\s+)?(?:approximately\s+)?"
        + _PERCENT)),
    (GrowthComponentType.FX, re.compile(
        r"(?i)(?:currency|foreign\s+exchange|fx)\s+"
        r"(?:translation\s+)?(?:contributed|added|reduced|decreased|increased|"
        r"had\s+an?\s+impact\s+of)\s+"
        # A verb and its number are routinely separated by what was
        # affected ("reduced NET SALES by 2.0%"), so a few words are
        # allowed between them -- bounded, and non-greedy, so the
        # number still has to belong to this clause.
        r"(?:\w+\s+){0,3}?(?:by\s+)?(?:approximately\s+)?" + _PERCENT)),
    (GrowthComponentType.COMMODITY_PRICE_EFFECT, re.compile(
        r"(?i)(?:commodity|raw\s+material|metal|copper|aluminum|aluminium|"
        r"input)\s+(?:price\s+)?(?:pass[\s-]?through\s+)?"
        r"(?:contributed|added|accounted\s+for|increased\s+\w+\s+by)\s+"
        r"(?:approximately\s+)?" + _PERCENT)),
    (GrowthComponentType.DIVESTITURE, re.compile(
        r"(?i)divestitures?\s+(?:reduced|decreased|lowered)\s+"
        r"(?:\w+\s+){0,3}?(?:by\s+)?(?:approximately\s+)?" + _PERCENT)),
)

# Drivers named WITHOUT a number. Recorded so the bridge can say the issuer
# attributed growth to something it did not size -- which is different from
# the issuer saying nothing at all.
_QUALITATIVE_PATTERNS = (
    (GrowthComponentType.PRICING,
     re.compile(r"(?i)\b(?:driven|led|aided|supported)\s+by\s+[\w\s,]{0,30}?pric")),
    (GrowthComponentType.ACQUISITION,
     re.compile(r"(?i)\b(?:driven|led|aided|supported)\s+by\s+[\w\s,]{0,30}?acquisition")),
    (GrowthComponentType.ORGANIC_VOLUME,
     re.compile(r"(?i)\b(?:driven|led|aided|supported)\s+by\s+[\w\s,]{0,30}?"
                r"(?:volume|organic)")),
    (GrowthComponentType.COMMODITY_PRICE_EFFECT,
     re.compile(r"(?i)\b(?:driven|led|aided|supported)\s+by\s+[\w\s,]{0,30}?"
                r"(?:commodity|raw\s+material|metal)")),
    (GrowthComponentType.FX,
     re.compile(r"(?i)\b(?:driven|led|aided|supported)\s+by\s+[\w\s,]{0,30}?"
                r"(?:currency|foreign\s+exchange)")),
)

# A sign-flipping context: "reduced", "decreased", "headwind" mean the
# contribution subtracts from the reported change.
_NEGATIVE_CONTEXT = re.compile(
    r"(?i)\b(?:reduced|decreased|lowered|headwind|drag|unfavou?rable|negative)\b")


def _excerpt(text: str, match) -> str:
    start = max(0, match.start() - 60)
    return " ".join(text[start:match.end() + 40].split())


def extract_components(text: str, evidence_id: Optional[str] = None
                       ) -> List[GrowthComponent]:
    """Contributions the issuer stated, from one filed passage.

    Numbers only. A driver named without a magnitude becomes a qualitative
    component carrying no contribution, so the arithmetic below cannot be
    completed by something the company never quantified.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    components: List[GrowthComponent] = []
    seen_types = set()

    for component_type, pattern in _ATTRIBUTION_PATTERNS:
        match = pattern.search(text)
        if not match or component_type in seen_types:
            continue
        try:
            value = float(match.group("value")) / 100.0
        except (TypeError, ValueError):
            continue
        window = text[max(0, match.start() - 50):match.end() + 20]
        if component_type == GrowthComponentType.DIVESTITURE or \
                _NEGATIVE_CONTEXT.search(window):
            value = -abs(value)
        components.append(GrowthComponent(
            type=component_type, contribution=value,
            evidence_ids=tuple(filter(None, (evidence_id,
                                             COMPONENT_EVIDENCE_IDS.get(component_type)))),
            source_excerpt=_excerpt(text, match)))
        seen_types.add(component_type)

    for component_type, pattern in _QUALITATIVE_PATTERNS:
        if component_type in seen_types:
            continue
        match = pattern.search(text)
        if not match:
            continue
        components.append(GrowthComponent(
            type=component_type, contribution=None, qualitative_only=True,
            evidence_ids=tuple(filter(None, (evidence_id,
                                             COMPONENT_EVIDENCE_IDS.get(component_type)))),
            source_excerpt=_excerpt(text, match)))
        seen_types.add(component_type)

    return components


def _classify(bridge: GrowthBridge) -> str:
    """Section 19: a coarse verdict, only from measured contributions."""
    measured = bridge.measured
    if not measured or not bridge.reported_growth:
        return GrowthQuality.UNKNOWN
    total = abs(bridge.reported_growth)
    if total == 0:
        return GrowthQuality.UNKNOWN

    shares = {c.type: abs(c.contribution) / total for c in measured}
    dominant = max(shares.items(), key=lambda kv: kv[1])
    if dominant[1] < _DOMINANCE:
        return GrowthQuality.MIXED
    return {
        GrowthComponentType.ACQUISITION: GrowthQuality.ACQUISITION_HEAVY,
        GrowthComponentType.PRICING: GrowthQuality.PRICING_HEAVY,
        GrowthComponentType.COMMODITY_PRICE_EFFECT: GrowthQuality.COMMODITY_PRICE_HEAVY,
        GrowthComponentType.FX: GrowthQuality.FX_HEAVY,
        GrowthComponentType.ORGANIC_VOLUME: GrowthQuality.PREDOMINANTLY_ORGANIC,
    }.get(dominant[0], GrowthQuality.MIXED)


def build_growth_bridge(reported_growth: Optional[float], period: Optional[str] = None,
                        source_texts=None, revenue_guidance_present: bool = False
                        ) -> GrowthBridge:
    """Sections 16-17 and 27: the bridge, and how much of it is missing.

    `unexplained_component` is the part of the reported change no stated
    contribution accounts for. It is reported rather than distributed among
    the known components, because attributing it would be the invention this
    module exists to avoid.
    """
    bridge = GrowthBridge(period=period, reported_growth=reported_growth)

    for entry in (source_texts or []):
        if isinstance(entry, dict):
            text, evidence_id = entry.get("text"), entry.get("evidence_id")
        else:
            text, evidence_id = entry, None
        bridge.components.extend(extract_components(text, evidence_id))

    measured = bridge.measured
    if reported_growth is not None and measured:
        explained = sum(c.contribution for c in measured)
        bridge.unexplained_component = reported_growth - explained
        if abs(reported_growth) > 0:
            share = abs(explained) / abs(reported_growth)
            bridge.coverage_status = (GrowthCoverage.FULL if share >= _FULL_COVERAGE
                                      else GrowthCoverage.PARTIAL)
        else:
            bridge.coverage_status = GrowthCoverage.PARTIAL
    elif bridge.components:
        # Drivers named but never sized.
        bridge.coverage_status = GrowthCoverage.NONE
        bridge.unexplained_component = reported_growth
    else:
        bridge.coverage_status = GrowthCoverage.NONE
        bridge.unexplained_component = reported_growth

    bridge.quality = _classify(bridge)

    # Section 27: high headline growth, an incomplete bridge and no revenue
    # guidance together mean the forecast is resting on a number whose
    # composition nobody has established. That reduces confidence; it does
    # not invalidate the model.
    if (reported_growth is not None
            and reported_growth > MATERIAL_GROWTH_FOR_COMPOSITION
            and bridge.coverage_status != GrowthCoverage.FULL
            and not revenue_guidance_present):
        bridge.findings.append({
            "code": HEADLINE_GROWTH_COMPOSITION_UNCERTAIN,
            "severity": "info",
            "message": (
                f"Reported revenue growth of {reported_growth:.1%} has not been decomposed: "
                f"the issuer's filings state {'no' if not measured else 'only part of the'} "
                "numeric attribution, and it has published no revenue guidance. How much of "
                "this rate reflects volume, pricing, acquisitions or currency is therefore "
                "unestablished, and its persistence cannot be assumed."),
            "reported_growth": reported_growth,
            "coverage_status": bridge.coverage_status,
        })
    return bridge


def describe_for_assumptions(bridge: GrowthBridge) -> dict:
    """Sections 20-26: what the assumption builder is given.

    Deliberately a DECOMPOSITION and not a single "organic growth" number.
    Section 25 is explicit that the deterministic layer must not invent one
    when the evidence is incomplete; it supplies the parts and validates what
    comes back.
    """
    return {
        "reported_growth": bridge.reported_growth,
        "period": bridge.period,
        "coverage_status": bridge.coverage_status,
        "quality": bridge.quality,
        "contributions": {c.type: c.contribution for c in bridge.measured},
        "qualitative_drivers": [c.type for c in bridge.components if c.qualitative_only],
        "unexplained_component": bridge.unexplained_component,
        "evidence_ids": sorted({eid for c in bridge.components for eid in c.evidence_ids}),
        "caution": (
            "Reported growth is not automatically organic or persistent. Where a "
            "contribution is stated it is given above; what is unexplained is stated as "
            "such. Do not treat the headline rate as a volume trend without evidence."),
    }
