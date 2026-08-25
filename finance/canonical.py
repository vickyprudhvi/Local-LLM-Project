"""Phase H.9 — one canonical set of CURRENT metrics, in its own namespace.

THE BUG THIS EXISTS TO FIX (live analyses, 2026-08-19)
======================================================
Two companies, same shape. The report claimed one current trailing-twelve-
month financial base and then handed the research agents last fiscal year's
numbers:

    metric                     current TTM      what agents received
    ------------------------   -------------    --------------------
    operating margin           15.71%           10.66%   (FY2025)
    free cash flow             $8.40B           $6.74B   (FY2025)
    net margin                 15.58%           12.51%   (FY2025)

and on the other:

    operating margin           26.76%           None     (FY2025)
    free cash flow             $22.24B          $19.70B  (FY2025)

`fundamental_metrics` is computed from the ANNUAL statements. The freshness
planner separately builds validated TTM values. Both were in `facts`, both
were called things like `operating_margin` and `free_cash_flow`, and the
Snapshot renderer and every research role read whichever one their code path
happened to reach. Nothing was wrong with either number; what was wrong is
that they share a name.

WHAT THIS MODULE DOES
=====================
It publishes the current metrics under an explicitly CURRENT namespace --
`current.ttm.operating_margin` -- alongside the historical ones under
`historical.fy.operating_margin`, so a consumer that wants the current figure
asks for it and a consumer that reaches for a historical one cannot get it by
accident. Every field carries its period, definition, basis, source and
evidence id, so "which number is this?" is answerable from the value itself.

Derived margins (section 8) are computed HERE from the validated TTM
components rather than read from an annual statement, and only when both
components cover the same window -- a margin built from a TTM numerator and
an annual denominator belongs to no period at all.

Nothing here is issuer-specific.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from finance import growth as growth_module

CANONICAL_CURRENT_EVIDENCE_CONFLICT = "CANONICAL_CURRENT_EVIDENCE_CONFLICT"
TTM_BASE_PERIOD_MISMATCH = "TTM_BASE_PERIOD_MISMATCH"

# Phase H.11, section 39.
DERIVED_METRIC_STALE_SOURCE = "DERIVED_METRIC_STALE_SOURCE"
DERIVED_RATIO_PERIOD_MISMATCH = "DERIVED_RATIO_PERIOD_MISMATCH"
CURRENT_GROWTH_USED_HISTORICAL_PERIOD = "CURRENT_GROWTH_USED_HISTORICAL_PERIOD"

# How far two supposedly-canonical values may differ before the difference is
# a conflict rather than rounding.
CONSISTENCY_TOLERANCE = 0.005


class PeriodKind:
    TTM = "TTM"
    ANNUAL = "ANNUAL"
    INTERIM = "INTERIM"
    INSTANT = "INSTANT"
    DERIVED = "DERIVED"
    ALL = (TTM, ANNUAL, INTERIM, INSTANT, DERIVED)


@dataclass(frozen=True)
class CanonicalMetric:
    """One metric with everything needed to say WHICH metric it is."""

    key: str
    value: Optional[float]
    period: Optional[str] = None
    period_type: str = PeriodKind.TTM
    definition: str = ""
    accounting_basis: str = "reported_GAAP"
    source: str = "sec"
    evidence_id: str = ""
    freshness_status: Optional[str] = None
    validation_status: Optional[str] = None
    # Phase H.11, sections 1-2 and 37. A derived figure is only as current as
    # the facts underneath it, and a bare float cannot say what those were.
    # `derivation_formula` states how the value was produced and
    # `source_metrics` names the canonical inputs, which together form the
    # lightweight derivation graph section 37 asks for: current_ratio ->
    # current_assets@date, current_liabilities@date.
    derivation_formula: str = ""
    source_metrics: tuple = ()
    source_periods: tuple = ()

    def to_dict(self) -> dict:
        return {
            "key": self.key, "value": self.value, "period": self.period,
            "period_type": self.period_type, "definition": self.definition,
            "accounting_basis": self.accounting_basis, "source": self.source,
            "evidence_id": self.evidence_id,
            "freshness_status": self.freshness_status,
            "validation_status": self.validation_status,
            "derivation_formula": self.derivation_formula,
            "source_metrics": list(self.source_metrics),
            "source_periods": list(self.source_periods),
        }


@dataclass
class CanonicalFinancialEvidence:
    """The single current/historical evidence packet every consumer reads."""

    current: Dict[str, CanonicalMetric] = field(default_factory=dict)
    historical: Dict[str, CanonicalMetric] = field(default_factory=dict)
    # Which growth measure filled the current slot, and how to name it.
    current_growth_kind: Optional[str] = None
    current_growth_label: Optional[str] = None
    base_period: Optional[str] = None
    base_period_aligned: bool = True
    findings: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def value(self, key: str) -> Optional[float]:
        metric = self.current.get(key)
        return metric.value if metric else None

    def to_dict(self) -> dict:
        return {
            "current": {k: v.to_dict() for k, v in self.current.items()},
            "historical": {k: v.to_dict() for k, v in self.historical.items()},
            "base_period": self.base_period,
            "base_period_aligned": self.base_period_aligned,
            "current_growth_kind": self.current_growth_kind,
            "current_growth_label": self.current_growth_label,
            "findings": [dict(f) for f in self.findings],
            "warnings": list(self.warnings),
        }

    def research_packet(self) -> dict:
        """Section 4's separated packet, with no ambiguous generic names.

        Keys are fully qualified (`current.ttm.operating_margin`), which is
        what stops a research role from reaching for a historical annual
        figure and describing it as the company's current condition.
        """
        return {
            "current": {f"current.{m.period_type.lower()}.{k}": m.to_dict()
                        for k, m in self.current.items()},
            "historical": {f"historical.{m.period}.{k}": m.to_dict()
                           for k, m in self.historical.items()},
        }


def _finding(code: str, severity: str, message: str, **detail) -> dict:
    entry = {"code": code, "severity": severity, "message": message}
    entry.update(detail)
    return entry


# Flow metrics that must share one end date to be presented as one base
# (section 7).
_BASE_PERIOD_METRICS = ("revenue", "operating_income", "net_income",
                        "operating_cash_flow", "free_cash_flow",
                        # Depreciation feeds the DCF directly, so an annual
                        # D&A sitting inside a trailing-twelve-month base is
                        # the same mismatch as any other -- and on a live
                        # issuer it was the only metric that could not be
                        # built on a TTM basis while everything around it
                        # could.
                        "depreciation_and_amortization")

# (canonical key, numerator flow, denominator flow, definition) for the
# margins section 8 requires to be DERIVED from current components rather
# than read from a stale annual statement.
# (canonical key, numerator, denominator, definition) for the balance-sheet
# ratios of sections 7-9. Every one is computed from components carrying the
# SAME instant date; a numerator from one date over a denominator from
# another is not a ratio of anything.
#
# The live failure: a report stated a balance-sheet date of the latest
# quarter and a current ratio of 5.92 taken from the prior fiscal year, while
# that quarter's own current assets and liabilities -- both present, both
# already selected -- gave 4.78. The annual ratio was not wrong; it was an
# answer to a question nobody asked.
_DERIVED_INSTANT_RATIOS = (
    ("current_ratio", "current_assets", "current_liabilities",
     "current assets divided by current liabilities at the same balance-sheet date"),
    ("debt_to_equity", "total_debt", "stockholders_equity",
     "total debt divided by shareholders' equity at the same balance-sheet date"),
    ("net_debt_to_equity", "net_debt", "stockholders_equity",
     "net debt divided by shareholders' equity at the same balance-sheet date"),
    ("cash_to_debt", "cash_and_cash_equivalents", "total_debt",
     "cash and equivalents divided by total debt at the same balance-sheet date"),
)

# A denominator at or below this magnitude makes the ratio meaningless rather
# than large -- the existing `not_meaningful` treatment for zero or negative
# equity, applied at the point the ratio is derived instead of after.
_RATIO_DENOMINATOR_FLOOR = 0.0

_DERIVED_MARGINS = (
    ("operating_margin", "operating_income", "revenue",
     "operating income divided by revenue, both over the same trailing window"),
    ("net_margin", "net_income", "revenue",
     "net income divided by revenue, both over the same trailing window"),
    ("free_cash_flow_margin", "free_cash_flow", "revenue",
     "free cash flow divided by revenue, both over the same trailing window"),
    ("operating_cash_flow_margin", "operating_cash_flow", "revenue",
     "operating cash flow divided by revenue, both over the same trailing window"),
)


def build_canonical_evidence(state, historical_metrics: Optional[dict] = None,
                             fcf_definition: str = "simple_fcf",
                             growth_set=None) -> CanonicalFinancialEvidence:
    """Assemble the canonical packet from an already-validated state.

    Reads only `CurrentFinancialState`, which has already decided which
    period every input comes from. This module adds no new selection policy;
    it adds a NAMESPACE and the derived ratios, so that the decision the
    freshness planner already made survives all the way to the report and the
    research roles.
    """
    evidence = CanonicalFinancialEvidence()
    flows = getattr(state, "flows", {}) or {}
    balance = getattr(state, "balance_sheet", {}) or {}

    # -- current flows -----------------------------------------------------
    end_dates = {}
    for name, selection in flows.items():
        value = getattr(selection, "value", None)
        if value is None:
            continue
        record = getattr(selection, "ttm", None) or {}
        period_type = (PeriodKind.TTM if getattr(selection, "source", "") == "ttm_calculation"
                       else PeriodKind.ANNUAL)
        evidence.current[name] = CanonicalMetric(
            key=name, value=value,
            period=(f"{getattr(selection, 'period_start', None)}.."
                    f"{getattr(selection, 'as_of_date', None)}"),
            period_type=period_type,
            definition=(fcf_definition if name == "free_cash_flow"
                        else record.get("construction_method") or "as reported"),
            source=getattr(selection, "provider", "sec"),
            evidence_id=f"current.{period_type.lower()}.{name}",
            freshness_status=getattr(selection, "freshness_status", None),
            validation_status=record.get("validation_status"))
        if name in _BASE_PERIOD_METRICS and period_type == PeriodKind.TTM:
            end_dates[name] = getattr(selection, "as_of_date", None)

    # -- section 7: do these metrics actually share one base period? -------
    distinct = {end for end in end_dates.values() if end}
    evidence.base_period = max(distinct) if distinct else None
    if len(distinct) > 1:
        evidence.base_period_aligned = False
        lagging = sorted(name for name, end in end_dates.items()
                         if end and end != evidence.base_period)
        evidence.findings.append(_finding(
            TTM_BASE_PERIOD_MISMATCH, "warning",
            f"The metrics presented as one trailing-twelve-month base do not share an end "
            f"date: {', '.join(lagging)} end earlier than {evidence.base_period}. Each is a "
            "valid twelve months; together they are not one financial base.",
            base_period=evidence.base_period, lagging=lagging))

    # A flow that is NOT on a TTM basis while the base is must be named, or
    # the whole snapshot silently inherits a label one metric does not meet.
    annual_in_ttm_base = sorted(
        name for name, metric in evidence.current.items()
        if name in _BASE_PERIOD_METRICS and metric.period_type == PeriodKind.ANNUAL)
    if annual_in_ttm_base and evidence.base_period:
        evidence.base_period_aligned = False
        evidence.findings.append(_finding(
            TTM_BASE_PERIOD_MISMATCH, "warning",
            f"{', '.join(annual_in_ttm_base)} could only be built on an ANNUAL basis while "
            "the rest of the base is a trailing twelve months; the snapshot is not uniformly "
            "TTM and each metric carries its own period.",
            annual_metrics=annual_in_ttm_base))

    # -- section 8: derived current margins --------------------------------
    for key, numerator, denominator, definition in _DERIVED_MARGINS:
        top, bottom = evidence.current.get(numerator), evidence.current.get(denominator)
        if not top or not bottom or not bottom.value:
            continue
        if top.period != bottom.period:
            evidence.warnings.append(
                f"{key} was not derived: {numerator} covers {top.period} and {denominator} "
                f"covers {bottom.period}, so the ratio would belong to neither period.")
            continue
        evidence.current[key] = CanonicalMetric(
            key=key, value=top.value / bottom.value, period=top.period,
            period_type=PeriodKind.DERIVED, definition=definition,
            accounting_basis=top.accounting_basis, source=top.source,
            evidence_id=f"current.derived.{key}",
            validation_status=top.validation_status,
            derivation_formula=f"{numerator} / {denominator}",
            source_metrics=(numerator, denominator),
            source_periods=(top.period, bottom.period))

    # -- current balance-sheet points --------------------------------------
    for name in ("cash_and_cash_equivalents", "short_term_investments", "long_term_debt",
                 "short_term_debt", "stockholders_equity", "current_assets",
                 "current_liabilities"):
        selection = balance.get(name)
        value = getattr(selection, "value", None)
        if value is None:
            continue
        evidence.current[name] = CanonicalMetric(
            key=name, value=value, period=getattr(selection, "as_of_date", None),
            period_type=PeriodKind.INSTANT, definition="as reported on the balance sheet",
            source=getattr(selection, "provider", "sec"),
            evidence_id=f"current.instant.{name}",
            freshness_status=getattr(selection, "freshness_status", None))

    total_debt = getattr(state, "total_debt", None)
    if getattr(total_debt, "value", None) is not None:
        evidence.current["total_debt"] = CanonicalMetric(
            key="total_debt", value=total_debt.value,
            period=getattr(total_debt, "as_of_date", None), period_type=PeriodKind.INSTANT,
            definition="sum of reported debt components", evidence_id="current.instant.total_debt")
    if getattr(state, "net_debt", None) is not None:
        policy = ((getattr(state, "net_debt_detail", None) or {}).get("net_debt_policy")
                  or "cash_only")
        evidence.current["net_debt"] = CanonicalMetric(
            key="net_debt", value=state.net_debt,
            period=getattr(state, "financial_as_of", None), period_type=PeriodKind.INSTANT,
            definition=f"net debt under the {policy!r} policy",
            evidence_id="current.instant.net_debt")

    # -- sections 7-9: ratios derived from ONE balance-sheet date ----------
    #
    # Computed here, from the same current components every other current
    # figure comes from, so the Snapshot has a current ratio to show and
    # never has to fall back on the annual one. A component pair that does
    # not share an instant is refused rather than divided.
    for key, numerator, denominator, definition in _DERIVED_INSTANT_RATIOS:
        top = evidence.current.get(numerator)
        bottom = evidence.current.get(denominator)
        if top is None or bottom is None or top.value is None or bottom.value is None:
            continue
        if top.period != bottom.period:
            evidence.findings.append(_finding(
                DERIVED_RATIO_PERIOD_MISMATCH, "warning",
                f"{key} was not derived: {numerator} is measured at {top.period} and "
                f"{denominator} at {bottom.period}. A numerator from one date over a "
                "denominator from another is not a ratio of anything.",
                metric=key, numerator_period=top.period, denominator_period=bottom.period))
            continue
        if bottom.value <= _RATIO_DENOMINATOR_FLOOR:
            # Zero or negative equity, or no debt at all. The existing
            # `not_meaningful` treatment applies; deriving a number here
            # would only give a later stage something false to quote.
            continue
        evidence.current[key] = CanonicalMetric(
            key=key, value=top.value / bottom.value, period=top.period,
            period_type=PeriodKind.DERIVED, definition=definition,
            accounting_basis=top.accounting_basis, source=top.source,
            evidence_id=f"current.instant.{key}",
            freshness_status=top.freshness_status,
            derivation_formula=f"{numerator} / {denominator}",
            source_metrics=(numerator, denominator),
            source_periods=(top.period, bottom.period))

    # -- sections 3-6: the growth family, each member under its own id -----
    #
    # `revenue_growth` as a single name is what let last fiscal year's rate
    # be displayed under a heading that said trailing twelve months. The
    # members are published separately and the CURRENT one is marked, so a
    # consumer asking for current growth cannot receive a historical rate by
    # default.
    if growth_set is not None:
        for kind, growth in (growth_set.metrics or {}).items():
            if growth.value is None:
                continue
            historical = growth.evidence_id.startswith("historical.")
            metric = CanonicalMetric(
                key=f"{growth.metric}_growth_{kind.lower()}", value=growth.value,
                period=growth.current_period,
                period_type=PeriodKind.DERIVED, definition=growth.definition,
                evidence_id=growth.evidence_id,
                validation_status=growth.validation_status,
                derivation_formula=(f"{growth.metric}({growth.current_period}) / "
                                    f"{growth.metric}({growth.comparison_period}) - 1"),
                source_metrics=(growth.metric,),
                source_periods=tuple(x for x in (growth.current_period,
                                                 growth.comparison_period) if x))
            bucket = evidence.historical if historical else evidence.current
            bucket[metric.key] = metric

        current_growth = growth_set.current
        if current_growth is not None and current_growth.value is not None:
            # The one a Snapshot should show, carrying the label that says
            # which period it covers.
            evidence.current["revenue_growth"] = CanonicalMetric(
                key="revenue_growth", value=current_growth.value,
                period=current_growth.current_period, period_type=PeriodKind.DERIVED,
                definition=f"{current_growth.definition} ({current_growth.label})",
                evidence_id=current_growth.evidence_id,
                validation_status=current_growth.validation_status,
                derivation_formula=(f"{current_growth.metric}"
                                    f"({current_growth.current_period}) / "
                                    f"{current_growth.metric}"
                                    f"({current_growth.comparison_period}) - 1"),
                source_metrics=(current_growth.metric,),
                source_periods=tuple(x for x in (current_growth.current_period,
                                                 current_growth.comparison_period) if x))
            evidence.current_growth_kind = current_growth.kind
            evidence.current_growth_label = current_growth.label
            if current_growth.kind == growth_module.GrowthKind.FY_YOY:
                evidence.findings.append(_finding(
                    CURRENT_GROWTH_USED_HISTORICAL_PERIOD, "info",
                    "No trailing-twelve-month, year-to-date or quarterly growth rate could be "
                    "built for this issuer, so the current growth figure is the last fiscal "
                    "year's change. It is labelled as such wherever it appears.",
                    kind=current_growth.kind))

    # -- historical, in its OWN namespace ----------------------------------
    for name, entry in (historical_metrics or {}).items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        if value is None:
            continue
        periods = (entry.get("inputs") if isinstance(entry, dict) else None) or []
        evidence.historical[name] = CanonicalMetric(
            key=name, value=value,
            period=(str(periods[0]) if periods else "prior_periods"),
            period_type=PeriodKind.ANNUAL,
            definition=(entry.get("formula") if isinstance(entry, dict) else "") or "",
            accounting_basis=(entry.get("accounting_basis")
                              if isinstance(entry, dict) else None) or "reported_GAAP",
            evidence_id=f"historical.{(periods[0] if periods else 'prior')}.{name}")
    return evidence


def validate_section_consistency(sections: Dict[str, Dict[str, float]],
                                 evidence: CanonicalFinancialEvidence) -> List[dict]:
    """Section 5: every section must quote the canonical current value.

    `sections` maps a section name to the values it renders. A section may
    legitimately use a different figure -- a different FCF definition, a
    prior-year comparison -- but only when it LABELS it, which in this
    structure means not claiming the canonical key.
    """
    findings: List[dict] = []
    for section, values in sections.items():
        for key, value in (values or {}).items():
            canonical = evidence.current.get(key)
            if canonical is None or canonical.value is None or value is None:
                continue
            denominator = abs(canonical.value) or 1.0
            if abs(value - canonical.value) / denominator <= CONSISTENCY_TOLERANCE:
                continue
            findings.append(_finding(
                CANONICAL_CURRENT_EVIDENCE_CONFLICT, "warning",
                f"The {section} section reports {key} as {value:,.4g} while the canonical "
                f"current value for {canonical.period} is {canonical.value:,.4g}. A section "
                "may use a different figure only when it states the different definition or "
                "period it is on.",
                section=section, metric=key, section_value=value,
                canonical_value=canonical.value, canonical_period=canonical.period))
    return findings


STALE_METRIC_CITATION = "STALE_METRIC_CITATION"

# Below this the two periods agree closely enough that quoting either tells
# a reader the same thing, and forcing a re-citation would be pedantry.
STALE_CITATION_TOLERANCE = 0.02


def conflicting_historical_citations(cited_ids, index) -> list:
    """Section 5: which cited `fundamental.<name>` ids have a current twin?

    The live failure this catches: a Snapshot showing free cash flow for the
    trailing twelve months and a Bull Case citing `fundamental.free_cash_flow`
    -- last fiscal year's -- in the same report. Both numbers are correct and
    both are cited; a reader cannot reconcile them, and nothing in the older
    index distinguished them.

    Only names present in BOTH buckets are considered, so a genuinely
    historical metric with no current counterpart (a five-year CAGR, a
    year-over-year growth rate) is never touched. A name whose two values
    agree within `STALE_CITATION_TOLERANCE` is not a conflict either.

    Returns [(name, historical_value, current_value), ...].
    """
    conflicts = []
    for cited in cited_ids or []:
        if not isinstance(cited, str) or not cited.startswith("fundamental."):
            continue
        name = cited[len("fundamental."):]
        historical = index.get(cited)
        current = index.get(f"current.{name}")
        if historical is None or current is None:
            continue
        try:
            old_value = float(historical.value)
            new_value = float(current.value)
        except (TypeError, ValueError):
            continue
        scale = max(abs(old_value), abs(new_value))
        if scale == 0 or abs(old_value - new_value) / scale <= STALE_CITATION_TOLERANCE:
            continue
        conflicts.append((name, old_value, new_value))
    return conflicts
