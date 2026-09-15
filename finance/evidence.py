"""Phase H.2 — the evidence-ID index behind the staged research pipeline.

Every citable fact in a `build_compact_synthesis_payload()` snapshot gets a
stable, deterministic ID (e.g. `fundamental.roe_ending_equity`,
`dcf.value_per_share.bull`, `valuation_gap.direction`). Every research-pipeline
stage (`finance/research_pipeline.py`) is given the rendered index as its ONLY
source of facts and is required to cite IDs from it for every claim.

This is the structural enforcement behind "never invent provider facts" and
"cite internal evidence IDs": `validate_evidence_citations` rejects any ID a
stage's output cites that is not actually in the index, which is a stronger
guarantee than a prompt instruction alone — a model that hallucinates a
citation fails validation and the stage is marked FAILED (fail closed), never
silently accepted.

The index and its rendered string form are built ONCE per analysis and never
mutated — every stage receives the same immutable JSON string (see
`render_evidence_index`), not a live mutable object it could alter for a later
stage to see differently.
"""

import json
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from finance.dcf import DcfValidationStatus
from finance.dcf_packet import (
    STATUS_EXPLANATION,
    VALUATION_DERIVED_CONCLUSIONS,
    ValuationStatus,
    build_valuation_research_evidence,
    classify_valuation,
)

# Curated subsets — deliberately not "every field in the payload": an evidence
# ID should name something a research stage would actually reason about, not
# every internal bookkeeping value (e.g. schema_version, calculation_version
# strings are excluded; they aren't claims to cite).
_COMPANY_FIELDS = ("name", "sector", "industry", "market_capitalisation",
                  "pe_ratio", "forward_pe", "price_to_book", "beta",
                  "dividend_yield", "shares_outstanding")
_QUOTE_FIELDS = ("price", "price_basis", "change_percent", "latest_trading_day",
                 "previous_close")
# H.4 corrective patch: "estimated_intrinsic_value_per_share" was renamed to
# "estimated_base_modeled_value_per_share" (finance/workflow.py::
# _valuation_gap) so the word "intrinsic" never appears in evidence text a
# research stage reads and might echo back — DCF scenario outputs are
# "modeled values," never an "intrinsic value" (see finance/claim_validation.py's
# DCF-terminology scan for the deterministic backstop on the same rule).
# DIS valuation-comparison patch: `market_price_premium_pct` (denominated in
# the MODELED value) and `modeled_return_to_value_pct` (denominated in the
# market PRICE) are two DIFFERENT numbers -- both indexed under their own,
# explicit, non-interchangeable evidence IDs. `difference_pct` (the legacy
# name, numerically identical to `modeled_return_to_value_pct`) stays
# indexed too, for any existing citation of it, but a research stage should
# prefer the explicit names for anything new.
_VALUATION_GAP_FIELDS = ("market_price", "market_price_basis",
                         "estimated_base_modeled_value_per_share", "difference",
                         "difference_pct", "market_price_premium_pct",
                         "modeled_return_to_value_pct", "direction")
_SCENARIO_SPREAD_FIELDS = ("bull_value_per_share", "base_value_per_share",
                           "bear_value_per_share", "spread", "spread_pct_of_base")
_DCF_SCENARIO_FIELDS = ("value_per_share", "enterprise_value", "equity_value",
                        "net_debt", "terminal_value_share_of_enterprise_value")
_DCF_TOP_FIELDS = ("net_debt", "net_debt_policy", "value_per_share")

# H.4 corrective patch (Problem 3) — every material DCF assumption gets its
# own stable evidence ID, not just the resulting scenario value, so a
# research stage can cite and explain WHY base/bull/bear differ (e.g. "the
# bull scenario uses a higher revenue_growth and a lower wacc") rather than
# only citing the modeled value each produces. revenue_growth/operating_
# margin/wacc/terminal_growth genuinely vary per scenario; capex_pct_revenue/
# depreciation_pct_revenue/working_capital_pct_revenue/tax_rate are IDENTICAL
# across every scenario by construction (reported-history-first, never a
# scenario's own growth/margin delta — see docs/PHASE_H1_STOCK_ANALYSIS.md's
# "Assumption provenance" section), so those four get ONE shared evidence ID
# each rather than one per scenario.
_DCF_PER_SCENARIO_ASSUMPTION_FIELDS = ("revenue_growth", "operating_margin",
                                       "wacc", "terminal_growth")
_DCF_SHARED_ASSUMPTION_FIELDS = ("capex_pct_revenue", "depreciation_pct_revenue",
                                 "working_capital_pct_revenue", "tax_rate")
_DCF_ASSUMPTION_UNITS = {
    "revenue_growth": "ratio", "operating_margin": "ratio", "wacc": "ratio",
    "terminal_growth": "ratio", "capex_pct_revenue": "ratio",
    "depreciation_pct_revenue": "ratio", "working_capital_pct_revenue": "ratio",
    "tax_rate": "ratio",
}


@dataclass(frozen=True)
class EvidenceItem:
    """One citable fact.

    `evidence_type`/`units`/`scenario`/`source_type`/`source_periods`/
    `source_evidence_ids`/`derivation`/`approval_status`/`calculation_version`
    are OPTIONAL, richer provenance fields — populated for DCF assumption
    items (`dcf.assumption.*`) so a research stage can reason about WHY a
    value is what it is, not just cite the bare number; every other evidence
    item leaves them unset, exactly as before this field set was added.
    """

    evidence_id: str
    label: str
    value: object
    evidence_type: Optional[str] = None
    units: Optional[str] = None
    scenario: Optional[str] = None
    source_type: Optional[str] = None
    source_periods: Optional[List[str]] = None
    source_evidence_ids: Optional[List[str]] = None
    derivation: Optional[str] = None
    approval_status: Optional[str] = None
    calculation_version: Optional[str] = None

    def to_dict(self) -> dict:
        out = {"id": self.evidence_id, "label": self.label, "value": self.value}
        for key, val in (
            ("type", self.evidence_type), ("units", self.units), ("scenario", self.scenario),
            ("source_type", self.source_type), ("source_periods", self.source_periods),
            ("source_evidence_ids", self.source_evidence_ids), ("derivation", self.derivation),
            ("approval_status", self.approval_status),
            ("calculation_version", self.calculation_version),
        ):
            if val is not None:
                out[key] = val
        return out


def _add(index: Dict[str, EvidenceItem], evidence_id: str, label: str, value):
    if value is None:
        return  # nothing to cite when the underlying fact is absent
    index[evidence_id] = EvidenceItem(evidence_id, label, value)


def _add_dcf_assumption(index: Dict[str, EvidenceItem], evidence_id: str, label: str, value,
                        *, units, scenario, provenance_entry, calculation_version):
    """One `dcf.assumption.*` / `dcf.terminal_value_share.*` evidence item —
    carries the assumption-provenance detail (source_type, source_periods,
    derivation, approval_status) inline, not just the bare number, so
    `render_evidence_index` can show a research stage WHY the value is what
    it is (Problem 3)."""
    if value is None:
        return
    provenance_entry = provenance_entry or {}
    index[evidence_id] = EvidenceItem(
        evidence_id=evidence_id, label=label, value=value,
        evidence_type="dcf_assumption",
        units=units or provenance_entry.get("units"),
        scenario=scenario,
        source_type=provenance_entry.get("source_type"),
        source_periods=list(provenance_entry.get("source_periods") or []) or None,
        source_evidence_ids=list(provenance_entry.get("source_evidence_ids") or []) or None,
        derivation=provenance_entry.get("derivation"),
        approval_status=provenance_entry.get("approval_status"),
        calculation_version=calculation_version,
    )


def _assumption_provenance_for(scenario_dict, shared_provenance, field_name):
    """This scenario's OWN assumption_provenance entry for `field_name` when
    present, else the hoisted `dcf.shared_assumption_provenance` block
    (finance/workflow.py::_hoist_shared_assumption_provenance de-duplicates a
    provenance entry that is byte-identical across every scenario into that
    shared block and removes it from each scenario's own dict — see Phase
    H.3 corrective patch, Problem 11)."""
    assumptions = (scenario_dict or {}).get("assumptions") or {}
    own = (assumptions.get("assumption_provenance") or {}).get(field_name)
    return own or (shared_provenance or {}).get(field_name)


def _assumption_value_for(scenario_dict, field_name, provenance_entry):
    """The single representative number for one assumption field: the
    provenance entry's own 'value' (what propose_assumptions actually
    proposed) when available, else the scenario's own field — which may be a
    per-year array (finance/dcf.py expands a scalar assumption to one entry
    per forecast year), in which case the first year stands in as the
    citable scalar. The FULL per-year array is never lost — it stays intact
    in dcf.scenarios[].assumptions and dcf.scenarios[].forecast, both of
    which survive compaction whole; this is an additional, smaller pointer,
    not a replacement."""
    if provenance_entry and provenance_entry.get("value") is not None:
        return provenance_entry["value"]
    assumptions = (scenario_dict or {}).get("assumptions") or {}
    raw = assumptions.get(field_name)
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw


def _add_dcf_assumption_evidence(index: Dict[str, EvidenceItem], dcf: dict) -> None:
    """H.4 corrective patch (Problem 3): every material DCF assumption gets
    its own stable evidence ID, carrying WHY it has the value it does
    (source_type/derivation/source_periods), not just the resulting scenario
    value — so a research stage can explain that "the bull scenario uses a
    higher revenue_growth and a lower wacc than base," not merely that the
    bull modeled value is higher. Also exposes dcf.terminal_value_share.*
    (a shorter alias of the existing dcf.terminal_value_share_of_enterprise_
    value.* id already produced by the generic _DCF_SCENARIO_FIELDS loop
    above — both are indexed; nothing is removed)."""
    calculation_version = dcf.get("calculation_version")
    scenarios = dcf.get("scenarios") or []
    shared_provenance = dcf.get("shared_assumption_provenance") or {}

    for scenario_dict in scenarios:
        name = scenario_dict.get("scenario")
        if not name:
            continue
        for field_name in _DCF_PER_SCENARIO_ASSUMPTION_FIELDS:
            provenance_entry = _assumption_provenance_for(scenario_dict, shared_provenance, field_name)
            value = _assumption_value_for(scenario_dict, field_name, provenance_entry)
            _add_dcf_assumption(
                index, f"dcf.assumption.{field_name}.{name}",
                f"DCF assumption: {field_name.replace('_', ' ')} ({name} scenario)", value,
                units=_DCF_ASSUMPTION_UNITS.get(field_name), scenario=name,
                provenance_entry=provenance_entry, calculation_version=calculation_version)
        _add_dcf_assumption(
            index, f"dcf.terminal_value_share.{name}",
            f"DCF terminal value share of enterprise value ({name} scenario)",
            scenario_dict.get("terminal_value_share_of_enterprise_value"),
            units="ratio", scenario=name, provenance_entry=None,
            calculation_version=calculation_version)

    primary_name = dcf.get("primary_scenario")
    primary_scenario = next((s for s in scenarios if s.get("scenario") == primary_name),
                            scenarios[0] if scenarios else None)
    for field_name in _DCF_SHARED_ASSUMPTION_FIELDS:
        provenance_entry = _assumption_provenance_for(primary_scenario, shared_provenance, field_name)
        value = _assumption_value_for(primary_scenario, field_name, provenance_entry)
        _add_dcf_assumption(
            index, f"dcf.assumption.{field_name}",
            f"DCF assumption: {field_name.replace('_', ' ')} (shared across scenarios)", value,
            units=_DCF_ASSUMPTION_UNITS.get(field_name), scenario="shared",
            provenance_entry=provenance_entry, calculation_version=calculation_version)



# ---------------------------------------------------------------------------
# Parts 4-5 - the valuation evidence gate
# ---------------------------------------------------------------------------
#
# Before this, the index read the DCF result directly and decided for itself,
# in three separate places and on three different criteria, whether each
# valuation figure was safe to expose: the scenario values consulted
# `validation_status`, the gap consulted `dcf_suitability` and the share
# reconciliation, and the scenario spread consulted nothing at all. Three
# gates disagreeing about one question is the same as no gate.
#
# There is now one status for the run and one function that decides what it
# licenses, and every numeric valuation id in this index is behind it.


# A status under which the DIRECTION of the price-versus-value comparison
# survives while its magnitude does not. LIMITED means the valuation is
# usable with substantial caveats -- "the price is above the modelled value"
# is a statement those inputs still support, and a percentage rendered to one
# decimal place is not. Every other non-VALID status withholds the direction
# too, because under those the modelled value is not established at all.
_DIRECTION_SURVIVES = (ValuationStatus.LIMITED,)

# A share count that does not reconcile against the reported market
# capitalisation puts the per-share numerator and the market-price
# denominator on different bases. The valuation may be arithmetically
# perfect and still not comparable with a price.
_UNRESOLVED_SHARE_BASIS = ("MATERIAL_DIFFERENCE", "INCOMPATIBLE_BASIS")


def valuation_evidence_status(compact_payload: dict) -> str:
    """The ONE valuation status this index gates on.

    Prefers the status the analysis itself decided (`valuation_status`, set
    by finance/workflow.py once, from the packet failure, the model's own
    validation and the business model). A payload without that field -- a
    replayed artifact, or a hand-built fixture -- is classified here from the
    same inputs rather than being assumed valid.
    """
    dcf = compact_payload.get("dcf") or {}
    suitability_record = compact_payload.get("dcf_suitability") or {}
    # An unassessed suitability is not a limited one -- see
    # finance/workflow.py::_valuation_status.
    suitability = (suitability_record.get("dcf_suitability")
                   if suitability_record.get("assessed", True) else None)
    share_basis = ((compact_payload.get("dcf_financial_basis") or {})
                   .get("share_reconciliation") or {})

    status = compact_payload.get("valuation_status")
    if status not in ValuationStatus.ALL:
        status = classify_valuation(
            packet_failure=compact_payload.get("dcf_packet_failure"),
            # A payload that carries a usable comparison carries a valuation,
            # whichever key it arrived under.
            dcf_available=bool(dcf.get("available")
                               or (compact_payload.get("valuation_gap") or {}).get("available")),
            dcf_validation_status=dcf.get("validation_status"),
            suitability_status=suitability)

    # Applied after classification rather than inside it: an unresolved share
    # basis does not make the enterprise valuation wrong, it makes the
    # per-share comparison incomparable, and LIMITED is exactly that claim.
    if status == ValuationStatus.VALID_FOR_RESEARCH \
            and share_basis.get("status") in _UNRESOLVED_SHARE_BASIS:
        return ValuationStatus.LIMITED
    return status


def _valuation_root_cause(compact_payload: dict, status: str):
    """The specific reason behind the status, when one was recorded."""
    failure = compact_payload.get("dcf_packet_failure") or {}
    reasons = failure.get("reasons") or []
    if reasons:
        return reasons[0]
    dcf = compact_payload.get("dcf") or {}
    if dcf.get("validation_reasons"):
        return "; ".join(dcf["validation_reasons"])
    return dcf.get("reason") or STATUS_EXPLANATION.get(status)


def _add_valuation_status_evidence(index, compact_payload, status, gate) -> None:
    """The status vocabulary is ALWAYS citable; the numbers are not.

    A role that cannot see why a valuation is absent will explain the absence
    itself, and the explanations it invents are worse than the true one --
    "the model failed" for a business the model never applied to.
    """
    _add(index, "valuation.status", "Valuation status for research use", status)
    if gate.get("explanation"):
        _add(index, "valuation.explanation", "Why the valuation is not fully usable",
             gate["explanation"])
    if gate.get("root_cause"):
        _add(index, "valuation.root_cause", "The specific cause behind the valuation status",
             gate["root_cause"])
    if status != ValuationStatus.VALID_FOR_RESEARCH:
        _add(index, "valuation.conclusions_withheld",
             "Valuation conclusions that may NOT be stated for this analysis",
             ", ".join(VALUATION_DERIVED_CONCLUSIONS))


def build_evidence_index(compact_payload: dict) -> Dict[str, EvidenceItem]:
    """Flatten a compact synthesis payload into ID -> EvidenceItem.

    Deterministic: the same payload always produces the same IDs. Absent
    facts (value is None) are simply not indexed — a stage can never cite
    an ID for a fact that was never reported, because the ID does not exist.
    """
    index: Dict[str, EvidenceItem] = {}
    symbol = compact_payload.get("symbol")
    _add(index, "meta.symbol", "Ticker", symbol)
    _add(index, "meta.analysis_mode", "Analysis completeness", compact_payload.get("analysis_mode"))

    company = compact_payload.get("company") or {}
    for field in _COMPANY_FIELDS:
        _add(index, f"company.{field}", f"Company {field.replace('_', ' ')}", company.get(field))

    quote = compact_payload.get("quote") or {}
    for field in _QUOTE_FIELDS:
        _add(index, f"quote.{field}", f"Quote {field.replace('_', ' ')}", quote.get(field))

    # -- Phase H.9, sections 1-5: the CURRENT bucket is citable ------------
    #
    # Before this, the only citable free cash flow was `fundamental.
    # free_cash_flow`, computed from the ANNUAL statements. A live analysis
    # therefore printed a trailing-twelve-month figure in the Snapshot and a
    # last-fiscal-year figure in the Bull Case, both correct, both cited,
    # and irreconcilable to a reader. A role cannot quote a current value it
    # has no id for, so the current values get ids -- each carrying the
    # period it covers, because a figure without its period is exactly the
    # ambiguity this phase exists to remove.
    # -- Phase H.12, sections 4/29/31: what this business model licenses ---
    #
    # The classification used to stop at the DCF gate, so a research role saw
    # "current.free_cash_flow" for an insurer with nothing attached and used
    # it exactly as it would for a manufacturer. The packet is indexed as
    # evidence in its own right, and the cash-flow item carries its
    # restriction inline -- a role reading the index cannot miss it.
    packet = compact_payload.get("business_model_evidence") or {}
    if packet:
        _add(index, "business_model.classification", "Business model",
             packet.get("business_model"))
        # Read from the payload's own top-level key: the compact packet drops
        # it precisely because it is already there, and taking it from the
        # packet meant the validator saw nothing and the guard never fired.
        _add(index, "business_model.valuation_method_status",
             "Standard FCFF valuation applicability",
             compact_payload.get("valuation_method_status")
             or packet.get("valuation_method_status"))
        if packet.get("primary_metrics"):
            _add(index, "business_model.primary_metrics",
                 "Metrics that carry the analysis for this business model",
                 ", ".join(packet["primary_metrics"]))
        if packet.get("low_information_metrics"):
            _add(index, "business_model.low_information_metrics",
                 "Metrics with low economic information value for this business model",
                 ", ".join(packet["low_information_metrics"]))
        for i, restriction in enumerate(packet.get("prohibited_interpretations") or []):
            index[f"business_model.prohibited.{i}"] = EvidenceItem(
                evidence_id=f"business_model.prohibited.{i}",
                label=(f"Prohibited interpretation: {restriction.get('metric_id')} may not "
                       f"support a {restriction.get('use', '').replace('_', ' ').lower()}"),
                value=restriction.get("suitability"),
                evidence_type="business_model_policy",
                source_type="business_model_policy",
                derivation=restriction.get("reason"))

    prohibited_metrics = {r.get("metric_id")
                          for r in (packet.get("prohibited_interpretations") or [])}
    cash_flow_label = packet.get("cash_flow_label") or "FCF"

    canonical = compact_payload.get("canonical_evidence") or {}
    for name, metric in (canonical.get("current") or {}).items():
        if not isinstance(metric, dict) or metric.get("value") is None:
            continue
        period = metric.get("period")
        # Section 31: a metric whose ordinary name asserts economics this
        # business model does not support is labelled by what was actually
        # computed. "Free cash flow" claims owner cash; "cash flow after
        # capital expenditure" claims only the subtraction that was done.
        display_name = name.replace("_", " ")
        restriction_note = ""
        if name in prohibited_metrics:
            if name in ("free_cash_flow", "simple_fcf"):
                display_name = cash_flow_label.lower()
            restriction_note = (
                " NOTE: for this business model this figure may be reported but may NOT be "
                "used to claim cash available to the owners, liquidity strength, or "
                "valuation support -- see the business_model.prohibited.* items.")
        index[f"current.{name}"] = EvidenceItem(
            evidence_id=f"current.{name}",
            label=f"Current {display_name} ({period})",
            value=metric["value"],
            evidence_type="canonical_current",
            source_type="canonical_current",
            source_periods=[period] if period else None,
            derivation=(
                f"The current value, covering {period} on a "
                f"{(metric.get('period_type') or 'reported').lower()} basis. This is "
                "the figure the Snapshot shows; cite it, not the `fundamental.*` "
                "entry of the same name, whenever a claim is about the company "
                "today." + restriction_note),
        )

    for name, metric in (compact_payload.get("fundamental_metrics") or {}).items():
        if isinstance(metric, dict) and metric.get("value") is not None:
            # Section 2: history keeps its own namespace and says so. The
            # label names the fiscal periods the value was computed from, so
            # a role reading the index can see at a glance that
            # `fundamental.free_cash_flow` and `current.free_cash_flow` are
            # two different periods rather than two versions of one number.
            periods = [p for p in (metric.get("inputs") or []) if p]
            span = f" (fiscal {', '.join(periods)})" if periods else ""
            index[f"fundamental.{name}"] = EvidenceItem(
                evidence_id=f"fundamental.{name}",
                label=f"Fundamental: {name.replace('_', ' ')}{span}",
                value=metric["value"],
                source_type="reported_historical",
                source_periods=periods or None,
                derivation=(
                    "Computed from the annual statements"
                    + (f" for {', '.join(periods)}" if periods else "")
                    + ". This is reported history, not the current period."
                ) if periods else None,
            )

    for name, metric in (compact_payload.get("technical_metrics") or {}).items():
        if isinstance(metric, dict) and metric.get("value") is not None:
            _add(index, f"technical.{name}", f"Technical: {name.replace('_', ' ')}",
                 metric["value"])

    dcf = compact_payload.get("dcf") or {}
    # Parts 4-5. Decided ONCE, here, and consulted by every valuation id
    # below. `gate` holds what may be published; anything it does not return
    # is not indexed, so a role cannot cite it -- which is a stronger
    # guarantee than instructing a role not to use it.
    # §9: a figure that is not from the current reported period may not be
    # presented as current. Applied HERE, at the single boundary where roles
    # receive facts, rather than asked of each role in prose. Absent under
    # the default actualization mode, which attaches no freshness view.
    apply_research_freshness(index, compact_payload.get("actualization_freshness"))

    valuation_status = valuation_evidence_status(compact_payload)
    gate = build_valuation_research_evidence(
        valuation_status,
        valuation={"dcf": dcf,
                   "scenario_spread": compact_payload.get("dcf_scenario_spread") or {},
                   "valuation_gap": compact_payload.get("valuation_gap") or {}},
        root_cause=_valuation_root_cause(compact_payload, valuation_status))
    _add_valuation_status_evidence(index, compact_payload, valuation_status, gate)
    valuation_publishable = valuation_status == ValuationStatus.VALID_FOR_RESEARCH

    if not dcf.get("available") and dcf.get("reason") == DcfValidationStatus.ASSUMPTION_REQUIRED:
        # HOOD corrective patch: a DCF that was never even ATTEMPTED (missing
        # history for automatic assumption generation -- finance/workflow.py::
        # run_full_stock_analysis sets `{"available": False, "reason":
        # "DCF_ASSUMPTION_REQUIRED", "detail": ...}`) uses a DIFFERENT dict
        # shape than a DCF that ran but FAILED validation (`{"available": True,
        # "validation_status": ..., "validation_reasons": [...]}`, handled
        # below). Both are members of DcfValidationStatus.INVALID -- research_
        # pipeline.py's own `_dcf_validation_failed` and the shared guardrails
        # given to every stage ("If 'dcf.validation_status' appears in the
        # evidence index and is NOT 'DCF_VALID'...") treat them identically —
        # but before this fix, only the SECOND shape ever populated 'dcf.
        # validation_status' here, so a stage citing it for the FIRST shape
        # (exactly what that guardrail text invites) failed evidence-ID
        # validation: the ID simply did not exist. A live HOOD run hit this in
        # 2 of 5 runs, always on final_investment_synthesizer, always as a
        # schema/citation error with NO repair path (a different failure mode
        # from a content-policy violation). Indexed under the SAME ID/shape so
        # a stage's expectation is met identically regardless of WHICH of the
        # two ways a DCF ends up unusable.
        _add(index, "dcf.validation_status", "DCF validation status",
             DcfValidationStatus.ASSUMPTION_REQUIRED)
        if dcf.get("detail"):
            _add(index, "dcf.validation_reasons", "DCF validation reasons", dcf["detail"])
    elif dcf.get("available"):
        # TSLA DCF validation patch: `validation_status` and `validation_
        # reasons` are indexed UNCONDITIONALLY (when present) so every
        # research-pipeline stage can see the DCF's health regardless of
        # whether it passed. `_add` already skips a None value, so an older/
        # hand-built payload with no `validation_status` field at all simply
        # never gets this ID -- fully backward compatible.
        _add(index, "dcf.validation_status", "DCF validation status",
             dcf.get("validation_status"))
        if dcf.get("validation_reasons"):
            _add(index, "dcf.validation_reasons", "DCF validation reasons",
                 "; ".join(dcf["validation_reasons"]))

        if dcf.get("validation_status") in DcfValidationStatus.INVALID:
            # The DCF ran but FAILED deterministic validation (e.g. a
            # non-monotonic bull/base/bear ordering, or a negative terminal-
            # year FCFF) -- its scenario values and assumptions must NOT
            # become citable evidence. This is enforced STRUCTURALLY, not by
            # prompt instruction alone: `dcf.value_per_share.*`/`dcf.
            # assumption.*`/etc. are simply never added to the index below,
            # so a research stage citing one fails evidence-ID validation
            # (finance/evidence.py::validate_evidence_citations) exactly
            # like citing any other nonexistent ID -- it cannot "use DCF
            # outputs as valid evidence" because there is no valid DCF
            # evidence in the index to cite. Only the validation status/
            # reasons above are indexed; every scenario/assumption/net-debt
            # field below is withheld entirely for this analysis.
            pass
        elif valuation_publishable:
            for field in _DCF_TOP_FIELDS:
                _add(index, f"dcf.{field}", f"DCF {field.replace('_', ' ')}", dcf.get(field))
            for scenario in dcf.get("scenarios") or []:
                name = scenario.get("scenario")
                if not name:
                    continue
                for field in _DCF_SCENARIO_FIELDS:
                    _add(index, f"dcf.{field}.{name}",
                         f"DCF {field.replace('_', ' ')} ({name} scenario)", scenario.get(field))
            _add_dcf_assumption_evidence(index, dcf)

    # -- Phase H.4: the financial BASE and forward evidence --
    #
    # Section 17's requirement, made structural: a research stage cannot
    # distinguish reported history from a trailing-twelve-month figure from
    # management guidance unless the three are separately CITABLE. Each is
    # indexed under its own id with an explicit `source_type`, so a stage
    # that writes "revenue growth is 7.2%" when the company has guided to
    # 2-3% is making a claim it has to attach an id to — and the ids say
    # which kind of thing each number is.
    basis = compact_payload.get("dcf_financial_basis") or {}
    for field, label in (
        ("base_revenue_basis", "Financial base for the valuation"),
        ("balance_sheet_as_of", "Balance-sheet date used in the equity bridge"),
        ("flow_period_start", "Financial base period start"),
        ("flow_period_end", "Financial base period end"),
    ):
        _add(index, f"dcf.basis.{field}", label, basis.get(field))
    _add(index, "dcf.valuation_freshness", "Valuation freshness",
         compact_payload.get("valuation_freshness"))
    _add(index, "dcf.data_completeness", "Data completeness",
         compact_payload.get("data_completeness"))

    guidance = compact_payload.get("management_guidance") or {}
    # GENERALIZED READER-RECALL / claim-horizon phase: every CURRENT statement,
    # not only the one the name-keyed `metrics` view had room for -- a company
    # guiding both a quarter and a full year has both, and each is indexed
    # under its OWN period so a role citing the quarterly one cannot be
    # confused with the annual one merely because they share a metric name.
    # The label states the canonical `fiscal_period` ("Q4 FY2026") rather than
    # a bare fiscal year ("fiscal 2026") -- the bare year is what let a
    # quarterly outlook be described as full-year guidance, since both
    # legitimately contain the token "2026".
    guidance_statements = (guidance.get("all_metrics")
                          or list((guidance.get("metrics") or {}).values()))
    for metric in guidance_statements:
        if not isinstance(metric, dict) or metric.get("low") is None:
            continue
        name = metric.get("name")
        if not name:
            continue
        period_label = metric.get("fiscal_period") or f"fiscal {metric.get('fiscal_year')}"
        period_slug = (period_label or "").lower().replace(" ", "_")
        evidence_id = f"dcf.guidance.{name}.{period_slug}.current"
        index[evidence_id] = EvidenceItem(
            evidence_id=evidence_id,
            label=(f"Current management guidance: {name.replace('_', ' ')} "
                   f"({period_label}, {metric.get('basis')})"),
            value=f"{metric.get('low')} to {metric.get('high')}",
            evidence_type="management_guidance",
            units=metric.get("unit"),
            source_periods=[period_label] if period_label else None,
            # FORWARD-LOOKING, and labelled as such on the item itself so no
            # stage can present it as something the company reported.
            source_type="management_guidance",
            derivation=(f"Stated by management in {guidance.get('source_document')} on "
                        f"{guidance.get('guidance_date')}, for {period_label}. This is a "
                        "forward-looking projection by the company, not a reported "
                        "historical fact, and it does not describe any other period."),
        )

    # Phase H.9, section 22, made structural. The compact report already
    # refuses to print a price-vs-value percentage when the DCF is not
    # economically suitable or the per-share basis does not reconcile -- but
    # the gap fields stayed citable, so a live report carried "Market-price
    # comparison: not meaningful" in one section and a precise percentage
    # premium in the next, both true, both cited. A percentage the report
    # will not state is not evidence a research role may state either, so it
    # is WITHHELD from the index rather than merely discouraged, exactly as
    # a failed DCF's scenario values are.
    # Part 5: the same status decides this, rather than a second opinion
    # assembled from suitability and the share reconciliation. Those two
    # inputs still matter -- they are what `valuation_evidence_status` reads
    # -- but they are read once, in one place, for the whole index.
    gap = compact_payload.get("valuation_gap") or {}
    if gap.get("available") and valuation_publishable:
        for field in _VALUATION_GAP_FIELDS:
            _add(index, f"valuation_gap.{field}", f"Valuation gap: {field.replace('_', ' ')}",
                 gap.get(field))
    elif gap.get("available") and valuation_status in _DIRECTION_SURVIVES:
        # The direction survives -- "the price is above the modeled value" is
        # a statement the inputs do support. Only the magnitude goes.
        _add(index, "valuation_gap.direction", "Valuation gap: direction",
             gap.get("direction"))
        _add(index, "valuation_gap.comparison_withheld",
             "Why the price-vs-value percentage is unavailable",
             ("The DCF inputs do not support a percentage comparison against the market "
              "price. State the direction only; do not quote or estimate a percentage "
              "premium, discount, or implied return."))

    # The spread IS three modelled per-share values. Publishing it while
    # withholding `dcf.value_per_share.*` would hand a role the same numbers
    # under a different name.
    spread = compact_payload.get("dcf_scenario_spread") or {}
    if spread.get("available") and valuation_publishable:
        for field in _SCENARIO_SPREAD_FIELDS:
            _add(index, f"scenario_spread.{field}", f"Scenario spread: {field.replace('_', ' ')}",
                 spread.get(field))

    for i, warning in enumerate(compact_payload.get("warnings") or []):
        _add(index, f"warning.{i}", "Warning", warning)

    for dataset, provenance in (compact_payload.get("data_provenance") or {}).items():
        if not isinstance(provenance, dict):
            continue
        _add(index, f"provenance.{dataset}.origin", f"{dataset} data origin",
             provenance.get("origin"))
        _add(index, f"provenance.{dataset}.stale", f"{dataset} data stale?",
             provenance.get("stale"))
        # Phase H.3: which of Yahoo/SEC/Alpha Vantage actually supplied this
        # dataset -- lets a research stage correctly attribute a claim (e.g.
        # "per SEC filings") instead of assuming a single provider.
        _add(index, f"provenance.{dataset}.provider", f"{dataset} data provider",
             provenance.get("provider"))

    plan = compact_payload.get("plan") or {}
    for dataset in plan.get("omitted_datasets") or []:
        _add(index, f"plan.omitted.{dataset}", f"Omitted dataset: {dataset}",
             (plan.get("omission_effects") or {}).get(dataset))

    return index


def _render_line(item: EvidenceItem) -> str:
    line = f"{item.evidence_id}: {item.label} = {json.dumps(item.value, default=str)}"
    # H.4 corrective patch (Problem 3): DCF assumption items additionally
    # carry WHY the value is what it is (scenario/units/source/derivation),
    # so a research stage reading this line can explain a scenario
    # difference ("bull uses a higher revenue_growth and a lower wacc"),
    # not just cite that the modeled values differ. Every other evidence
    # item has none of these fields set, so its rendering is byte-identical
    # to before this field set existed.
    detail = []
    if item.units:
        detail.append(f"units={item.units}")
    if item.scenario:
        detail.append(f"scenario={item.scenario}")
    if item.source_type:
        detail.append(f"source={item.source_type}")
    if item.approval_status:
        detail.append(f"approval={item.approval_status}")
    if item.derivation:
        detail.append(f"derivation={item.derivation}")
    if detail:
        line += " [" + "; ".join(detail) + "]"
    return line


def render_evidence_index(index: Dict[str, EvidenceItem]) -> str:
    """A compact, LLM-facing rendering: one line per evidence item.

    Deliberately NOT the original nested JSON (which a stage might be tempted
    to quote wholesale) — a flat `id: label = value` listing keeps every
    citable fact tied to the ID a stage must actually use. A `dcf.assumption.*`
    item additionally carries a bracketed provenance summary (see
    `_render_line`) so a stage can reason about WHY a value is what it is.
    """
    return "\n".join(_render_line(item) for item in index.values())


def validate_evidence_citations(cited_ids, index: Dict[str, EvidenceItem]) -> Tuple[bool, List[str]]:
    """Check that every cited ID actually exists in the index.

    Returns (all_valid, unknown_ids). Fail closed: a non-list `cited_ids`, or
    any entry that is not a known ID, makes `all_valid` False — this is the
    code-level backstop behind "never invent provider facts."
    """
    if not isinstance(cited_ids, list):
        return False, []
    unknown = [cid for cid in cited_ids if not isinstance(cid, str) or cid not in index]
    return (len(unknown) == 0), unknown


def apply_research_freshness(index: Dict[str, "EvidenceItem"],
                             freshness: Optional[dict]) -> List[str]:
    """Strip the claim of currency from any item that cannot support it.

    Returns the evidence IDs that were requalified, so the caller can record
    what changed rather than the change being invisible.

    The item KEEPS its value. A prior-quarter figure is real and a role may
    legitimately need it; what it may not do is wear the word "current".
    Dropping it would lose information, and leaving it unqualified is the
    failure this exists to prevent.
    """
    from finance.actualization import (
        CURRENT_EVIDENCE_PREFIX,
        requalify_evidence_label,
    )

    if not freshness:
        return []
    requalified = freshness.get("requalified") or {}
    if not requalified:
        return []

    changed: List[str] = []
    for evidence_id, item in list(index.items()):
        if not evidence_id.startswith(CURRENT_EVIDENCE_PREFIX):
            continue
        metric = evidence_id[len(CURRENT_EVIDENCE_PREFIX):]
        period = requalified.get(metric)
        if not period:
            continue
        index[evidence_id] = dataclasses.replace(
            item,
            label=requalify_evidence_label(item.label, period),
            source_periods=[period])
        changed.append(evidence_id)
    return changed
