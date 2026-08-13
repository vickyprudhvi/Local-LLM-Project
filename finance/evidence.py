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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from finance.dcf import DcfValidationStatus

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

    for name, metric in (compact_payload.get("fundamental_metrics") or {}).items():
        if isinstance(metric, dict) and metric.get("value") is not None:
            _add(index, f"fundamental.{name}", f"Fundamental: {name.replace('_', ' ')}",
                 metric["value"])

    for name, metric in (compact_payload.get("technical_metrics") or {}).items():
        if isinstance(metric, dict) and metric.get("value") is not None:
            _add(index, f"technical.{name}", f"Technical: {name.replace('_', ' ')}",
                 metric["value"])

    dcf = compact_payload.get("dcf") or {}
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
        else:
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
    for name, metric in (guidance.get("metrics") or {}).items():
        if not isinstance(metric, dict) or metric.get("low") is None:
            continue
        index[f"dcf.guidance.{name}.current"] = EvidenceItem(
            evidence_id=f"dcf.guidance.{name}.current",
            label=(f"Current management guidance: {name.replace('_', ' ')} "
                   f"(fiscal {metric.get('fiscal_year')}, {metric.get('basis')})"),
            value=f"{metric.get('low')} to {metric.get('high')}",
            evidence_type="management_guidance",
            units=metric.get("unit"),
            # FORWARD-LOOKING, and labelled as such on the item itself so no
            # stage can present it as something the company reported.
            source_type="management_guidance",
            derivation=(f"Stated by management in {guidance.get('source_document')} on "
                        f"{guidance.get('guidance_date')}. This is a forward-looking "
                        "projection by the company, not a reported historical fact."),
        )

    gap = compact_payload.get("valuation_gap") or {}
    if gap.get("available"):
        for field in _VALUATION_GAP_FIELDS:
            _add(index, f"valuation_gap.{field}", f"Valuation gap: {field.replace('_', ' ')}",
                 gap.get(field))

    spread = compact_payload.get("dcf_scenario_spread") or {}
    if spread.get("available"):
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
