"""Phase H.1 — the deterministic FCFF enterprise-value DCF engine.

Pure arithmetic over explicit assumptions. This module performs the AUTHORITATIVE
valuation: the local LLM may PROPOSE assumptions, but it never does the maths and
it can never change a number after the fact.

Formulas (calculation version below is bumped whenever any of these change):

    FCFF_t  = EBIT_t x (1 - tax_rate)
              + depreciation_amortization_t
              - capital_expenditure_t
              - change_in_net_working_capital_t

    EBIT_t  = revenue_t x operating_margin_t
    revenue_t = revenue_{t-1} x (1 + revenue_growth_t)

    discount_factor_t = 1 / (1 + wacc)^t

    terminal_value    = FCFF_n x (1 + g) / (wacc - g)          [g < wacc, enforced]
    PV(terminal)      = terminal_value x discount_factor_n

    enterprise_value  = sum(FCFF_t x discount_factor_t) + PV(terminal)

    net_debt (policy "cash_only")                    = total_debt - cash_and_cash_equivalents
    net_debt (policy "cash_and_marketable_securities") = total_debt - cash_and_cash_equivalents
                                                         - eligible_short_term_investments
    equity_value      = enterprise_value - net_debt + other_non_operating_assets
                        - preferred_equity - minority_interest
    value_per_share   = equity_value / diluted_shares

Net debt may be NEGATIVE (a net-cash position) and is never clamped to zero —
see `NetDebtPolicy` / `compute_net_debt` for the two supported policies, both
of which are recorded on every result so the equity bridge is fully auditable.

Hard rules enforced here:

* A missing REQUIRED assumption returns DCF_ASSUMPTION_REQUIRED naming exactly
  what is missing. Nothing is silently defaulted or invented.
* Terminal growth must be strictly below WACC, everywhere — including in every
  cell of the sensitivity grid, where violating combinations are rejected rather
  than clamped.
* There are no hidden clamps. Every bound is configured, and breaching one raises
  a named validation error instead of quietly moving the number.
* No NaN or infinity may appear in any output.
* Identical input produces byte-identical output.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import tools.config as config
from tools.base import ToolFailure
from tools.models import (
    DCF_ASSUMPTION_INVALID,
    DCF_ASSUMPTION_REQUIRED,
    DCF_CURRENCY_MISMATCH,
    DCF_DISCOUNT_RATE_OUT_OF_BOUNDS,
    DCF_EQUITY_BRIDGE_FAILURE,
    DCF_FORECAST_HORIZON_INVALID,
    DCF_INVALID_SCENARIO_ORDER,
    DCF_NEGATIVE_TERMINAL_FCFF,
    DCF_NON_FINITE_RESULT,
    DCF_SCENARIO_INVALID,
    DCF_SCENARIO_MONOTONICITY_FAILURE,
    DCF_SHARES_INVALID,
    DCF_TERMINAL_GROWTH_TOO_HIGH,
)

CALCULATION_VERSION = "fcff_enterprise_v1"
MODEL_TYPE = "FCFF_ENTERPRISE_DCF"


class ScenarioResultStatus:
    """Per-scenario classification (TSLA DCF validation patch). A scenario's
    arithmetic can complete (finite, no raised error) and still not be
    trustworthy valuation evidence — this is orthogonal to the hard,
    raise-on-bad-input checks elsewhere in this module."""

    VALID = "valid"
    # Equity value is genuinely negative. NEVER clamped to zero (see
    # `compute_net_debt`'s own docstring for the same rule on net debt) —
    # this is a real, auditable arithmetic result, just an economically
    # noteworthy one.
    NEGATIVE_EQUITY_VALUE = "negative_equity_value"
    # Terminal-year FCFF is negative and `dcf_allow_negative_terminal_fcff()`
    # is not set — a Gordon-growth perpetuity built on a negative terminal
    # cash flow does not represent a going concern that grows forever, so
    # this scenario's terminal value (and everything downstream of it) is
    # not treated as valid.
    INVALID = "invalid"

    ALL = (VALID, NEGATIVE_EQUITY_VALUE, INVALID)


class DcfValidationStatus:
    """The overall, deterministic RESULT-validation status (distinct from the
    ToolFailure codes raised for bad INPUT before any scenario is computed —
    see tools/models.py's comment on the DCF section). Always present on a
    successfully returned `run_dcf()` result; never raised as an exception.

    VALID / VALID_WITH_WARNINGS mean the result is usable as valuation
    evidence (a negative equity value alone only ever produces
    VALID_WITH_WARNINGS — section 7 of the patch this implements is explicit
    that a negative equity value is mathematically possible and must be
    classified, not hidden or treated as a hard failure). Every other value
    means downstream consumers (finance/evidence.py, finance/research_
    pipeline.py, finance/workflow.py's report renderer) must NOT present the
    scenario values as ordinary findings.
    """

    VALID = "DCF_VALID"
    VALID_WITH_WARNINGS = "DCF_VALID_WITH_WARNINGS"
    INVALID_INPUT = "DCF_INVALID_INPUT"
    INVALID_SCENARIO_ORDER = DCF_INVALID_SCENARIO_ORDER
    NEGATIVE_TERMINAL_FCFF = DCF_NEGATIVE_TERMINAL_FCFF
    NONFINITE_OUTPUT = "DCF_NONFINITE_OUTPUT"
    EQUITY_BRIDGE_FAILURE = DCF_EQUITY_BRIDGE_FAILURE
    ASSUMPTION_REQUIRED = DCF_ASSUMPTION_REQUIRED

    ALL = (VALID, VALID_WITH_WARNINGS, INVALID_INPUT, INVALID_SCENARIO_ORDER,
          NEGATIVE_TERMINAL_FCFF, NONFINITE_OUTPUT, EQUITY_BRIDGE_FAILURE,
          ASSUMPTION_REQUIRED)
    # Every status under which valuation-derived numbers (value_per_share,
    # the scenario spread, the market-price valuation gap) ARE usable
    # evidence. Every consumer checks membership in THIS (or the INVALID
    # allowlist below), never "not in USABLE" — an absent/unrecognized
    # `validation_status` (e.g. a hand-built test fixture predating this
    # field) must default to "treated as usable," not silently withheld.
    USABLE = (VALID, VALID_WITH_WARNINGS)
    INVALID = (INVALID_INPUT, INVALID_SCENARIO_ORDER, NEGATIVE_TERMINAL_FCFF,
              NONFINITE_OUTPUT, EQUITY_BRIDGE_FAILURE, ASSUMPTION_REQUIRED)


class NetDebtPolicy:
    """The two supported net-debt policies for the equity bridge.

    CASH_ONLY nets total debt against cash and cash equivalents alone — the
    conservative default: it never assumes an unverified security is liquid.

    CASH_AND_MARKETABLE_SECURITIES additionally nets off
    `eligible_short_term_investments`. This is available but NOT the default:
    per the project's policy, short-term investments are only ELIGIBLE for
    this treatment when explicitly configured as such (see
    `tools.config.dcf_short_term_investments_eligible`) — this module never
    decides liquidity on its own, it only applies whichever policy the caller
    selected to whichever `eligible_short_term_investments` value the caller
    supplied (0 when not eligible).
    """

    CASH_ONLY = "cash_only"
    CASH_AND_MARKETABLE_SECURITIES = "cash_and_marketable_securities"

    ALL = (CASH_ONLY, CASH_AND_MARKETABLE_SECURITIES)


def compute_net_debt(total_debt, cash_and_cash_equivalents,
                     eligible_short_term_investments, policy) -> float:
    """The ONLY place net debt is computed. Never clamped to zero — a result
    below zero is a genuine net-cash position and is returned as-is."""
    total_debt = float(total_debt or 0.0)
    cash_and_cash_equivalents = float(cash_and_cash_equivalents or 0.0)
    eligible_short_term_investments = float(eligible_short_term_investments or 0.0)

    if policy == NetDebtPolicy.CASH_ONLY:
        return total_debt - cash_and_cash_equivalents
    if policy == NetDebtPolicy.CASH_AND_MARKETABLE_SECURITIES:
        return total_debt - cash_and_cash_equivalents - eligible_short_term_investments
    raise ToolFailure(
        DCF_ASSUMPTION_INVALID,
        f"net_debt_policy {policy!r} is not supported; must be one of {NetDebtPolicy.ALL}.")

# Assumptions with no defensible default. Absence is reported, never guessed.
REQUIRED_SCENARIO_FIELDS = (
    "revenue_growth",
    "operating_margin",
    "tax_rate",
    "depreciation_pct_revenue",
    "capex_pct_revenue",
    "working_capital_pct_revenue",
    "wacc",
    "terminal_growth",
)

_ROUND = 6


def _round(value):
    """Round to a fixed precision so repeated runs are byte-identical."""
    return round(float(value), _ROUND)


def _require_finite(value, label):
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ToolFailure(DCF_NON_FINITE_RESULT,
                          f"{label} evaluated to a non-finite number.")
    return number


def _as_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolFailure(DCF_ASSUMPTION_INVALID, f"{label} must be a number.")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ToolFailure(DCF_ASSUMPTION_INVALID, f"{label} must be a finite number.")
    return number


def _as_series(value, horizon, label):
    """Accept either a scalar (held flat across the horizon) or an exact-length
    array. A wrong-length array is an error — never padded or truncated."""
    if isinstance(value, (list, tuple)):
        if len(value) != horizon:
            raise ToolFailure(
                DCF_ASSUMPTION_INVALID,
                f"{label} has {len(value)} entries but the forecast horizon is {horizon}.")
        return [_as_number(v, f"{label}[{i}]") for i, v in enumerate(value)]
    return [_as_number(value, label)] * horizon


def _as_magnitude_series(value, horizon, label):
    """Like `_as_series`, but additionally REJECTS a negative entry — for
    assumptions that are magnitudes by contract (CapEx / D&A as a percent of
    revenue: FCFF subtracts capex and adds D&A as positive quantities, see
    this module's own docstring), never a signed flow.

    TSLA DCF validation patch (section 2, "FCFF sign-convention validation"):
    a provider or an LLM PROPOSING assumptions directly (see `finance.dcf_
    model`'s docstring: "the local LLM may PROPOSE assumptions") could hand
    this tool a still-negative-signed CapEx. `FCFF = NOPAT + D&A - capex -
    ΔNWC` would then silently ADD capex instead of subtracting it — the
    exact double-negative bug described in the patch. Rejected HERE, at
    input validation, rather than letting a mis-signed value flow through
    the arithmetic and produce a silently-wrong result.

    `working_capital_pct_revenue` is deliberately NOT validated this way —
    negative net working capital is economically real and must be preserved
    (see `ScenarioAssumptions`); only CapEx and D&A are non-negative-by-
    contract magnitudes.
    """
    series = _as_series(value, horizon, label)
    for i, v in enumerate(series):
        if v < 0:
            raise ToolFailure(
                DCF_ASSUMPTION_INVALID,
                f"{label}[{i}] is negative ({v!r}); this assumption is a magnitude (percent "
                "of revenue) and must be >= 0 — FCFF already subtracts CapEx and adds D&A as "
                "positive quantities, so a negative value here would silently flip a sign in "
                "the calculation (e.g. ADD capex instead of subtracting it).")
    return series



# The closed vocabulary for WHERE an assumption's value came from. Validated
# whenever a scenario declares provenance for a field — an unrecognized
# source_type is a controlled error, not a silently-accepted free string.
class AssumptionSourceType:
    PROVIDER_FACT = "provider_fact"
    DETERMINISTIC_CALCULATION = "deterministic_calculation"
    CONFIGURED_DEFAULT = "configured_default"
    USER_SUPPLIED = "user_supplied"
    LLM_PROPOSED = "llm_proposed"
    # Phase H.4 additions. These are all still deterministic-calculation in
    # spirit, but collapsing them into one label loses the single distinction
    # the freshness phase exists to preserve: WHICH KIND of evidence an
    # assumption rests on. "Revenue growth is 7.2%" and "management guides to
    # 2-3%" are not the same claim, and a reader cannot tell them apart from
    # a shared `deterministic_calculation` tag.
    MANAGEMENT_GUIDANCE = "management_guidance"
    TTM_CALCULATION = "ttm_calculation"
    HISTORICAL_CALCULATION = "historical_calculation"

    ALL = (PROVIDER_FACT, DETERMINISTIC_CALCULATION, CONFIGURED_DEFAULT,
          USER_SUPPLIED, LLM_PROPOSED, MANAGEMENT_GUIDANCE, TTM_CALCULATION,
          HISTORICAL_CALCULATION)

    # Source types that are FORWARD-looking evidence rather than a measured
    # historical fact. Used by the report and the research pipeline so a
    # forecast is never presented as something the company reported.
    FORWARD_LOOKING = (MANAGEMENT_GUIDANCE, LLM_PROPOSED)


_PROVENANCE_FIELD_NAMES = REQUIRED_SCENARIO_FIELDS


def _validate_assumption_provenance(raw, name) -> Dict[str, dict]:
    """Validate an OPTIONAL per-scenario provenance block (fail closed on a
    malformed shape). Not required — a scenario with none simply carries an
    empty dict — but any entry that IS given must be well-formed: this is
    metadata that will be shown to the user as "why this number", so a
    silently-wrong entry is worse than none."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ToolFailure(DCF_ASSUMPTION_INVALID,
                          f"Scenario {name!r}: 'assumption_provenance' must be an object.")
    out = {}
    for field_name, entry in raw.items():
        if field_name not in _PROVENANCE_FIELD_NAMES:
            raise ToolFailure(
                DCF_ASSUMPTION_INVALID,
                f"Scenario {name!r}: assumption_provenance names unknown field {field_name!r}.")
        if not isinstance(entry, dict):
            raise ToolFailure(
                DCF_ASSUMPTION_INVALID,
                f"Scenario {name!r}: assumption_provenance[{field_name!r}] must be an object.")
        source_type = entry.get("source_type")
        if source_type is not None and source_type not in AssumptionSourceType.ALL:
            raise ToolFailure(
                DCF_ASSUMPTION_INVALID,
                f"Scenario {name!r}: assumption_provenance[{field_name!r}].source_type "
                f"{source_type!r} is not one of {AssumptionSourceType.ALL}.")
        out[field_name] = {
            "value": entry.get("value"),
            "source_type": source_type,
            # Phase H.3 corrective patch (Problem 2): "source_period"/"reason"
            # are the ORIGINAL singular-period/free-text fields, kept for
            # backward compatibility with callers still setting them
            # directly. "source_periods" (plural, every period actually
            # averaged) and "derivation" (the human-readable audit-trail
            # string, e.g. "Average of 5 reported CapEx/revenue ratios
            # (2025=2.00%, 2024=1.85%, ...) = 1.84%") are the CURRENT fields
            # finance/workflow.py::propose_assumptions's provenance_entry()
            # actually emits -- this validator used to silently DROP both
            # (rebuilding every entry from a fixed old-field allowlist),
            # which meant CapEx/D&A/NWC's whole multi-year audit trail
            # never survived past this function despite being computed
            # correctly. Caught via a real, live-data COST regression
            # fixture (tests/test_finance_cost_regression.py) -- a
            # synthetic/hand-built provenance dict in an earlier test never
            # exercised the plural/derivation fields, so the drop was
            # invisible until real multi-year history flowed through.
            "source_period": entry.get("source_period"),
            "source_periods": list(entry.get("source_periods") or []),
            "source_evidence_ids": list(entry.get("source_evidence_ids") or []),
            "reason": entry.get("reason"),
            "derivation": entry.get("derivation"),
            "approval_status": entry.get("approval_status"),
            "units": entry.get("units"),
            # -- CLAMP PROVENANCE --
            # These were being DROPPED. The MLI patch added `clamped` so a
            # capped assumption could never be mistaken for a measured one,
            # and finance/workflow.py::_assumption_quality_limitations reads
            # it off the RESULT -- but this function rebuilds every entry
            # from a fixed allowlist, so the flag never survived the call and
            # that limitation could never fire. Same class of drop as the
            # source_periods/derivation loss recorded above, found the same
            # way: by following a field from where it is set to where it is
            # read, rather than trusting that it arrives.
            "clamped": bool(entry.get("clamped")),
            "raw_value": entry.get("raw_value"),
            "applied_value": entry.get("applied_value"),
            "clamp_bounds": list(entry.get("clamp_bounds") or []) or None,
            "clamp_reason": entry.get("clamp_reason"),
            # -- Phase H.4: the per-YEAR forecast path --
            # A growth assumption is now a path, not a scalar, so its
            # provenance is per forecast year (section 11). Preserved
            # verbatim; this engine never reads it for arithmetic.
            "forecast_path": list(entry.get("forecast_path") or []) or None,
        }
    return out


@dataclass(frozen=True)
class ScenarioAssumptions:
    """One fully-specified scenario. Every field is explicit."""

    name: str
    revenue_growth: List[float]
    operating_margin: List[float]
    tax_rate: float
    depreciation_pct_revenue: List[float]
    capex_pct_revenue: List[float]
    working_capital_pct_revenue: List[float]
    wacc: float
    terminal_growth: float
    # Where each assumption came from (provider_fact / deterministic_calculation
    # / configured_default / user_supplied / llm_proposed), echoed back verbatim
    # in the result. The LLM may PROPOSE assumptions via this metadata, but
    # cannot alter them after the run: the DCF engine only ever reads the
    # numeric fields above for arithmetic, never this dict, and the result
    # reflects exactly what was validated here — never mutated afterward.
    assumption_provenance: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "revenue_growth": [_round(v) for v in self.revenue_growth],
            "operating_margin": [_round(v) for v in self.operating_margin],
            "tax_rate": _round(self.tax_rate),
            "depreciation_pct_revenue": [_round(v) for v in self.depreciation_pct_revenue],
            "capex_pct_revenue": [_round(v) for v in self.capex_pct_revenue],
            "working_capital_pct_revenue": [_round(v)
                                            for v in self.working_capital_pct_revenue],
            "wacc": _round(self.wacc),
            "terminal_growth": _round(self.terminal_growth),
            "assumption_provenance": dict(self.assumption_provenance),
        }


@dataclass(frozen=True)
class DcfInputs:
    """Everything the engine needs. Nothing is read from anywhere else.

    The equity bridge is expressed as explicit, individually-reported
    components — never a single pre-netted `net_debt` — so the bridge that
    produced a result can always be reconstructed from the result alone.
    `total_debt` and `cash_and_cash_equivalents` come from the balance-sheet
    debt/cash aggregation policy in `finance/normalization.py`;
    `eligible_short_term_investments` is 0 unless the caller's configured
    policy marks short-term investments eligible (see `NetDebtPolicy`).
    """

    ticker: str
    valuation_date: str
    currency: str
    base_revenue: float
    forecast_years: int
    diluted_shares: float
    total_debt: float = 0.0
    cash_and_cash_equivalents: float = 0.0
    # Always populated for transparency (the report can show "STI exists but
    # is excluded"), even when the active policy does not treat it as eligible.
    short_term_investments: float = 0.0
    eligible_short_term_investments: float = 0.0
    net_debt_policy: str = NetDebtPolicy.CASH_ONLY
    preferred_equity: float = 0.0
    minority_interest: float = 0.0
    other_non_operating_assets: float = 0.0
    base_working_capital: Optional[float] = None
    source_periods: Tuple[str, ...] = ()
    statement_currency: Optional[str] = None
    provenance: Dict[str, str] = field(default_factory=dict)


def missing_assumptions(raw_scenario) -> Tuple[str, ...]:
    """Which REQUIRED assumption fields are absent from a raw scenario dict."""
    if not isinstance(raw_scenario, dict):
        return REQUIRED_SCENARIO_FIELDS
    return tuple(f for f in REQUIRED_SCENARIO_FIELDS if raw_scenario.get(f) is None)


def build_scenario(raw, horizon) -> ScenarioAssumptions:
    """Validate one raw scenario dict into typed assumptions (fail closed)."""
    if not isinstance(raw, dict):
        raise ToolFailure(DCF_SCENARIO_INVALID, "Each scenario must be an object.")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolFailure(DCF_SCENARIO_INVALID, "Each scenario needs a non-empty 'name'.")

    absent = missing_assumptions(raw)
    if absent:
        raise ToolFailure(
            DCF_ASSUMPTION_REQUIRED,
            f"Scenario {name!r} is missing required assumption(s): {', '.join(absent)}.")

    wacc = _as_number(raw["wacc"], f"{name}.wacc")
    terminal_growth = _as_number(raw["terminal_growth"], f"{name}.terminal_growth")
    tax_rate = _as_number(raw["tax_rate"], f"{name}.tax_rate")

    lo, hi = config.dcf_min_discount_rate(), config.dcf_max_discount_rate()
    if not (lo <= wacc <= hi):
        raise ToolFailure(
            DCF_DISCOUNT_RATE_OUT_OF_BOUNDS,
            f"Scenario {name!r}: WACC {wacc:.4f} is outside the configured "
            f"bounds [{lo}, {hi}].")
    if terminal_growth >= wacc:
        raise ToolFailure(
            DCF_TERMINAL_GROWTH_TOO_HIGH,
            f"Scenario {name!r}: terminal growth {terminal_growth:.4f} must be strictly "
            f"below WACC {wacc:.4f}; the perpetuity is undefined otherwise.")
    if not (0.0 <= tax_rate < 1.0):
        raise ToolFailure(DCF_ASSUMPTION_INVALID,
                          f"Scenario {name!r}: tax_rate must be in [0, 1).")

    return ScenarioAssumptions(
        name=name.strip(),
        revenue_growth=_as_series(raw["revenue_growth"], horizon, f"{name}.revenue_growth"),
        operating_margin=_as_series(raw["operating_margin"], horizon,
                                    f"{name}.operating_margin"),
        tax_rate=tax_rate,
        depreciation_pct_revenue=_as_magnitude_series(raw["depreciation_pct_revenue"], horizon,
                                                      f"{name}.depreciation_pct_revenue"),
        capex_pct_revenue=_as_magnitude_series(raw["capex_pct_revenue"], horizon,
                                               f"{name}.capex_pct_revenue"),
        working_capital_pct_revenue=_as_series(raw["working_capital_pct_revenue"], horizon,
                                               f"{name}.working_capital_pct_revenue"),
        wacc=wacc,
        terminal_growth=terminal_growth,
        assumption_provenance=_validate_assumption_provenance(
            raw.get("assumption_provenance"), name),
    )


def validate_inputs(inputs: DcfInputs):
    """Structural validation independent of any single scenario."""
    lo, hi = config.dcf_min_forecast_years(), config.dcf_max_forecast_years()
    if not isinstance(inputs.forecast_years, int) or isinstance(inputs.forecast_years, bool):
        raise ToolFailure(DCF_FORECAST_HORIZON_INVALID,
                          "'forecast_years' must be an integer.")
    if not (lo <= inputs.forecast_years <= hi):
        raise ToolFailure(
            DCF_FORECAST_HORIZON_INVALID,
            f"'forecast_years' must be between {lo} and {hi}; got {inputs.forecast_years}.")
    if inputs.diluted_shares is None or inputs.diluted_shares <= 0:
        raise ToolFailure(DCF_SHARES_INVALID,
                          "'diluted_shares' must be a positive number.")
    if math.isnan(inputs.diluted_shares) or math.isinf(inputs.diluted_shares):
        raise ToolFailure(DCF_SHARES_INVALID, "'diluted_shares' must be finite.")
    if inputs.base_revenue is None or math.isnan(inputs.base_revenue) \
            or math.isinf(inputs.base_revenue):
        raise ToolFailure(DCF_ASSUMPTION_INVALID, "'base_revenue' must be a finite number.")
    if not isinstance(inputs.currency, str) or not inputs.currency.strip():
        raise ToolFailure(DCF_ASSUMPTION_INVALID, "'currency' is required.")
    if inputs.statement_currency and inputs.statement_currency != inputs.currency:
        raise ToolFailure(
            DCF_CURRENCY_MISMATCH,
            f"The statements are reported in {inputs.statement_currency!r} but the "
            f"valuation currency is {inputs.currency!r}; convert explicitly first.")
    if inputs.net_debt_policy not in NetDebtPolicy.ALL:
        raise ToolFailure(
            DCF_ASSUMPTION_INVALID,
            f"'net_debt_policy' {inputs.net_debt_policy!r} is not supported; "
            f"must be one of {NetDebtPolicy.ALL}.")


def _project_raw(inputs: DcfInputs, scenario: ScenarioAssumptions) -> List[dict]:
    """Year-by-year forecast at FULL float precision.

    Rounding happens only when the result is serialized (`project`). Aggregating
    pre-rounded intermediates would accumulate error into enterprise value, so
    every downstream calculation consumes these raw rows.
    """
    rows = []
    revenue = float(inputs.base_revenue)
    prior_working_capital = (float(inputs.base_working_capital)
                             if inputs.base_working_capital is not None
                             else revenue * scenario.working_capital_pct_revenue[0])

    for year in range(1, inputs.forecast_years + 1):
        index = year - 1
        revenue = revenue * (1.0 + scenario.revenue_growth[index])
        _require_finite(revenue, f"year {year} revenue")

        ebit = revenue * scenario.operating_margin[index]
        nopat = ebit * (1.0 - scenario.tax_rate)
        depreciation = revenue * scenario.depreciation_pct_revenue[index]
        capex = revenue * scenario.capex_pct_revenue[index]
        working_capital = revenue * scenario.working_capital_pct_revenue[index]
        change_in_wc = working_capital - prior_working_capital
        prior_working_capital = working_capital

        fcff = nopat + depreciation - capex - change_in_wc
        _require_finite(fcff, f"year {year} FCFF")

        discount_factor = 1.0 / ((1.0 + scenario.wacc) ** year)
        rows.append({
            "year": year,
            "revenue": revenue,
            "revenue_growth": scenario.revenue_growth[index],
            "operating_margin": scenario.operating_margin[index],
            "ebit": ebit,
            "tax_rate": scenario.tax_rate,
            "nopat": nopat,
            "depreciation_amortization": depreciation,
            "capital_expenditure": capex,
            "net_working_capital": working_capital,
            "change_in_net_working_capital": change_in_wc,
            "fcff": fcff,
            "discount_factor": discount_factor,
            "present_value_fcff": fcff * discount_factor,
        })
    return rows


def project(inputs: DcfInputs, scenario: ScenarioAssumptions) -> List[dict]:
    """The forecast table as it appears in output: fixed precision, so repeated
    runs are byte-identical."""
    return [{k: (_round(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in _project_raw(inputs, scenario)]


def value_scenario(inputs: DcfInputs, scenario: ScenarioAssumptions) -> dict:
    """Run one scenario end to end and return its FULLY AUDITABLE result: every
    forecast-year line, the terminal-value build-up, and every equity-bridge
    component the engine used — never just the final value per share."""
    raw_rows = _project_raw(inputs, scenario)
    rows = [{k: (_round(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in raw_rows]
    horizon = inputs.forecast_years

    # Aggregate at full precision — see `_project_raw`.
    pv_forecast = sum(r["present_value_fcff"] for r in raw_rows)
    terminal_year_fcff = raw_rows[-1]["fcff"]
    final_discount_factor = raw_rows[-1]["discount_factor"]

    denominator = scenario.wacc - scenario.terminal_growth
    if denominator <= 0:
        # Unreachable via build_scenario, but the engine must never divide by a
        # non-positive spread even if called directly.
        raise ToolFailure(
            DCF_TERMINAL_GROWTH_TOO_HIGH,
            "Terminal growth must be strictly below WACC.")

    # TSLA DCF validation patch: a NEGATIVE terminal-year FCFF fed into the
    # Gordon-growth perpetuity does not represent a going concern that grows
    # forever -- it silently produces a perpetuity value with the
    # WRONG economic meaning (e.g. a negative FCFF divided by a SMALLER
    # positive WACC-minus-g spread becomes MORE negative, not less, so a
    # scenario with a narrower spread looks "worse" purely from an artifact
    # of the formula, not the underlying business). Computed BEFORE building
    # `terminal_value` below so the classification reflects the true sign,
    # never a post-hoc reinterpretation of an already-computed number.
    allow_negative_terminal_fcff = config.dcf_allow_negative_terminal_fcff()
    negative_terminal_fcff = terminal_year_fcff < 0 and not allow_negative_terminal_fcff

    terminal_value = terminal_year_fcff * (1.0 + scenario.terminal_growth) / denominator
    _require_finite(terminal_value, "terminal value")
    pv_terminal = terminal_value * final_discount_factor

    enterprise_value = pv_forecast + pv_terminal

    # The equity bridge: every component named explicitly, computed by THIS
    # engine alone, never mutated after the fact. Net debt may be negative
    # (net cash) and is never clamped.
    net_debt = compute_net_debt(inputs.total_debt, inputs.cash_and_cash_equivalents,
                                inputs.eligible_short_term_investments,
                                inputs.net_debt_policy)
    equity_value = (enterprise_value
                    - net_debt
                    + float(inputs.other_non_operating_assets)
                    - float(inputs.preferred_equity)
                    - float(inputs.minority_interest))
    _require_finite(equity_value, "equity value")

    value_per_share = equity_value / float(inputs.diluted_shares)
    _require_finite(value_per_share, "value per share")

    # Per-scenario status (section 7 of the TSLA DCF validation patch): a
    # negative equity value is mathematically possible and is preserved
    # exactly, NEVER clamped to zero -- it is classified, not hidden. A
    # negative terminal FCFF takes priority over a negative-equity
    # classification, since the terminal value itself is not trustworthy in
    # that case (the equity value being negative is then a downstream
    # SYMPTOM of that, not an independent finding).
    scenario_warnings = []
    if negative_terminal_fcff:
        scenario_status = ScenarioResultStatus.INVALID
        scenario_warnings.append(
            f"Scenario {scenario.name!r}: terminal-year FCFF ({_round(terminal_year_fcff):,}) is "
            "negative, so the Gordon-growth perpetuity does not represent a going concern that "
            "grows forever; this scenario's terminal value and enterprise/equity value are not "
            "treated as valid valuation evidence.")
    elif equity_value < 0:
        scenario_status = ScenarioResultStatus.NEGATIVE_EQUITY_VALUE
        scenario_warnings.append(
            f"Scenario {scenario.name!r}: equity value ({_round(equity_value):,}) is negative "
            "after the net debt adjustment. This is preserved exactly, not clamped to zero.")
    else:
        scenario_status = ScenarioResultStatus.VALID

    return {
        "scenario": scenario.name,
        "calculation_version": CALCULATION_VERSION,
        "assumptions": scenario.to_dict(),
        "forecast": rows,
        "forecast_years": horizon,
        "present_value_of_forecast_fcff": _round(pv_forecast),
        "terminal_year_fcff": _round(terminal_year_fcff),
        "terminal_value": _round(terminal_value),
        "present_value_of_terminal_value": _round(pv_terminal),
        "terminal_value_share_of_enterprise_value": (
            _round(pv_terminal / enterprise_value) if enterprise_value else None),
        "enterprise_value": _round(enterprise_value),
        # -- equity bridge: every component, explicit --
        "total_debt": _round(inputs.total_debt),
        "cash_and_cash_equivalents": _round(inputs.cash_and_cash_equivalents),
        "short_term_investments": _round(inputs.short_term_investments),
        "eligible_short_term_investments": _round(inputs.eligible_short_term_investments),
        "net_debt_policy": inputs.net_debt_policy,
        "net_debt": _round(net_debt),
        "other_non_operating_assets": _round(inputs.other_non_operating_assets),
        "preferred_equity": _round(inputs.preferred_equity),
        "minority_interest": _round(inputs.minority_interest),
        "equity_value": _round(equity_value),
        "diluted_shares": _round(inputs.diluted_shares),
        "value_per_share": _round(value_per_share),
        # -- TSLA DCF validation patch: per-scenario classification --
        "status": scenario_status,
        "warnings": scenario_warnings,
    }


def sensitivity_table(inputs: DcfInputs, scenario: ScenarioAssumptions,
                      wacc_values: Sequence[float],
                      terminal_growth_values: Sequence[float]) -> dict:
    """Two-dimensional WACC x terminal-growth grid of value per share.

    A combination where terminal growth >= WACC is REJECTED, not clamped: the
    cell reports the reason and carries no number, so an undefined perpetuity can
    never be mistaken for a valuation.
    """
    rows = []
    for wacc in wacc_values:
        cells = []
        for growth in terminal_growth_values:
            if growth >= wacc:
                cells.append({
                    "wacc": _round(wacc),
                    "terminal_growth": _round(growth),
                    "value_per_share": None,
                    "rejected": True,
                    "reason": DCF_TERMINAL_GROWTH_TOO_HIGH,
                })
                continue
            variant = ScenarioAssumptions(
                name=scenario.name,
                revenue_growth=scenario.revenue_growth,
                operating_margin=scenario.operating_margin,
                tax_rate=scenario.tax_rate,
                depreciation_pct_revenue=scenario.depreciation_pct_revenue,
                capex_pct_revenue=scenario.capex_pct_revenue,
                working_capital_pct_revenue=scenario.working_capital_pct_revenue,
                wacc=float(wacc),
                terminal_growth=float(growth),
            )
            result = value_scenario(inputs, variant)
            cells.append({
                "wacc": _round(wacc),
                "terminal_growth": _round(growth),
                "value_per_share": result["value_per_share"],
                "rejected": False,
            })
        rows.append({"wacc": _round(wacc), "cells": cells})
    return {
        "axis_x": "terminal_growth",
        "axis_y": "wacc",
        "wacc_values": [_round(w) for w in wacc_values],
        "terminal_growth_values": [_round(g) for g in terminal_growth_values],
        "rows": rows,
    }


def _first(value):
    """The representative scalar for a (possibly per-year) assumption
    series -- year 1, matching how a scenario's own headline assumption is
    normally quoted (finance/evidence.py::_assumption_value_for uses the
    same "first entry stands for the scalar" convention)."""
    return value[0] if isinstance(value, (list, tuple)) else value


def _scenario_monotonicity_check(scenarios: List[ScenarioAssumptions],
                                 results_by_name: Dict[str, dict]) -> Optional[dict]:
    """Section 6 of the TSLA DCF validation patch: given the STANDARD
    base/bull/bear construction where bull's assumptions are ALL at least as
    favorable as base's (higher revenue growth, higher operating margin,
    lower WACC, higher terminal growth) and bear's are ALL at least as
    unfavorable, the calculated value_per_share is expected to satisfy
    bull >= base >= bear. This function only ever DETECTS a violation of
    that expectation -- it never reorders, clamps, or otherwise "fixes" a
    value to force the ordering, since forcing it would hide the exact bug
    this check exists to catch (see the TSLA root-cause writeup in
    docs/PHASE_H1_STOCK_ANALYSIS.md: a working-capital assumption that
    included TSLA's own cash balance made bull's higher revenue growth
    manufacture a LARGER fake "cash investment" every forecast year than
    base's, inverting the ordering even though every individual assumption
    was itself constructed correctly).

    Returns None when the check does not apply at all (scenario names are
    not exactly {"base","bull","bear"}, or the assumption construction does
    not actually imply an ordering -- e.g. a caller-supplied "bull" that
    does not have uniformly more favorable assumptions than "base"). This is
    a NARROW, structural check: it validates the RELATIONSHIP between
    scenarios that were already built to imply one, never a general
    "scenarios must always be monotonic" rule.

    Returns {"checked": True, "passed": bool, ...} otherwise. On failure,
    records the exact assumptions and values compared, and which relation(s)
    were violated (section 6: "Record: scenario assumptions, scenario
    values, violated relation").
    """
    if not config.dcf_scenario_monotonicity_check_enabled():
        return None
    by_name = {s.name: s for s in scenarios}
    if set(by_name) != {"base", "bull", "bear"}:
        return None
    base, bull, bear = by_name["base"], by_name["bull"], by_name["bear"]

    def assumption_snapshot(s: ScenarioAssumptions) -> dict:
        return {
            "revenue_growth": _first(s.revenue_growth),
            "operating_margin": _first(s.operating_margin),
            "wacc": s.wacc,
            "terminal_growth": s.terminal_growth,
        }

    bull_a, base_a, bear_a = (assumption_snapshot(bull), assumption_snapshot(base),
                              assumption_snapshot(bear))
    bull_ordered = (bull_a["revenue_growth"] >= base_a["revenue_growth"]
                    and bull_a["operating_margin"] >= base_a["operating_margin"]
                    and bull_a["wacc"] <= base_a["wacc"]
                    and bull_a["terminal_growth"] >= base_a["terminal_growth"])
    bear_ordered = (bear_a["revenue_growth"] <= base_a["revenue_growth"]
                    and bear_a["operating_margin"] <= base_a["operating_margin"]
                    and bear_a["wacc"] >= base_a["wacc"]
                    and bear_a["terminal_growth"] <= base_a["terminal_growth"])
    if not (bull_ordered and bear_ordered):
        # The premise does not hold (this is not a "normal" bull/base/bear
        # construction) -- nothing to enforce, and NOT a violation.
        return None

    bull_v = results_by_name["bull"]["value_per_share"]
    base_v = results_by_name["base"]["value_per_share"]
    bear_v = results_by_name["bear"]["value_per_share"]
    violated = []
    if not (bull_v >= base_v):
        violated.append(f"bull value_per_share ({bull_v}) is not >= base value_per_share ({base_v})")
    if not (base_v >= bear_v):
        violated.append(f"base value_per_share ({base_v}) is not >= bear value_per_share ({bear_v})")

    outcome = {
        "checked": True,
        "passed": not violated,
        "assumptions": {"bull": bull_a, "base": base_a, "bear": bear_a},
        "values": {"bull": bull_v, "base": base_v, "bear": bear_v},
        "violated_relations": violated,
    }
    return outcome


def run_dcf(inputs: DcfInputs, raw_scenarios, sensitivity=None) -> dict:
    """The single entry point. Returns the full structured valuation result."""
    validate_inputs(inputs)

    if not isinstance(raw_scenarios, (list, tuple)) or not raw_scenarios:
        raise ToolFailure(DCF_ASSUMPTION_REQUIRED,
                          "At least one scenario with explicit assumptions is required.")

    names = []
    scenarios = []
    for raw in raw_scenarios:
        scenario = build_scenario(raw, inputs.forecast_years)
        if scenario.name in names:
            raise ToolFailure(DCF_SCENARIO_INVALID,
                              f"Duplicate scenario name {scenario.name!r}.")
        names.append(scenario.name)
        scenarios.append(scenario)

    results = [value_scenario(inputs, s) for s in scenarios]
    by_name = {r["scenario"]: r for r in results}

    # The sensitivity grid always hangs off the FIRST scenario (conventionally
    # "base"), so the table is never silently computed against bull or bear.
    primary = scenarios[0]
    grid = None
    if sensitivity is not None:
        grid = sensitivity_table(
            inputs, primary,
            sensitivity.get("wacc_values") or _default_axis(primary.wacc, 0.01),
            sensitivity.get("terminal_growth_values")
            or _default_axis(primary.terminal_growth, 0.0025),
        )

    warnings = _collect_warnings(inputs, scenarios, results)

    # -- TSLA DCF validation patch: overall RESULT validation, computed AFTER
    # every scenario's arithmetic completes. Never raised -- a full,
    # auditable result is always returned; `validation_status` tells every
    # downstream consumer (finance/evidence.py, finance/research_pipeline.py,
    # finance/workflow.py's report renderer) whether these numbers are usable
    # valuation evidence. Priority order, worst first: a negative terminal
    # FCFF anywhere means the underlying arithmetic used an economically
    # invalid perpetuity, which is worse than an ordering violation (an
    # ordering violation might itself be CAUSED by a negative-terminal-FCFF
    # scenario, so that root cause is surfaced first); a monotonicity
    # violation is next; a negative equity value alone (structurally sound,
    # just economically negative) only ever produces a WARNING, never
    # invalidates the result -- see `ScenarioResultStatus`/section 7.
    monotonicity = _scenario_monotonicity_check(scenarios, by_name)
    validation_reasons = []
    invalid_terminal_scenarios = [r["scenario"] for r in results
                                  if r["status"] == ScenarioResultStatus.INVALID]
    negative_equity_scenarios = [r["scenario"] for r in results
                                 if r["status"] == ScenarioResultStatus.NEGATIVE_EQUITY_VALUE]

    if invalid_terminal_scenarios:
        validation_status = DcfValidationStatus.NEGATIVE_TERMINAL_FCFF
        validation_reasons.append(
            f"{DCF_NEGATIVE_TERMINAL_FCFF}: negative terminal-year FCFF in scenario(s) "
            f"{', '.join(invalid_terminal_scenarios)}; the perpetuity terminal value for "
            "those scenarios is not treated as valid valuation evidence.")
    elif monotonicity is not None and not monotonicity["passed"]:
        validation_status = DcfValidationStatus.INVALID_SCENARIO_ORDER
        validation_reasons.append(
            f"{DCF_SCENARIO_MONOTONICITY_FAILURE}: assumptions are constructed so bull >= base "
            ">= bear should hold, but the calculated value_per_share violates this: "
            + "; ".join(monotonicity["violated_relations"]) + ". Assumptions: "
            f"{monotonicity['assumptions']}. Values: {monotonicity['values']}. This is NOT "
            "auto-corrected -- it indicates a deterministic bug upstream of this result.")
    elif negative_equity_scenarios:
        validation_status = DcfValidationStatus.VALID_WITH_WARNINGS
        validation_reasons.append(
            f"Negative (but structurally valid) equity value in scenario(s) "
            f"{', '.join(negative_equity_scenarios)}.")
    else:
        validation_status = DcfValidationStatus.VALID

    warnings = warnings + [w for r in results for w in r["warnings"] if w not in warnings]

    return {
        "model_type": MODEL_TYPE,
        "calculation_version": CALCULATION_VERSION,
        "ticker": inputs.ticker,
        "valuation_date": inputs.valuation_date,
        "currency": inputs.currency,
        "source_periods": list(inputs.source_periods),
        "forecast_years": inputs.forecast_years,
        "diluted_shares": _round(inputs.diluted_shares),
        "net_debt": _round(compute_net_debt(
            inputs.total_debt, inputs.cash_and_cash_equivalents,
            inputs.eligible_short_term_investments, inputs.net_debt_policy)),
        "net_debt_policy": inputs.net_debt_policy,
        "base_revenue": _round(inputs.base_revenue),
        "scenarios": results,
        "scenario_names": names,
        "primary_scenario": primary.name,
        "value_per_share": by_name[primary.name]["value_per_share"],
        "sensitivity": grid,
        "warnings": warnings,
        "provenance": dict(inputs.provenance),
        # -- TSLA DCF validation patch --
        "validation_status": validation_status,
        "validation_reasons": validation_reasons,
        "scenario_monotonicity": monotonicity,
    }


def _default_axis(centre, step, points=5):
    """A symmetric axis around the scenario's own value."""
    half = points // 2
    return [_round(centre + (i - half) * step) for i in range(points)]


def _collect_warnings(inputs, scenarios, results) -> List[str]:
    """Model-limitation warnings. Facts about the model, not interpretation."""
    warnings = []
    for result in results:
        share = result.get("terminal_value_share_of_enterprise_value")
        if share is not None and share > 0.75:
            warnings.append(
                f"Scenario {result['scenario']!r}: {share:.0%} of enterprise value comes "
                "from the terminal value, so the result is highly sensitive to WACC and "
                "terminal growth.")
        if result["equity_value"] < 0:
            warnings.append(
                f"Scenario {result['scenario']!r}: equity value is negative after the net "
                "debt adjustment.")
    for scenario in scenarios:
        spread = scenario.wacc - scenario.terminal_growth
        if spread < 0.02:
            warnings.append(
                f"Scenario {scenario.name!r}: the WACC-to-terminal-growth spread is only "
                f"{spread:.2%}; small changes will swing the valuation sharply.")
    if not inputs.source_periods:
        warnings.append("No source reporting periods were recorded for this valuation.")
    return warnings
