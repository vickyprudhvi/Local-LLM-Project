"""Phase H.6 — net debt from named components, under a named policy.

THE BUG THIS EXISTS TO FIX (live NVDA, valuation run 2026-08-17)
================================================================
NVIDIA's equity bridge subtracted a net debt of -$3.77B. Reconciling that
against the actual 2026-04-26 balance sheet:

    reported                                  used by the bridge
    ------------------------------------      ------------------
    cash and equivalents      $13,237M        $13,237M
    marketable securities     $37,098M        NOT FOUND  (0)
    equity securities (FVNI)  $30,237M        not applicable
    current debt               $1,000M        $1,000M
    current portion of LTD     $1,000M        $1,000M   <- the SAME $1,000M
    long-term debt             $7,470M        $7,470M
    total debt                 $8,470M        $9,470M

Two independent defects, compounding:

1. `DebtCurrent` and `LongTermDebtCurrent` are BOTH $1,000M because they are
   the same obligation. `DebtCurrent` is defined in us-gaap as the amount of
   short-term AND current-portion-of-long-term debt — it CONTAINS the other.
   Adding them overstated total debt by $1.0B. NVDA's own `LongTermDebt` tag
   ($8,470M, the total including current maturities) says so directly, and
   nothing was checking it.

2. `short_term_investments` resolved to nothing because NVDA renamed the tag
   (`MarketableSecuritiesCurrent` -> `DebtSecuritiesCurrent`) in fiscal 2027,
   so $37.1B of liquidity was simply absent. Fixed in
   finance/xbrl_mapping.py; the reconciliation here is what would have
   CAUGHT it — a company with $151.0B of current assets, of which the bridge
   could account for $13.2B, is not a reconciled balance sheet.

WHAT THIS MODULE GUARANTEES
===========================
* Components stay SEPARATE and individually citable (section 14). Nothing
  downstream ever sees a pre-netted number without also being able to see
  what went into it.
* The policy is NAMED and recorded on the result. `cash_only` and
  `cash_and_marketable_securities` produce different, both-defensible
  numbers; silently switching between them is what makes two runs of the
  same company disagree for no visible reason.
* Overlapping concepts are DETECTED, not summed (section 15).
* The final figure is RECALCULATED from the components before the DCF runs,
  and a mismatch is a structured failure (section 16), never something the
  model is asked to explain away.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from finance import validity

# Deterministic guard code (section 16).
DCF_NET_DEBT_RECONCILIATION_FAILURE = "DCF_NET_DEBT_RECONCILIATION_FAILURE"
# Reported when two selected debt concepts are known to overlap.
DCF_NET_DEBT_COMPONENT_OVERLAP = "DCF_NET_DEBT_COMPONENT_OVERLAP"

# How far a recalculated net debt may differ from the reported one before it
# is a reconciliation FAILURE. Tight: this compares a number against its own
# components, so anything beyond floating-point noise is a real disagreement.
RECONCILIATION_RELATIVE_TOLERANCE = 0.001
RECONCILIATION_ABSOLUTE_FLOOR = 1.0


class NetDebtPolicyName:
    """The two supported definitions. Mirrors finance/dcf.py::NetDebtPolicy
    exactly — this module decides WHICH COMPONENTS the policy consumes;
    finance/dcf.py remains the only place the equity bridge is computed."""

    CASH_ONLY = "cash_only"
    CASH_AND_MARKETABLE_SECURITIES = "cash_and_marketable_securities"
    ALL = (CASH_ONLY, CASH_AND_MARKETABLE_SECURITIES)


# us-gaap concepts whose value ALREADY INCLUDES the current portion of
# long-term debt. When one of these supplies `short_term_debt`, adding a
# separately-resolved `current_portion_of_long_term_debt` counts the same
# obligation twice.
#
# `DebtCurrent` is the one that matters in practice and its us-gaap
# definition is explicit: "Amount of short-term and current portion of
# long-term debt." NVDA reports it at exactly the same $1,000M as
# `LongTermDebtCurrent`, which is the whole of its current debt.
CURRENT_DEBT_INCLUSIVE_CONCEPTS = frozenset({
    "DebtCurrent",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "DebtLongtermAndShorttermCombinedAmount",
})

# Concepts whose value already includes BOTH current and noncurrent debt, so
# they are a cross-check on the component sum rather than a component of it.
TOTAL_DEBT_INCLUSIVE_CONCEPTS = frozenset({
    "LongTermDebt",
    "DebtLongtermAndShorttermCombinedAmount",
})

DEBT_COMPONENT_FIELDS = ("short_term_debt", "current_portion_of_long_term_debt",
                         "long_term_debt")
LIQUIDITY_COMPONENT_FIELDS = ("cash_and_cash_equivalents", "short_term_investments")


@dataclass
class NetDebtComponents:
    """Every field section 14 requires kept separate, plus what was excluded.

    `excluded` is not decoration: a reader who sees total debt of $8.47B
    against components of 1.0 + 1.0 + 7.47 needs to be told which $1.0B was
    dropped and why, or the arithmetic looks wrong.
    """

    cash_and_cash_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    equity_securities_at_fair_value: Optional[float] = None
    # Section 27. Restricted cash is NOT available to repay debt -- it is
    # pledged, escrowed or held against a specific obligation -- so it is
    # reported as its own component and never netted by either policy.
    restricted_cash: Optional[float] = None
    short_term_debt: Optional[float] = None
    current_portion_of_long_term_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    # Section 27. Lease liabilities are a financing obligation under IFRS 16
    # and ASC 842 but are excluded from this project's documented total-debt
    # policy. Kept visible so a reader can see the size of what the policy
    # leaves out rather than having to notice its absence.
    lease_liabilities: Optional[float] = None
    total_debt: Optional[float] = None
    # Phase H.15: whether `total_debt` may be consumed. INVALID means the
    # figure is not known to describe total debt, and `total_debt` is None.
    total_debt_validity: str = "VALID"
    total_debt_reasons: list = field(default_factory=list)
    reported_total_debt: Optional[float] = None
    # Section 28. Management's OWN net-debt measure, with its own definition.
    # Never overwrites the model figure; the two are reconciled and their
    # difference classified.
    company_reported_net_debt: Optional[float] = None
    company_net_debt_definition: Optional[str] = None
    as_of_date: Optional[str] = None
    unit: str = "USD"
    concepts: Dict[str, str] = field(default_factory=dict)
    evidence_ids: Dict[str, str] = field(default_factory=dict)
    excluded: List[dict] = field(default_factory=list)
    findings: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cash_and_cash_equivalents": self.cash_and_cash_equivalents,
            "short_term_investments": self.short_term_investments,
            "equity_securities_at_fair_value": self.equity_securities_at_fair_value,
            "restricted_cash": self.restricted_cash,
            "lease_liabilities": self.lease_liabilities,
            "company_reported_net_debt": self.company_reported_net_debt,
            "company_net_debt_definition": self.company_net_debt_definition,
            "short_term_debt": self.short_term_debt,
            "current_portion_of_long_term_debt": self.current_portion_of_long_term_debt,
            "long_term_debt": self.long_term_debt,
            "total_debt": self.total_debt,
            # Spec section 9: the validity travels with the figure, so the
            # dependency graph can invalidate what depends on it.
            "total_debt_validity": self.total_debt_validity,
            "total_debt_reasons": list(self.total_debt_reasons),
            "reported_total_debt": self.reported_total_debt,
            "as_of_date": self.as_of_date,
            "unit": self.unit,
            "concepts": dict(self.concepts),
            "evidence_ids": dict(self.evidence_ids),
            "excluded": [dict(e) for e in self.excluded],
            "findings": [dict(f) for f in self.findings],
        }


@dataclass
class NetDebtResult:
    """A net-debt figure that can be re-derived from what is printed with it."""

    value: Optional[float]
    policy: str
    components: NetDebtComponents
    eligible_marketable_securities: float = 0.0
    derivation: str = ""
    reconciled: bool = True
    findings: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "net_debt": self.value,
            "net_debt_policy": self.policy,
            "eligible_marketable_securities": self.eligible_marketable_securities,
            "derivation": self.derivation,
            "reconciled": self.reconciled,
            "components": self.components.to_dict(),
            # Spec section 9: hoisted to the top level so the dependency
            # graph can read it without reaching into `components`.
            "total_debt_validity": self.components.total_debt_validity,
            "total_debt_reasons": list(self.components.total_debt_reasons),
            "findings": [dict(f) for f in self.findings],
        }


def _finding(code: str, severity: str, message: str, **detail) -> dict:
    entry = {"code": code, "severity": severity, "message": message}
    entry.update(detail)
    return entry


def _reported_total_is_inclusive(reported_total, concept, selections) -> bool:
    """Does this reported figure really cover BOTH current and noncurrent debt?

    `DebtLongtermAndShorttermCombinedAmount` says so in its name and is taken
    at face value. `LongTermDebt` is the ambiguous one: the us-gaap
    definition includes current maturities, but the tag is widely used for
    the noncurrent portion alone. When the issuer reports
    `LongTermDebtNoncurrent` at the same value, the tag is being used for the
    part -- and a "total" that equals the noncurrent portion cannot be a
    total whenever any current debt exists.

    Returns False when inclusiveness cannot be established, because a
    cross-check that might be comparing a whole against its own part is worth
    less than no cross-check at all.
    """
    if concept == "DebtLongtermAndShorttermCombinedAmount":
        return True
    if concept != "LongTermDebt" or reported_total is None:
        return False
    noncurrent = None
    for name in ("long_term_debt", "long_term_debt_noncurrent"):
        selection = (selections or {}).get(name)
        value = getattr(selection, "value", None)
        if value is None and isinstance(selection, dict):
            value = selection.get("value")
        if value is not None:
            noncurrent = float(value)
            break
    if noncurrent is None:
        return False
    # Equal to the noncurrent portion -> it IS the noncurrent portion.
    return not _within_tolerance(noncurrent, reported_total)


def collect_components(selections: Dict[str, object],
                       reported_total_debt: Optional[float] = None,
                       reported_total_debt_concept: Optional[str] = None,
                       as_of_date: Optional[str] = None) -> NetDebtComponents:
    """Gather the named components from a `{field: SelectedValue}` mapping and
    resolve the overlaps BEFORE anything is summed.

    Takes the SelectedValue objects (finance/freshness.py) rather than plain
    numbers so each component keeps its concept and evidence id — the overlap
    rule is about which CONCEPT supplied a value, and a bare float cannot
    answer that.
    """
    components = NetDebtComponents(as_of_date=as_of_date,
                                   reported_total_debt=reported_total_debt)

    def read(name):
        selection = selections.get(name)
        if selection is None:
            return None, None
        value = getattr(selection, "value", None)
        concept = _concept_of(selection)
        if value is not None and concept:
            components.concepts[name] = concept
            evidence = getattr(selection, "evidence_id", None)
            if evidence:
                components.evidence_ids[name] = evidence
            unit = getattr(selection, "unit", None)
            if unit:
                components.unit = unit
        return value, concept

    components.cash_and_cash_equivalents, _ = read("cash_and_cash_equivalents")
    components.short_term_investments, _ = read("short_term_investments")
    components.restricted_cash, _ = read("restricted_cash")
    components.lease_liabilities, _ = read("lease_liabilities")
    components.equity_securities_at_fair_value, _ = read("equity_securities_at_fair_value")
    short_term_debt, short_term_concept = read("short_term_debt")
    current_portion, _ = read("current_portion_of_long_term_debt")
    long_term_debt, _ = read("long_term_debt")

    components.short_term_debt = short_term_debt
    components.current_portion_of_long_term_debt = current_portion
    components.long_term_debt = long_term_debt

    # -- overlap rule (section 15) --------------------------------------
    # A `short_term_debt` taken from a concept that already includes the
    # current portion of long-term debt makes the separately-resolved current
    # portion a DUPLICATE, not an addition.
    parts = {"short_term_debt": short_term_debt,
             "current_portion_of_long_term_debt": current_portion,
             "long_term_debt": long_term_debt}
    if (short_term_debt is not None and current_portion is not None
            and short_term_concept in CURRENT_DEBT_INCLUSIVE_CONCEPTS):
        parts["current_portion_of_long_term_debt"] = None
        components.excluded.append({
            "field": "current_portion_of_long_term_debt",
            "value": current_portion,
            "reason": (
                f"short_term_debt was taken from {short_term_concept}, whose us-gaap definition "
                "already includes the current portion of long-term debt. Adding the separately "
                f"reported current portion ({current_portion:,.0f}) would count the same "
                "obligation twice."),
        })
        components.findings.append(_finding(
            DCF_NET_DEBT_COMPONENT_OVERLAP, "info",
            f"current_portion_of_long_term_debt ({current_portion:,.0f}) is contained within "
            f"short_term_debt ({short_term_debt:,.0f}, {short_term_concept}) and was excluded "
            "from total debt to avoid double-counting.",
            excluded_field="current_portion_of_long_term_debt",
            excluded_value=current_portion, containing_concept=short_term_concept))

    known = [(name, value) for name, value in parts.items() if value is not None]
    components.total_debt = sum(value for _n, value in known) if known else None

    # -- cross-check against a reported total, when the issuer files one ---
    #
    # Phase H.10, section 14. `LongTermDebt` is on the inclusive list because
    # its us-gaap definition covers current maturities too -- but plenty of
    # filers tag it as the NONCURRENT portion alone, and against those the
    # component sum was being compared with one of its own components and the
    # difference reported as a discrepancy. The concept name is no longer
    # taken as proof of what it contains: when the issuer also reports the
    # noncurrent portion separately and the two are equal, the tag is the
    # part, not the whole, and no cross-check is possible.
    component_sum_before = components.total_debt
    inclusive = _reported_total_is_inclusive(
        reported_total_debt, reported_total_debt_concept, selections)
    if (components.total_debt is not None and reported_total_debt is not None
            and reported_total_debt_concept in TOTAL_DEBT_INCLUSIVE_CONCEPTS
            and inclusive):
        if not _within_tolerance(components.total_debt, reported_total_debt):
            # Phase H.15, sections 4-7. This used to keep the component sum
            # and file a warning beside it. Measured on a synthetic issuer,
            # an $80.0B sum against a $95B reported total produced exactly
            # that -- and net debt, leverage, the equity bridge, the modelled
            # value per share and the risk that followed all consumed the
            # 80B without ever reading the finding.
            #
            # Identity has already been established above (`inclusive`), so
            # precedence may now apply: the issuer's own consolidated total
            # outranks a sum assembled here, which can omit a component the
            # issuer included. That is a resolution, not a preference for a
            # provider, and it is stated on the metric.
            resolved = validity.resolve_or_invalidate(
                "total_debt",
                component_sum=components.total_debt,
                reported_total=reported_total_debt,
                reported_is_authoritative=True,
                reported_concept=reported_total_debt_concept)
            components.total_debt_validity = resolved.validity
            components.total_debt_reasons = list(resolved.reasons)
            if resolved.usable:
                components.total_debt = resolved.raw_value
                components.findings.append(_finding(
                    DCF_NET_DEBT_COMPONENT_OVERLAP, "info",
                    resolved.reasons[0] if resolved.reasons else
                    "The issuer's reported total debt was used in place of the component sum.",
                    component_total=component_sum_before,
                    reported_total=reported_total_debt,
                    concept=reported_total_debt_concept,
                    resolution="issuer_reported_total"))
            else:
                components.total_debt = None
                components.findings.append(_finding(
                    validity.TOTAL_DEBT_CONFLICT, "error",
                    resolved.reasons[0] if resolved.reasons else
                    "Total debt could not be reconciled and is not used.",
                    component_total=component_sum_before,
                    reported_total=reported_total_debt,
                    concept=reported_total_debt_concept))
    return components


def _concept_of(selection) -> Optional[str]:
    """The winning XBRL concept behind a SelectedValue.

    `SelectedValue` records the concept inside its human-readable
    `derivation` text rather than as a field, so the concept is read from the
    dedicated attribute when one exists and parsed out of the derivation
    otherwise. Returns None rather than guessing when neither is available —
    an unknown concept must never satisfy an overlap rule.
    """
    concept = getattr(selection, "concept", None)
    if concept:
        return concept
    derivation = getattr(selection, "derivation", None) or ""
    marker = " as "
    if marker in derivation:
        tail = derivation.split(marker)[-1].strip().rstrip(".")
        candidate = tail.split()[0].rstrip(".") if tail else ""
        if candidate and candidate[0].isupper():
            return candidate
    return None


def _within_tolerance(left: float, right: float) -> bool:
    tolerance = max(abs(left) * RECONCILIATION_RELATIVE_TOLERANCE,
                    RECONCILIATION_ABSOLUTE_FLOOR)
    return abs(left - right) <= tolerance


def compute_net_debt(components: NetDebtComponents, policy: str,
                     marketable_securities_eligible: bool = True) -> NetDebtResult:
    """Apply ONE named policy to the collected components.

        cash_only                       net_debt = total_debt - cash
        cash_and_marketable_securities  net_debt = total_debt - cash
                                                   - eligible marketable securities

    `marketable_securities_eligible` is the operator's configured switch
    (tools.config.dcf_short_term_investments_eligible). When it is off under
    the securities policy, the securities are reported but contribute 0, and
    the derivation says so — the number never changes meaning silently.
    """
    if policy not in NetDebtPolicyName.ALL:
        raise ValueError(f"net_debt_policy {policy!r} is not supported; "
                         f"must be one of {NetDebtPolicyName.ALL}.")

    findings: List[dict] = list(components.findings)
    total_debt = components.total_debt
    cash = components.cash_and_cash_equivalents

    if total_debt is None or cash is None:
        missing = [name for name, value in (("total_debt", total_debt),
                                            ("cash_and_cash_equivalents", cash))
                   if value is None]
        return NetDebtResult(
            value=None, policy=policy, components=components, reconciled=False,
            derivation=f"Net debt could not be computed: {', '.join(missing)} is not available "
                       f"on the {components.as_of_date or 'selected'} balance sheet.",
            findings=findings)

    eligible = 0.0
    securities_note = ""
    if policy == NetDebtPolicyName.CASH_AND_MARKETABLE_SECURITIES:
        securities = components.short_term_investments
        if securities is None:
            securities_note = (" No current marketable-securities balance was reported at this "
                               "date, so none was netted.")
        elif not marketable_securities_eligible:
            securities_note = (
                f" Current marketable securities of {securities:,.0f} were reported but the "
                "configured policy does not treat them as eligible, so none was netted.")
        else:
            eligible = float(securities)
            securities_note = f" less eligible marketable securities ({eligible:,.0f})"

    value = total_debt - cash - eligible
    derivation = (f"Net debt under the {policy!r} policy at {components.as_of_date}: "
                  f"total debt ({total_debt:,.0f}) less cash and equivalents ({cash:,.0f})"
                  + securities_note + ".")
    if components.excluded:
        derivation += " " + " ".join(e["reason"] for e in components.excluded)

    result = NetDebtResult(value=value, policy=policy, components=components,
                           eligible_marketable_securities=eligible,
                           derivation=derivation, findings=findings)
    return result


def verify_net_debt(result: NetDebtResult, reported_net_debt: Optional[float]) -> NetDebtResult:
    """Section 16: recalculate net debt from the selected components and check
    it against what the DCF actually reported, BEFORE the valuation is used.

    A mismatch is recorded as DCF_NET_DEBT_RECONCILIATION_FAILURE on the
    result. It is deliberately not raised and deliberately not handed to the
    model to repair — an equity bridge that does not reconcile against its own
    inputs is a defect in this code, and the only honest response is to say so
    on the report.
    """
    if reported_net_debt is None or result.value is None:
        return result
    if _within_tolerance(result.value, reported_net_debt):
        return result
    result.reconciled = False
    result.findings = list(result.findings) + [_finding(
        DCF_NET_DEBT_RECONCILIATION_FAILURE, "error",
        f"Net debt reported by the valuation ({reported_net_debt:,.0f}) does not equal the "
        f"figure recalculated from the selected balance-sheet components "
        f"({result.value:,.0f}) under the {result.policy!r} policy. "
        f"{result.derivation}",
        reported_net_debt=reported_net_debt, recalculated_net_debt=result.value,
        difference=reported_net_debt - result.value, policy=result.policy)]
    return result


# ---------------------------------------------------------------------------
# Sections 28-29 — company-defined vs model-defined
# ---------------------------------------------------------------------------

LEVERAGE_DEFINITION_MISMATCH = "LEVERAGE_DEFINITION_MISMATCH"


def compare_company_and_model_net_debt(model: NetDebtResult,
                                       company_reported: Optional[float],
                                       definition: Optional[str] = None) -> dict:
    """Section 28: reconcile management's net debt with the model's.

    A difference here is normally a DEFINITION difference, not an error.
    Management commonly nets restricted cash, includes lease liabilities,
    counts long-term investments as liquidity, or reports on a
    proportionally-consolidated basis. Treating that as an arithmetic failure
    would fire on healthy companies; treating the two figures as
    interchangeable would let a company-defined measure silently replace the
    one the equity bridge uses. Both are recorded, and the difference is
    CLASSIFIED.
    """
    record = {
        "model_net_debt": model.value,
        "model_net_debt_policy": model.policy,
        "company_reported_net_debt": company_reported,
        "company_net_debt_definition": definition,
        "difference": None,
        "status": "UNAVAILABLE",
        "findings": [],
    }
    if model.value is None or company_reported is None:
        return record
    difference = company_reported - model.value
    record["difference"] = difference
    if _within_tolerance(model.value, company_reported):
        record["status"] = "CONSISTENT"
        return record
    record["status"] = LEVERAGE_DEFINITION_MISMATCH
    record["findings"].append(_finding(
        LEVERAGE_DEFINITION_MISMATCH, "info",
        f"Management reports net debt of {company_reported:,.0f}; this project's "
        f"{model.policy!r} policy computes {model.value:,.0f} from the reported balance-sheet "
        f"components, a difference of {difference:+,.0f}. This is a difference of DEFINITION, "
        "not an arithmetic error: the two measures are reported side by side and are never "
        "substituted for each other."
        + (f" Management's stated definition: {definition}" if definition else ""),
        model_net_debt=model.value, company_reported_net_debt=company_reported,
        difference=difference, policy=model.policy))
    return record


# ---------------------------------------------------------------------------
# Section 30 — free-cash-flow semantics
# ---------------------------------------------------------------------------


class FreeCashFlowDefinition:
    """Section 30. Three different numbers that are all called "free cash flow".

    Rendering operating cash flow AS free cash flow, or using a company's own
    adjusted definition in one section and the simple one in another, makes
    two sections of the same report disagree about the same quantity.
    """

    SIMPLE = "simple_fcf"                 # operating cash flow - capital expenditure
    COMPANY_DEFINED = "company_defined_fcf"
    ADJUSTED = "adjusted_fcf"
    ALL = (SIMPLE, COMPANY_DEFINED, ADJUSTED)

    LABELS = {
        SIMPLE: "free cash flow (operating cash flow less capital expenditure)",
        COMPANY_DEFINED: "free cash flow (as the company defines it)",
        ADJUSTED: "adjusted free cash flow",
    }


def describe_free_cash_flow(definition: str) -> str:
    return FreeCashFlowDefinition.LABELS.get(
        definition, "free cash flow (definition not stated)")
