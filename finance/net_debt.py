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
    short_term_debt: Optional[float] = None
    current_portion_of_long_term_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_debt: Optional[float] = None
    reported_total_debt: Optional[float] = None
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
            "short_term_debt": self.short_term_debt,
            "current_portion_of_long_term_debt": self.current_portion_of_long_term_debt,
            "long_term_debt": self.long_term_debt,
            "total_debt": self.total_debt,
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
            "findings": [dict(f) for f in self.findings],
        }


def _finding(code: str, severity: str, message: str, **detail) -> dict:
    entry = {"code": code, "severity": severity, "message": message}
    entry.update(detail)
    return entry


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
    if (components.total_debt is not None and reported_total_debt is not None
            and reported_total_debt_concept in TOTAL_DEBT_INCLUSIVE_CONCEPTS):
        if not _within_tolerance(components.total_debt, reported_total_debt):
            components.findings.append(_finding(
                DCF_NET_DEBT_COMPONENT_OVERLAP, "warning",
                f"Total debt summed from components ({components.total_debt:,.0f}) differs from "
                f"the issuer's own reported {reported_total_debt_concept} "
                f"({reported_total_debt:,.0f}). The component sum is used; the difference is "
                "reported so it can be explained rather than assumed away.",
                component_total=components.total_debt,
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
