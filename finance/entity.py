"""Phase H.7 — issuer/security identity, share basis, and the invariants that
catch a per-share figure computed on the wrong denominator.

THE BUG THIS EXISTS TO FIX (live foreign private issuer, 2026-08-19)
====================================================================
A full analysis published a modelled value of $36.55 per share against a
market price of $13.84 -- a 164% discount, the largest this project has ever
produced. It was arithmetic performed on two different securities.

    provider shares outstanding      776,405,057
    provider market capitalisation   $45,524,795,392
    price                            $13.84

    776,405,057 x $13.84  =  $10,745,446,107

That is 76% below the market capitalisation the same provider reported in
the same payload. The issuer has more than one share class; the provider's
"shares outstanding" covers the listed class and its market capitalisation
covers all of them. Dividing an equity value built on TOTAL enterprise
economics by ONE CLASS's share count inflates every per-share figure by the
ratio between them.

Nothing checked. `price x shares ~= market cap` is a one-line identity that
holds for every ordinary company, and it was never evaluated -- so the run
reported COMPLETE data, a VALID DCF, and a per-share value that was wrong by
a factor of four.

WHAT THIS MODULE DOES
=====================
It keeps the share-count TYPES separate (section 17), aggregates share
classes only on a stated economic basis (section 18), applies a depositary
ratio when one exists (section 19), and then RECONCILES the independent
figures against each other (sections 20-23). A material mismatch is reported
as a status, never resolved by picking a favourite.

The distinction that does the work: a share count is not one number. Issued,
treasury, outstanding, weighted-average basic, weighted-average diluted and
economic-outstanding are six different quantities, they are correct for
different purposes, and substituting one for another is invisible in the
output but changes every per-share result.

Nothing here is issuer-specific. Share classes, depositary ratios and
parent/subsidiary relationships are read from filing and provider metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Deterministic guard codes (sections 20-23).
MARKET_CAP_RECONCILIATION_FAILURE = "MARKET_CAP_RECONCILIATION_FAILURE"
PE_RECONCILIATION_FAILURE = "PE_RECONCILIATION_FAILURE"
PB_RECONCILIATION_FAILURE = "PB_RECONCILIATION_FAILURE"
SHARE_BASIS_INCOMPATIBLE = "SHARE_BASIS_INCOMPATIBLE"
ENTITY_SCOPE_MISMATCH = "ENTITY_SCOPE_MISMATCH"

# How far two independent derivations of the same quantity may differ before
# the difference is a finding.
#
# MINOR is set at 2% because a current share count and a provider market
# capitalisation are struck at different moments of the same day and drift by
# buybacks and option exercises; a real company routinely lands inside it.
# MATERIAL is set at 10% because beyond that the two figures are not the same
# quantity measured twice -- they are different quantities, and the run needs
# to say so rather than average them.
MINOR_DIFFERENCE_TOLERANCE = 0.02
MATERIAL_DIFFERENCE_TOLERANCE = 0.10


class ShareCountType:
    """Section 17. Six different quantities, none interchangeable."""

    ISSUED = "issued"
    TREASURY = "treasury"
    CURRENT_OUTSTANDING = "current_outstanding"
    WEIGHTED_AVERAGE_BASIC = "weighted_average_basic"
    WEIGHTED_AVERAGE_DILUTED = "weighted_average_diluted"
    # Outstanding across every class, restated onto one economic basis and
    # onto the traded security's units (ADR ratio applied where relevant).
    ECONOMIC_OUTSTANDING = "economic_outstanding"
    ALL = (ISSUED, TREASURY, CURRENT_OUTSTANDING, WEIGHTED_AVERAGE_BASIC,
           WEIGHTED_AVERAGE_DILUTED, ECONOMIC_OUTSTANDING)


class ReconciliationStatus:
    """Section 20."""

    RECONCILED = "RECONCILED"
    MINOR_DIFFERENCE = "MINOR_DIFFERENCE"
    MATERIAL_DIFFERENCE = "MATERIAL_DIFFERENCE"
    INCOMPATIBLE_BASIS = "INCOMPATIBLE_BASIS"
    UNKNOWN = "UNKNOWN"
    ALL = (RECONCILED, MINOR_DIFFERENCE, MATERIAL_DIFFERENCE, INCOMPATIBLE_BASIS,
           UNKNOWN)

    _RANK = {RECONCILED: 0, UNKNOWN: 1, MINOR_DIFFERENCE: 2,
             MATERIAL_DIFFERENCE: 3, INCOMPATIBLE_BASIS: 4}

    @classmethod
    def worst(cls, statuses) -> str:
        found = [s for s in statuses if s in cls._RANK]
        return max(found, key=lambda s: cls._RANK[s]) if found else cls.UNKNOWN


# Causes a material share-count difference is normally attributable to
# (section 20). Reported as CANDIDATES, never asserted: this module can see
# that two numbers disagree, and cannot see which of these produced it.
DIFFERENCE_CAUSES = (
    "a stock split or share-class conversion not yet reflected by one source",
    "an ADR/ordinary-share depositary ratio applied by one source and not the other",
    "more than one share class, counted by one source and not the other",
    "a weighted-average count compared against a current point-in-time count",
    "an offering or buyback between the two measurement dates",
    "a corporate reorganization with predecessor and successor share counts",
    "a provider data error",
)


@dataclass(frozen=True)
class ShareClass:
    """One class of stock, with its own economics."""

    label: str
    shares_outstanding: Optional[float]
    # Economic claim per share relative to the traded/reference class. 1.0 for
    # an ordinary second class with identical economics; a genuinely
    # different claim is stated explicitly, never assumed.
    economic_multiplier: float = 1.0
    votes_per_share: Optional[float] = None
    is_traded: bool = False
    evidence_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {"label": self.label, "shares_outstanding": self.shares_outstanding,
                "economic_multiplier": self.economic_multiplier,
                "votes_per_share": self.votes_per_share, "is_traded": self.is_traded,
                "evidence_id": self.evidence_id}


@dataclass(frozen=True)
class IssuerEntity:
    """The legal issuer (section 15)."""

    entity_id: str
    name: Optional[str] = None
    cik: Optional[str] = None
    country_of_incorporation: Optional[str] = None
    reporting_status: Optional[str] = None       # domestic_registrant | foreign_private_issuer
    taxonomy: Optional[str] = None
    parent_entity_id: Optional[str] = None
    predecessor_entity_id: Optional[str] = None
    successor_entity_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id, "name": self.name, "cik": self.cik,
            "country_of_incorporation": self.country_of_incorporation,
            "reporting_status": self.reporting_status, "taxonomy": self.taxonomy,
            "parent_entity_id": self.parent_entity_id,
            "predecessor_entity_id": self.predecessor_entity_id,
            "successor_entity_id": self.successor_entity_id,
        }


@dataclass(frozen=True)
class SecurityIdentity:
    """The security actually being analyzed (sections 15, 19).

    `depositary_ratio` is ordinary shares per traded unit. 1.0 for an ordinary
    listed share. For a depositary receipt representing two ordinary shares it
    is 2.0, and a share count expressed in ordinary shares must be DIVIDED by
    it before being multiplied by the receipt's price -- section 19's rule,
    which exists because doing it the other way round is silent and produces a
    market capitalisation that is wrong by exactly the ratio.
    """

    ticker: str
    entity_id: str
    exchange: Optional[str] = None
    security_type: str = "common_stock"          # common_stock | adr | gdr
    depositary_ratio: float = 1.0
    currency: str = "USD"
    share_classes: Tuple[ShareClass, ...] = ()

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "entity_id": self.entity_id,
            "exchange": self.exchange, "security_type": self.security_type,
            "depositary_ratio": self.depositary_ratio, "currency": self.currency,
            "share_classes": [c.to_dict() for c in self.share_classes],
        }

    @property
    def is_depositary_receipt(self) -> bool:
        return self.security_type in ("adr", "gdr") or self.depositary_ratio != 1.0


@dataclass
class ShareCountSet:
    """Every share count that was found, kept apart (section 17)."""

    counts: Dict[str, float] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    as_of: Dict[str, Optional[str]] = field(default_factory=dict)
    evidence_ids: Dict[str, str] = field(default_factory=dict)

    def set(self, kind: str, value: Optional[float], source: str,
            as_of: Optional[str] = None, evidence_id: Optional[str] = None) -> None:
        if value is None or value <= 0:
            return
        self.counts[kind] = float(value)
        self.sources[kind] = source
        self.as_of[kind] = as_of
        if evidence_id:
            self.evidence_ids[kind] = evidence_id

    def get(self, kind: str) -> Optional[float]:
        return self.counts.get(kind)

    def to_dict(self) -> dict:
        return {"counts": dict(self.counts), "sources": dict(self.sources),
                "as_of": dict(self.as_of), "evidence_ids": dict(self.evidence_ids)}


def economic_shares_outstanding(security: SecurityIdentity) -> Tuple[Optional[float], str]:
    """Total shares on ONE economic basis, in the TRADED security's units.

    Section 18: classes are aggregated only when each one's economic claim is
    stated. A class whose multiplier is unknown makes the total unknown --
    returning the traded class alone would be the exact substitution that
    produced a 76% market-capitalisation error.

    Section 19: the result is then converted into the traded unit by the
    depositary ratio, so it can be multiplied by the traded price.
    """
    if not security.share_classes:
        return None, "no share-class detail was available"
    missing = [c.label for c in security.share_classes if c.shares_outstanding is None]
    if missing:
        return None, (f"share classes {', '.join(missing)} have no reported outstanding "
                      "count, so a total economic share count cannot be built")
    total = sum(c.shares_outstanding * c.economic_multiplier
                for c in security.share_classes)
    if security.depositary_ratio and security.depositary_ratio != 1.0:
        total = total / security.depositary_ratio
        return total, (f"{len(security.share_classes)} share class(es) aggregated on their "
                       f"stated economic basis, then divided by the depositary ratio of "
                       f"{security.depositary_ratio:g} to express the total in traded units")
    return total, (f"{len(security.share_classes)} share class(es) aggregated on their "
                   "stated economic basis")


def _classify_difference(left: float, right: float) -> Tuple[str, float]:
    if right == 0:
        return ReconciliationStatus.UNKNOWN, 0.0
    gap = (left - right) / abs(right)
    if abs(gap) <= MINOR_DIFFERENCE_TOLERANCE:
        return ReconciliationStatus.RECONCILED, gap
    if abs(gap) <= MATERIAL_DIFFERENCE_TOLERANCE:
        return ReconciliationStatus.MINOR_DIFFERENCE, gap
    return ReconciliationStatus.MATERIAL_DIFFERENCE, gap


def _finding(code: str, severity: str, message: str, **detail) -> dict:
    entry = {"code": code, "severity": severity, "message": message}
    entry.update(detail)
    return entry


@dataclass
class ShareReconciliation:
    """The result of comparing every independent share/market-cap figure."""

    status: str = ReconciliationStatus.UNKNOWN
    selected_basis: Optional[str] = None
    selected_value: Optional[float] = None
    selected_reason: str = ""
    implied_market_cap: Optional[float] = None
    reported_market_cap: Optional[float] = None
    market_cap_gap: Optional[float] = None
    comparisons: List[dict] = field(default_factory=list)
    findings: List[dict] = field(default_factory=list)

    @property
    def is_material(self) -> bool:
        return self.status in (ReconciliationStatus.MATERIAL_DIFFERENCE,
                               ReconciliationStatus.INCOMPATIBLE_BASIS)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "selected_basis": self.selected_basis,
            "selected_value": self.selected_value,
            "selected_reason": self.selected_reason,
            "implied_market_cap": self.implied_market_cap,
            "reported_market_cap": self.reported_market_cap,
            "market_cap_gap": self.market_cap_gap,
            "comparisons": [dict(c) for c in self.comparisons],
            "findings": [dict(f) for f in self.findings],
        }


def reconcile_share_basis(counts: ShareCountSet, security: SecurityIdentity,
                          price: Optional[float] = None,
                          reported_market_cap: Optional[float] = None
                          ) -> ShareReconciliation:
    """Sections 20-21: choose a share basis and prove it against market cap.

    The market-capitalisation identity is the check that matters, because it
    is the only one that uses a number the system did not derive. A provider
    reports both a price and a market capitalisation; if the chosen share
    count does not reconcile the two, the count is not the one the market is
    pricing, whatever else it may be correct for.

    A material mismatch does NOT get resolved here. The candidates are ranked
    by how well they reconcile, the best is selected so the run can continue,
    and the disagreement is reported so readiness and confidence can reflect
    it (section 56: fail closed, not silently).
    """
    result = ShareReconciliation(reported_market_cap=reported_market_cap)

    economic, economic_reason = economic_shares_outstanding(security)
    if economic is not None:
        counts.set(ShareCountType.ECONOMIC_OUTSTANDING, economic, "share_class_aggregation")

    # Candidate bases for "what the market is pricing", best first. Weighted-
    # average counts are deliberately last: they are period AVERAGES and are
    # correct for EPS, not for a current market capitalisation.
    preference = (
        ShareCountType.ECONOMIC_OUTSTANDING,
        ShareCountType.CURRENT_OUTSTANDING,
        ShareCountType.WEIGHTED_AVERAGE_DILUTED,
        ShareCountType.WEIGHTED_AVERAGE_BASIC,
    )
    candidates = [(kind, counts.get(kind)) for kind in preference
                  if counts.get(kind) is not None]
    if not candidates:
        result.status = ReconciliationStatus.UNKNOWN
        result.selected_reason = "no share count of any kind was available"
        return result

    # -- pairwise comparison, for the record --------------------------------
    for index, (kind, value) in enumerate(candidates):
        for other_kind, other_value in candidates[index + 1:]:
            status, gap = _classify_difference(value, other_value)
            result.comparisons.append({
                "left": kind, "left_value": value,
                "right": other_kind, "right_value": other_value,
                "gap": round(gap, 6), "status": status,
            })

    # -- the market-capitalisation identity ---------------------------------
    if price and reported_market_cap:
        scored = []
        for kind, value in candidates:
            implied = value * price
            status, gap = _classify_difference(implied, reported_market_cap)
            scored.append((abs(gap), kind, value, implied, status, gap))
        scored.sort()
        _abs_gap, kind, value, implied, status, gap = scored[0]
        result.selected_basis = kind
        result.selected_value = value
        result.implied_market_cap = implied
        result.market_cap_gap = round(gap, 6)
        result.status = status
        result.selected_reason = (
            f"{kind} reconciles best against the reported market capitalisation "
            f"({implied:,.0f} implied vs {reported_market_cap:,.0f} reported, "
            f"{gap:+.1%})")
        if status in (ReconciliationStatus.MATERIAL_DIFFERENCE,
                      ReconciliationStatus.INCOMPATIBLE_BASIS):
            result.findings.append(_finding(
                MARKET_CAP_RECONCILIATION_FAILURE, "error",
                f"No available share count reconciles price x shares against the reported "
                f"market capitalisation. The closest, {kind} ({value:,.0f} shares), implies "
                f"{implied:,.0f} against a reported {reported_market_cap:,.0f} "
                f"({gap:+.1%}). Every per-share figure in this analysis depends on which "
                f"count is correct. Candidate causes: {'; '.join(DIFFERENCE_CAUSES[:4])}.",
                selected_basis=kind, selected_value=value, implied_market_cap=implied,
                reported_market_cap=reported_market_cap, gap=round(gap, 6)))
        return result

    # -- no market cap to check against: fall back to internal agreement ----
    kind, value = candidates[0]
    result.selected_basis = kind
    result.selected_value = value
    result.selected_reason = (
        f"{kind} selected by precedence; no reported market capitalisation was available "
        "to check price x shares against")
    worst = ReconciliationStatus.worst([c["status"] for c in result.comparisons])
    result.status = worst if result.comparisons else ReconciliationStatus.UNKNOWN
    if worst == ReconciliationStatus.MATERIAL_DIFFERENCE:
        result.findings.append(_finding(
            SHARE_BASIS_INCOMPATIBLE, "warning",
            "Available share counts disagree materially and there is no reported market "
            "capitalisation to arbitrate between them; per-share figures carry that "
            "uncertainty.",
            comparisons=result.comparisons))
    return result


# ---------------------------------------------------------------------------
# Sections 22-24 — per-share invariants
# ---------------------------------------------------------------------------


def reconcile_price_earnings(price: Optional[float], diluted_eps: Optional[float],
                             market_cap: Optional[float],
                             attributable_net_income: Optional[float]) -> dict:
    """Section 22: P/E derived two independent ways must agree.

    price / diluted EPS uses per-share figures; market cap / attributable net
    income uses aggregate ones. They are the same ratio when the share basis
    and the earnings attribution are compatible, and they diverge exactly when
    one of those is wrong -- which is why the check is worth running rather
    than computing P/E once and printing it.

    `attributable_net_income` must be the earnings attributable to the common
    shareholders of the analyzed security (section 24). Consolidated net
    income including a material noncontrolling interest is a different
    numerator and will not reconcile.
    """
    record = {"per_share_pe": None, "aggregate_pe": None, "status":
              ReconciliationStatus.UNKNOWN, "gap": None, "findings": []}
    if price and diluted_eps and diluted_eps > 0:
        record["per_share_pe"] = price / diluted_eps
    if market_cap and attributable_net_income and attributable_net_income > 0:
        record["aggregate_pe"] = market_cap / attributable_net_income
    if record["per_share_pe"] is None or record["aggregate_pe"] is None:
        # Negative or absent earnings make P/E not meaningful rather than
        # wrong; that is a rendering decision, not a reconciliation failure.
        record["status"] = ReconciliationStatus.UNKNOWN
        return record
    status, gap = _classify_difference(record["per_share_pe"], record["aggregate_pe"])
    record["status"] = status
    record["gap"] = round(gap, 6)
    if status == ReconciliationStatus.MATERIAL_DIFFERENCE:
        record["findings"].append(_finding(
            PE_RECONCILIATION_FAILURE, "warning",
            f"P/E from price and diluted EPS ({record['per_share_pe']:.1f}) disagrees with "
            f"P/E from market capitalisation and attributable net income "
            f"({record['aggregate_pe']:.1f}), a {gap:+.1%} difference. The two use different "
            "share and earnings bases; a single precise P/E is not rendered until the basis "
            "difference is resolved.",
            per_share_pe=record["per_share_pe"], aggregate_pe=record["aggregate_pe"],
            gap=round(gap, 6)))
    return record


def reconcile_price_book(price: Optional[float], shares: Optional[float],
                         attributable_equity: Optional[float],
                         total_equity: Optional[float] = None,
                         minority_interest: Optional[float] = None) -> dict:
    """Section 23: P/B must use equity attributable to the common shareholders.

    Total consolidated equity includes any noncontrolling interest, which
    belongs to somebody else. For an issuer with a material NCI the two
    denominators differ by exactly that amount, and using the wrong one
    understates P/B on a company whose subsidiaries are partly owned by
    others.
    """
    record = {"price_to_book": None, "equity_basis": None, "status":
              ReconciliationStatus.UNKNOWN, "findings": []}
    equity = attributable_equity
    basis = "equity_attributable_to_parent"
    if equity is None and total_equity is not None:
        if minority_interest:
            equity = total_equity - minority_interest
            basis = "total_equity_less_noncontrolling_interest"
        else:
            equity = total_equity
            basis = "total_equity"
    if not price or not shares or not equity or equity <= 0:
        return record
    record["equity_basis"] = basis
    record["price_to_book"] = (price * shares) / equity
    record["status"] = ReconciliationStatus.RECONCILED
    if minority_interest and total_equity and total_equity > 0:
        share_of_equity = abs(minority_interest) / total_equity
        if share_of_equity > 0.05 and basis == "total_equity":
            record["status"] = ReconciliationStatus.MATERIAL_DIFFERENCE
            record["findings"].append(_finding(
                PB_RECONCILIATION_FAILURE, "warning",
                f"Price-to-book was computed against TOTAL equity while a noncontrolling "
                f"interest of {minority_interest:,.0f} ({share_of_equity:.1%} of equity) "
                "belongs to other shareholders. The ratio overstates the book value backing "
                "the analyzed security.",
                minority_interest=minority_interest, total_equity=total_equity))
    return record


# ---------------------------------------------------------------------------
# Sections 25-26 — corporate actions belong to an entity
# ---------------------------------------------------------------------------


def validate_corporate_action_scope(action: dict, security: SecurityIdentity,
                                    issuer: IssuerEntity) -> Tuple[bool, Optional[str]]:
    """Does this corporate action belong to the security being analyzed?

    Section 25/26. A subsidiary's dividend is a real dividend declared by a
    real issuer, and it is not the parent stock's dividend. Where an action
    carries its own issuer/ticker metadata it must match; where it carries
    none it is accepted as the analyzed security's own, because that is what
    a per-ticker provider feed means -- accepting it is the documented
    default, and the absence of scope metadata is recorded rather than
    treated as a match.
    """
    declared_ticker = (action.get("ticker") or action.get("symbol") or "").strip().upper()
    declared_entity = (action.get("entity_id") or action.get("cik") or "").strip()

    if declared_ticker and declared_ticker != security.ticker.upper():
        return False, (f"this action was declared for {declared_ticker}, not "
                       f"{security.ticker}; a corporate action of another security is never "
                       "attributed to the analyzed one")
    if declared_entity and issuer.cik and str(declared_entity).lstrip("0") \
            != str(issuer.cik).lstrip("0"):
        return False, (f"this action was declared by entity {declared_entity}, not by "
                       f"{issuer.name or issuer.entity_id} (CIK {issuer.cik})")
    return True, None


def validate_dividend(dividend: dict, security: SecurityIdentity,
                      issuer: IssuerEntity) -> Tuple[bool, Optional[str]]:
    """Section 26: a dividend enters shareholder yield only if it is this
    security's, and only if its dates and currency are usable.

    The currency check is not pedantry: a dividend declared in the issuer's
    home currency against a price quoted in the listing currency produces a
    yield that is wrong by the exchange rate, and for a depositary receipt
    the per-share amount is also on the wrong unit basis.
    """
    in_scope, reason = validate_corporate_action_scope(dividend, security, issuer)
    if not in_scope:
        return False, reason
    amount = dividend.get("amount") or dividend.get("dividend")
    if amount is None:
        return False, "the dividend record carries no amount"
    currency = (dividend.get("currency") or security.currency or "USD").upper()
    if currency != (security.currency or "USD").upper():
        return False, (f"the dividend is declared in {currency} but the security is quoted "
                       f"in {security.currency}; converting it is not attempted, so it is "
                       "excluded from yield rather than mixed")
    if not (dividend.get("ex_date") or dividend.get("payment_date")
            or dividend.get("date")):
        return False, "the dividend record carries no ex-date or payment date"
    return True, None


# ---------------------------------------------------------------------------
# Building the identity from what the providers actually returned
# ---------------------------------------------------------------------------

# dei concepts carrying per-class outstanding counts. companyfacts strips the
# dimensional axis, so several classes collapse onto one concept and only the
# COUNT of distinct same-date values reveals that more than one class exists.
_DEI_SHARE_CONCEPTS = ("EntityCommonStockSharesOutstanding",)


def detect_share_classes(company_facts: dict, as_of: Optional[str] = None
                         ) -> Tuple[List[ShareClass], Optional[str]]:
    """Share classes from dei cover-page counts, or a stated reason.

    Returns classes only when the evidence is unambiguous. companyfacts
    discards the dimensional axis that names each class, so when several
    distinct counts share one date this function reports that a multi-class
    structure EXISTS without inventing labels or economics for it -- which is
    exactly the information the market-capitalisation check needs, and the
    only claim the data supports.
    """
    dei = (company_facts or {}).get("facts", {}).get("dei") or {}
    rows: List[dict] = []
    for concept in _DEI_SHARE_CONCEPTS:
        entry = dei.get(concept) or {}
        for unit_rows in (entry.get("units") or {}).values():
            if isinstance(unit_rows, list):
                rows.extend(unit_rows)
    if not rows:
        return [], "no cover-page share count was reported"

    latest_end = as_of or max((r.get("end") or "") for r in rows)
    same_date = [r for r in rows if r.get("end") == latest_end and r.get("val")]
    if not same_date:
        return [], f"no cover-page share count was reported at {latest_end}"

    distinct = sorted({float(r["val"]) for r in same_date}, reverse=True)
    if len(distinct) == 1:
        return [ShareClass(label="common", shares_outstanding=distinct[0],
                           economic_multiplier=1.0, is_traded=True,
                           evidence_id="dei.EntityCommonStockSharesOutstanding")], None
    return ([ShareClass(label=f"class_{index + 1}", shares_outstanding=value,
                        economic_multiplier=1.0, is_traded=index == 0,
                        evidence_id="dei.EntityCommonStockSharesOutstanding")
             for index, value in enumerate(distinct)],
            f"{len(distinct)} distinct cover-page share counts were reported at "
            f"{latest_end}; companyfacts does not carry the class axis, so each class's "
            "economic rights are assumed equal until stated otherwise")


def build_issuer_entity(company_facts: dict, submissions: Optional[dict],
                        cik: Optional[str], name: Optional[str]) -> IssuerEntity:
    from finance import taxonomy as taxonomy_module

    country = None
    reporting_status = "domestic_registrant"
    if submissions:
        country = (submissions.get("stateOfIncorporation")
                   or submissions.get("stateOfIncorporationDescription"))
        forms = set((submissions.get("filings") or {}).get("recent", {}).get("form") or [])
        if forms & {"20-F", "40-F", "6-K"}:
            reporting_status = "foreign_private_issuer"
    return IssuerEntity(
        entity_id=str(cik or name or "unknown"),
        name=name or (submissions or {}).get("name"),
        cik=cik,
        country_of_incorporation=country,
        reporting_status=reporting_status,
        taxonomy=taxonomy_module.detect_taxonomy(company_facts),
    )
