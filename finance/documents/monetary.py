"""Currency-independent monetary scale -- Phase H.22.

A monetary amount has THREE independent axes, never conflated:

  - VALUE: the printed number itself ("500")
  - CURRENCY: the unit of account (USD, EUR, GBP, JPY, ...)
  - SCALE: the printed magnitude word (thousand / million / billion / bare)

THE DEFECT THIS MODULE FIXES

Phase H.16-H.21's event reader borrowed `finance.extraction.schema`'s model
unit vocabulary (USD / USD_MILLION / USD_BILLION), which bakes currency and
scale into single tokens -- a deliberate, documented simplification for
guidance figures, which are overwhelmingly USD-denominated. Reused
unmodified for financing EVENTS, that coupling produced a live, H.21-traced
defect: a EUR-denominated notes issuance ("EUR500 million") gave the model
no scale-bearing unit token that wasn't also a false USD claim, so it chose
`unit=UNKNOWN` (a documented, instructed fallback for "no span states a
currency") while still correctly reporting `currency=EUR` in the separate
field. `EventCandidateValidator._scale` silently mapped the unrecognised
unit to a x1 multiplier -- UNIT scale, never independently checked against
the source's own "million" -- and the validator ACCEPTED a resolved event
of 500.0 EUR where the filing stated 500,000,000 EUR. Not a benchmark
scoring miss: the deterministic acceptance boundary itself published a
million-x magnitude error.

THE FIX

`MonetaryScale` decouples scale from currency entirely, with its own
grounding language (thousand/million/billion must appear in the cited
evidence, exactly as an amount ROLE must -- see `event_validator.py`'s
`_ROLE_LANGUAGE`) and its own multiplier table, independent of whatever
currency the amount is denominated in. `UNKNOWN` is a legitimate reader
answer -- "I found a currency amount but the source doesn't make its scale
clear" -- and, like every other UNKNOWN in this pipeline, is never silently
treated as any other value: for a currency-domain amount, UNKNOWN (or
unstated) scale refuses the candidate rather than defaulting to UNIT.

This is the ONE central scale mechanism (spec section 2): both the
validator's normalization and any benchmark/scorer code that needs to
replicate it import from here, rather than each keeping its own
scale-multiplier table (which is how the original defect went unnoticed --
`event_validator.EventCandidateValidator._scale` and
`scripts/run_live_document_pipeline_benchmark.py`'s own scaling helper were
two separate, silently-divergent implementations before this phase).
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from finance import guidance as gm
from finance.extraction.schema import domain_unit
from finance.extraction.validator import _NUMBER as _MENTION_NUMBER


class MonetaryScale:
    """UNIT / THOUSAND / MILLION / BILLION are the only real scales.

    UNKNOWN is a legitimate, PASSING reader answer for the same reason
    `EventAmountRole.UNKNOWN` is (see that class's docstring): it is never
    silently treated as any other value. For a currency-domain amount, an
    UNKNOWN or unstated scale refuses the candidate
    (`SCALE_NOT_GROUNDED`) rather than defaulting to UNIT.
    """

    UNIT = "UNIT"
    THOUSAND = "THOUSAND"
    MILLION = "MILLION"
    BILLION = "BILLION"
    UNKNOWN = "UNKNOWN"

    ALL = (UNIT, THOUSAND, MILLION, BILLION, UNKNOWN)
    # Scales a reader can actually GROUND (excludes UNKNOWN, which is a
    # non-answer, not a groundable claim).
    STATED = (UNIT, THOUSAND, MILLION, BILLION)


SCALE_MULTIPLIERS = {
    MonetaryScale.UNIT: 1.0,
    MonetaryScale.THOUSAND: 1_000.0,
    MonetaryScale.MILLION: 1_000_000.0,
    MonetaryScale.BILLION: 1_000_000_000.0,
}

# Language that STATES a scale word -- the same discipline as
# `event_validator._ROLE_LANGUAGE`: a scale asserted must be independently
# supported by the source's own wording, never inferred from the bare value
# being merely plausible at that magnitude. UNIT has no required language by
# design -- the ABSENCE of a scale word is exactly what UNIT means (spec
# section 5's "EUR500 without a scale word" case).
SCALE_LANGUAGE = {
    MonetaryScale.THOUSAND: re.compile(r"(?i)\bthousands?\b"),
    MonetaryScale.MILLION: re.compile(r"(?i)\bmillions?\b"),
    MonetaryScale.BILLION: re.compile(r"(?i)\bbillions?\b"),
}

# A small, deliberately explicit set of currency-recognition patterns --
# symbol, ISO code, and common name -- for the currencies this phase's spec
# names (USD/EUR/GBP/JPY). An ISO code NOT in this table still passes the
# bare format check (`event_validator._ISO_CURRENCY`) but skips language
# grounding -- permissive for the long tail, exactly like
# `EventAmountRole.OTHER`'s "I can tell it's material but not which named
# role" convention: absent a curated pattern, refusing every non-listed
# currency would be a false-negative machine, not a safety improvement.
CURRENCY_LANGUAGE = {
    "USD": re.compile(r"(?i)\$|\bUSD\b|\bU\.S\.\s+dollars?\b|\bdollars?\b"),
    "EUR": re.compile(r"(?i)€|\bEUR\b|\beuros?\b"),
    "GBP": re.compile(r"(?i)£|\bGBP\b|\bpounds?(?:\s+sterling)?\b|\bpence\b"),
    "JPY": re.compile(r"(?i)¥|\bJPY\b|\byen\b"),
}

# Domains where "scale" is not a meaningful concept at all -- a per-share
# price, a ratio, a bare share count. Only a CURRENCY-domain amount (a
# scaled total, however denominated) needs its scale grounded.
_NON_CURRENCY_DOMAINS = {
    gm.GuidanceUnit.CURRENCY_PER_SHARE, gm.GuidanceUnit.RATIO, gm.GuidanceUnit.SHARES,
}


def is_currency_amount(unit: Optional[str], currency: Optional[str]) -> bool:
    """Whether this amount is a scaled-currency total needing scale
    grounding -- as opposed to a per-share price, ratio, or share count
    (which have a currency-shaped unit sometimes, like $95.00/share, but no
    THOUSAND/MILLION/BILLION scale concept to ground)."""
    domain, _legacy_scale = domain_unit(unit)
    if domain in _NON_CURRENCY_DOMAINS:
        return False
    if domain == gm.GuidanceUnit.CURRENCY:
        return True
    # `unit` didn't resolve to a known domain (None, "UNKNOWN", or any
    # value `domain_unit` doesn't recognise) -- a currency code being
    # independently stated is itself evidence this is a monetary amount.
    return bool(currency)


# The pre-H.22 unit tokens that baked scale into the unit name itself.
# Deliberately NOT including "CURRENCY" (the new, scale-neutral token) --
# unlike bare "USD", which pre-H.22 candidates used precisely to mean "a
# currency amount with no scale word" (spec section 2's "if required"
# compatibility), "CURRENCY" carries no scale implication at all. A
# candidate using the new token must state `scale` explicitly; falling
# back to UNIT for it would silently recreate the exact defect this phase
# fixes, just for the new vocabulary instead of the old one.
_LEGACY_SCALE_UNITS = {"USD", "USD_MILLION", "USD_BILLION"}


def scale_from_legacy_unit(unit: Optional[str]) -> Optional[str]:
    """Backward compatibility (spec section 2): a pre-H.22 candidate whose
    `unit` baked in scale (USD_MILLION / USD_BILLION / bare USD) is read as
    if it had stated that scale explicitly -- through this ONE function,
    never a second parallel scale table. Returns None for anything that is
    not one of those specific legacy tokens, so neither a PER_SHARE/RATIO/
    SHARES amount nor the new scale-neutral "CURRENCY" token is ever
    mistaken for an implicitly-UNIT-scaled legacy amount."""
    unit_upper = (unit or "").upper()
    if unit_upper not in _LEGACY_SCALE_UNITS:
        return None
    _domain, legacy_scale = domain_unit(unit_upper)
    if legacy_scale == "million":
        return MonetaryScale.MILLION
    if legacy_scale == "billion":
        return MonetaryScale.BILLION
    return MonetaryScale.UNIT   # bare legacy "USD": stated, no scale word


def resolve_scale(unit: Optional[str], scale: Optional[str]) -> Optional[str]:
    """The ONE place a candidate's effective scale is decided. Prefers an
    explicitly stated `scale`; falls back to a legacy unit's baked-in
    scale; returns None (never UNIT) when nothing states one -- the caller
    must treat None as "not grounded", never as "assume UNIT"."""
    stated = (scale or "").upper()
    if stated in MonetaryScale.STATED:
        return stated
    return scale_from_legacy_unit(unit)


def scale_multiplier(scale: Optional[str]) -> float:
    return SCALE_MULTIPLIERS.get((scale or "").upper(), 1.0)


# ---------------------------------------------------------------------------
# Phase H.30 -- monetary MENTION binding
# ---------------------------------------------------------------------------
#
# THE DEFECT THIS SECTION FIXES
#
# H.28's live benchmark found a second, distinct magnitude defect, sitting
# one level below the one H.22 fixed above. Given cited evidence containing
# TWO monetary mentions in one span -- "...a new $2.0 billion multicurrency
# revolving credit facility..., of which up to EUR500 million is available
# for borrowings..." -- a candidate claiming value=2.0, currency=USD,
# scale=MILLION was ACCEPTED, because the validator's currency/scale checks
# (just above: `currency_pattern.search(resolved_text)` /
# `scale_pattern.search(resolved_text)`) only asked "does this LANGUAGE
# exist ANYWHERE in the cited text", never "does it belong to THIS NUMBER".
# "million" is genuinely present in that span -- it just modifies EUR500,
# not $2.0. H.22 made currency and scale independent of EACH OTHER; this
# section makes each of them independent of every OTHER MONETARY MENTION
# in the same evidence.
#
# THE FIX: A BOUNDED WINDOW PER NUMERIC MENTION, NOT A SECOND PARSER
#
# `resolved_text` (every span an event/amount cited, already concatenated
# in document order by `finance.documents.spans.resolve_span_ids`) is
# partitioned by every numeric token in it -- not just currency-shaped
# ones; a share count or a percentage is just as valid a boundary, because
# excluding it would let a scale/currency word that belongs to THAT number
# leak into a neighboring monetary mention's window instead. Each numeric
# mention OWNS the text from its immediate predecessor number (exclusive)
# to its immediate successor number (exclusive). A currency or scale
# pattern must match INSIDE the window of an occurrence of the CLAIMED
# value to ground that claim; matching only in a different mention's
# window is exactly the H.28 defect and is now refused.
#
# This reuses `finance.extraction.validator`'s own number-detection regex
# (imported, not restated) and needs no new offsets/span-id tracking: the
# already-existing, already-trusted `resolved_text` string is enough, since
# window boundaries are just OTHER matches of that SAME regex within it.
# `MonetaryMentionWitness` is deliberately the smallest representation that
# makes the invariant checkable -- a value and the exact span of text it
# owns -- not a parallel monetary parser.
#
# CROSS-SPAN SUPPORT IS PRESERVED, NOT REBUILT
#
# `resolved_text` already unions every span the candidate cited, in
# document order, before this function ever sees it (H.18's own design).
# So when a value and its scale word are split across two ADJACENT CITED
# spans with nothing else between them (spec section 4's structured-table
# case; a pre-existing golden fixture exercises exactly this: a value in
# one clause, its scale word in the next, joined by a semicolon and split
# into two spans by `spans.py`) -- and that value is the ONLY numeric
# mention in the combined text -- its window is the WHOLE resolved text,
# so the scale word in the adjacent span is found
# exactly as before. What changes is only the case with A SECOND numeric
# mention nearby: then, and only then, the window narrows to exclude
# language that belongs to that second number. No case that used to pass
# through legitimate cross-span association is narrowed unless a genuine
# second monetary mention is present to disambiguate against -- and where
# one is, the earlier behavior was already the defect, not a feature.
#
# WHAT THIS DELIBERATELY DOES NOT SUPPORT
#
# It does not attempt to associate a value in one CITED span with a scale
# word many spans away by inferring "meaning" -- a genuinely ambiguous
# multi-hop association fails closed via a normal SCALE_NOT_GROUNDED /
# CURRENCY_NOT_GROUNDED refusal, the same code every other ungrounded claim
# already returns, never a broadened acceptance rule.


@dataclass(frozen=True)
class MonetaryMentionWitness:
    """One numeric mention, and the bounded window of text it OWNS.

    `window_start`/`window_end` are offsets into whatever string this
    witness was built from (always `resolved_text`, the candidate's own
    cited evidence) -- from just after the PRECEDING numeric mention
    (exclusive) to just before the FOLLOWING one (exclusive), or to the
    string's own bounds when there is no neighbor on that side. A currency
    or scale pattern matching inside this window is evidence about THIS
    mention; matching only outside it is evidence about a DIFFERENT one.
    """

    value: float
    window_start: int
    window_end: int

    def window_text(self, source: str) -> str:
        return source[self.window_start:self.window_end]


def _mentions_match(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * 1e-6, 1e-9)


def monetary_mention_windows(text: str) -> List[MonetaryMentionWitness]:
    """Partition `text` into one window per numeric token in it, in
    document order. Every numeric mention -- not only ones adjacent to a
    currency symbol -- becomes a boundary, so a scale/currency word cannot
    leak past a share count, a percentage, or any other number that sits
    between two monetary mentions."""
    matches = list(_MENTION_NUMBER.finditer(text or ""))
    witnesses: List[MonetaryMentionWitness] = []
    for i, m in enumerate(matches):
        try:
            value = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        window_start = matches[i - 1].end() if i > 0 else 0
        window_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        witnesses.append(MonetaryMentionWitness(
            value=value, window_start=window_start, window_end=window_end))
    return witnesses


def mention_language_grounds_value(text: str, value: float,
                                   pattern: Optional[re.Pattern]) -> bool:
    """Whether `pattern` (a currency or scale language pattern) matches
    inside the window OWNED by some occurrence of `value` in `text` --
    never merely somewhere in `text`. This is the ONE check that replaces
    a bare `pattern.search(text)` for a currency/scale claim tied to a
    SPECIFIC amount, everywhere in the validator.

    `pattern is None` (a currency with no curated language table entry --
    see `CURRENCY_LANGUAGE`'s own docstring) is permissive, matching the
    prior behavior for that case: an uncurated currency code already
    passed its format check and skips language grounding entirely, exactly
    as before this phase.
    """
    if pattern is None:
        return True
    for witness in monetary_mention_windows(text):
        if _mentions_match(value, witness.value) and pattern.search(witness.window_text(text)):
            return True
    return False
