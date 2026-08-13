# Phase H.4 — DCF input freshness and forward assumptions

## The failure

Running a valuation of **A. O. Smith (AOS)** in August 2026, the DCF used
FY2025 annual balance-sheet values even though two 10-Qs had been filed since:

| | FY2025 (2025-12-31) | Q2 2026 (2026-06-30) |
|---|---|---|
| Long-term debt | $112.7M | **$598.0M** |
| Current portion of LTD | $42.3M | $39.5M |
| **Total debt** | **~$155M** | **~$637M** (4.1×) |
| Cash | $174.5M | $181.3M |

AOS financed the Leonard Valve acquisition in January 2026 — there is an 8-K
item 2.03, *Creation of a Direct Financial Obligation*, filed 2026-01-06. Six
months and two quarterly filings later, the equity bridge still subtracted
December's net debt, understating it by roughly **$482M**.

Nothing warned, and nothing was broken in the ordinary sense.
`_dcf_inputs_from_facts` read `statements["annual"]["balance_sheet"][0]`, and
FY2025 genuinely *was* the latest **annual** filing. The quarterly data was
fetched, normalized, and sat unread.

Separately, the historical revenue CAGR was copied into **every** forecast
year's growth assumption, while AOS had publicly guided to 2–3% sales growth
for 2026. The pipeline had no way to represent "management said" as a distinct
kind of evidence, so it could not weigh it against history.

## Root cause

There is no such thing as "the latest financial year". Freshness is a
**per-field** property, and different kinds of figure go stale differently:

* A **balance-sheet** field is a point in time. The newest wins outright, and
  it comes from a 10-Q far more often than a 10-K.
* A **flow** is a period. The newest twelve months wins — usually a trailing
  roll built from quarters, not the last completed fiscal year.
* **Guidance** is forward-looking. It never supersedes a reported fact; it is
  separate evidence.

A second, compounding cause: in SEC companyfacts, `fy`/`fp` describe **the
filing a fact appeared in**, not the period the fact measures. AOS's
`fy=2026 fp=Q2` bucket holds four revenue facts — 2025 and 2026, year-to-date
and discrete. Any freshness decision keyed on those tags is unsound.

## What was built

| Module | Responsibility |
|---|---|
| `finance/period_facts.py` | Facts keyed on real `start`/`end` dates. Discrete-quarter reconstruction from year-to-date columns, including Q4 from the annual figure. |
| `finance/freshness.py` | `DcfFreshnessPlanner`, `CurrentFinancialState`, TTM construction, stale guards, freshness classification. |
| `finance/guidance.py` | Deterministic extraction of SEC-filed guidance. No model reads the document. |
| `finance/forward_assumptions.py` | Per-year growth/margin paths from ranked evidence; validation of model-proposed paths. |

### Division of authority (unchanged in spirit)

```
the model may PROPOSE          -> forward_assumptions.validate_proposal
the system DECIDES             -> DcfFreshnessPlanner, validate_proposal
finance/dcf.py does the maths  -> untouched
```

The model never decides freshness and never sees a raw filing.

### Two statuses, not one

Conflating these is what let the AOS run look healthy:

* `data_completeness` — did every dataset we asked for arrive?
* `valuation_freshness` — is the DCF using the newest data it has?

AOS scored `COMPLETE` on the first and would have scored
`STALE_INPUT_WARNING` on the second.

### Assumption precedence

Near-term revenue growth: **current guidance → TTM trend → latest reported
year-over-year → historical normalized growth → configured default.**

Growth is now a **path**, not a scalar, fading from the year-1 anchor to the
scenario's own terminal growth so the explicit forecast joins the perpetuity
continuously. A flat five-year rate is not a neutral default — it asserts that
current conditions persist unchanged.

Guidance is an **input, not truth**. A forecast may sit below, inside, or
above it, but a base-case year-1 growth diverging materially from stated
guidance must carry an explicit justification or the guidance-anchored
baseline stands.

## Bugs found along the way

Each was found by following a value from where it is set to where it is read,
rather than trusting that it arrives.

1. **Verizon's long-term debt resolved to nothing.** VZ reports
   `LongTermDebtAndCapitalLeaseObligations` ($143.4B); neither mapped concept
   existed in its filings, so total debt came out as $21.8B — the current
   portion alone — against an actual ~$165B. Lease-inclusive tags added at
   lower precedence, and flagged when used.
2. **Concept selection ignored recency.** AOS's `short_term_debt` candidates
   resolved to a 2010 balance, because "first candidate with any fact" wins
   regardless of age. Now recency wins, precedence breaks ties, and a coherent
   balance-sheet date is required on top.
3. **Q4 never existed.** No company files a Q4 10-Q, so every TTM window
   spanning a year end had a hole. Reconstructed from annual minus nine-month
   year-to-date.
4. **A "TTM" that was not trailing.** AMZN's capital expenditure summed four
   contiguous 365-day quarters from **2016–2017**. Every structural check
   passed; "trailing" was the only property never tested.
5. **Clamp provenance was silently dropped.** `_validate_assumption_provenance`
   rebuilds entries from a fixed allowlist that omitted `clamped`, so the MLI
   clamp limitation could never fire. Same class of drop as the earlier
   `source_periods`/`derivation` loss.
6. **A scenario contradicting its own perpetuity.** The first fade applied the
   scenario delta to the *target* as well as the anchor, so bear's final
   forecast year assumed −1% growth while bear's perpetuity assumed +1.5%.

## Known limitation, not fixed here

`fundamental_metrics.revenue_cagr` is computed from the
`(fiscal_year, fiscal_period)`-bucketed annual history, which can **skip a
year**. Live AOS produces the series 2025, 2023, 2022, 2021, 2020 — FY2024
missing — and then divides a five-year span by four intervals: **7.25%**
against a true ~2%. The date-keyed CAGR in `forward_assumptions.py` is
correct, and states its span explicitly so the two are never confused.

Fixing the bucketed history means changing
`finance/xbrl_mapping.py::extract_statements` to key on real period end dates,
which moves every SEC-sourced fixture's annual history. Deliberately deferred
rather than folded into this phase.

## Verification

* `venv/Scripts/python.exe -m pytest tests/ -q` — 2064 passed, 3 skipped.
* `scripts/manual_verify_h4_dcf_freshness.py [SYMBOL]` — opt-in live run
  against real SEC/Yahoo, printing the full input provenance.
* `tests/fixtures/aos_regression.json` — real captured AOS data containing the
  FY2025 10-K, both 2026 10-Qs, and three successive earnings releases whose
  guidance was lowered twice.

## Not changed

Provider routing; the Yahoo/SEC/Alpha Vantage architecture; `finance/dcf.py`'s
arithmetic; evidence validation; `router.py`; `tools/executor.py`.
