# H.29 — EdgarTools Direct Python Integration Spike

edgartools==5.58.0, installed into the main venv (hash-verified via the existing
`config/mcp_locks/edgartools-5.58.0.txt` from H.23). `venv/` is gitignored and
`requirements.txt` was NOT modified, so this remains a local, non-committed,
reversible addition — not a new production dependency. **Side effect to note**:
the install upgraded several shared dependencies already in the venv (pydantic
2.13.4→2.13.5, pandas 3.0.5→3.0.6, cryptography 49→50, anyio, uvicorn, certifi,
cffi, charset-normalizer). Full regression suite (4127 tests) is clean after the
bump, but this is a real side effect worth knowing about before relying on this
venv state for anything else.

## Adapter

`finance/documents/edgartools_adapter.py` — lazy-imports `edgar` only inside
functions (module import itself never requires edgartools installed, verified
directly). Exposes `EdgarToolsCompanyRef`, `EdgarToolsFilingRef`,
`EdgarToolsSectionRef`, `EdgarToolsExhibitRef` — company/CIK lookup, filing
discovery, accession/form/date/period metadata, verbatim section/exhibit text.
Never calls `.to_context()` or any formatted summary (test-enforced: a fake
`to_context()` raises `AssertionError` if ever invoked — never triggered across
17 adapter unit tests). `finance_sec_provider()` config flag added
(`tools/config.py`), default `"current"`, never selected automatically.

## Comparison results (22 tickers, fixed pre-declared list, zero cherry-picking)

22/22 completed without crashing either path. 3 findings:

| Ticker | Class | Finding |
|---|---|---|
| MSFT (+JPM) | DOCUMENT_FETCH | Current pipeline's own document-size cap (`max_market_data_bytes()`) rejected the 10-K; EdgarTools handled it. A genuine current-pipeline limitation, not an EdgarTools quality issue. |
| XOM | DISCOVERY / ACCESSION_SELECTION | **Root-caused, not guessed**: SEC's own live `company_tickers.json` currently maps XOM → CIK 2115436 "ExxonMobil Holdings Corp" (a real, very recent holding-company reorganization — confirmed via an `8-K12B` "Succession by shell company" filing in that CIK's own recent filings). Our current pipeline correctly follows this live, authoritative mapping. EdgarTools resolved the *old* CIK 34088 "EXXON MOBIL CORP" instead — its own company-resolution index is stale relative to SEC's live ticker file for this one very-recent case. Classified `CURRENT_CORRECT_EDGARTOOLS_REGRESSES` for this specific dimension. |

Latency: current pipeline mean 0.49s/ticker vs EdgarTools mean 4.58s/ticker for
discovery (≈9x slower); SAP alone took EdgarTools 24s.

## Structural problem tests (A–J)

- **A/B (inline-XBRL-heavy 8-K, large ix:header)**: MSFT's 8-K — EdgarTools'
  `.text()` returned clean, readable text (5,473 chars) with no ix:header
  markup leakage, no separate stripping step needed on our side.
- **C/D (Item 1.01/2.03, multi-item 8-K)**: XOM's 7-item 8-K
  (1.01, 2.01, 3.01, 3.03, 5.02, 5.03, 9.01) — EdgarTools' `.sections`
  correctly, individually segmented **all seven** items plus Signatures, each
  with clean verbatim text and a human-readable title. Item codes agreed with
  SEC's own submissions.json `items` field for every one of the 22 filings
  compared (zero item-code disagreements recorded).
- **E (earnings-release exhibit retrieval)**: not independently deep-dived
  this phase (the comparison script's deep-dive logic preferred a
  financing-eligible 8-K over an earnings-release 8-K when an issuer had
  both) — **honest gap**, not tested.
- **F (latest 10-Q/10-K discovery)**: 20/22 domestic tickers full agreement;
  1 genuine disagreement (XOM, root-caused above).
- **G (foreign issuer 20-F/6-K)**: TSM and SONY — EdgarTools agreed with the
  current pipeline byte-for-byte on CIK, latest 20-F accession/period, and
  latest 6-K accession/period. Clean.
- **H (amended filing)**: SONY's `6-K/A` (2018) retrieved cleanly via
  EdgarTools (25,687 chars of text); `.sections` returned empty for it, but
  this matches the *current* pipeline's own behavior too (a 6-K/6-K/A carries
  no SEC-mandated item numbering to segment by) — not a defect.
- **I (multiple exhibits)**: MSFT's 13-attachment XBRL-heavy filing retrieved
  cleanly and fully via `.attachments`.
- **J (HTML tables)**: not independently byte-compared this phase — **honest
  gap**, not tested.

## Structured XBRL comparison

AAPL/UNH `us-gaap:Revenues`/`NetIncomeLoss`/`CashAndCashEquivalents...`
compared against EdgarTools' `.financials.income_statement()`. One
noteworthy, fully-explained artifact: a **naive single-concept lookup**
(`facts.get("Revenues")`, written for this spike only) found AAPL's legacy
`Revenues` XBRL tag stale (fiscal 2018) because Apple's own tagging migrated
to `RevenueFromContractWithCustomerExcludingAssessedTax` around ASC 606
adoption — a real, well-known taxonomy transition. **This is not a real
production bug**: `finance/xbrl_mapping.py` already implements the correct
multi-concept fallback chain for exactly this migration; only this spike's
own simplified diagnostic script lacked it. EdgarTools' `.financials` API
returned both old- and new-taxonomy concept names present in its own output
for UNH, suggesting it handles multi-concept resolution natively — worth a
closer look in any future phase that revisits this, but not adopted or
relied on here.

## JPM/SAP diagnostic (diagnostic only, per spec — no production fix)

Both JPM's latest 10-K and SAP's latest 20-F: **identical accession, filed
date, and reporting period** between the current pipeline and EdgarTools.
Classified `BOTH_CORRECT` at the filing-discovery layer specifically. The
originally-observed JPM/SAP "stale period" issue (from an earlier phase, not
otherwise detailed in this spike's scope) was **not reproduced** by this
test — if real, it must live downstream of filing discovery (e.g. in
`finance/freshness.py`'s canonical-period-selection logic operating on
already-fetched companyfacts), which this spike does not reach. No production
change made either way, per the phase's explicit instruction.

## Deprecation inventory

| Module | Verdict | Why |
|---|---|---|
| `finance/sec_provider.py` (SecEdgarClient) | **KEEP** | Reviewed, SSRF-hardened, zero extra dependency surface, already production-proven; XOM finding shows it's currently *more* current than EdgarTools on live ticker-mapping changes. |
| `finance/documents/package.py` | **KEEP** | Pure metadata classification over `submissions.json`; retrieval-source-agnostic, no HTML parsing, no improvement available from swapping the source. |
| `event_validator.py` / `actuals_validator.py` / `monetary.py` | **KEEP_AS_VALIDATOR** | Untouched by design; EdgarTools never reaches the acceptance boundary. |
| `finance/reported_actuals/discovery.py::select_results_document` (HTML index-table regex parsing) | **REPLACE_WITH_EDGARTOOLS candidate** | EdgarTools' `.attachments`/exhibit list gives this natively, no HTML-table regex needed; the *scoring logic* (which exhibit is the real earnings release) stays ours either way. |
| `finance/documents/text_normalization.py` (item-block splitting, ix:header stripping) | **REPLACE_WITH_EDGARTOOLS candidate / SIMPLIFY** | EdgarTools' `.sections` gives named, pre-segmented item blocks natively (verified on a 7-item 8-K); if adopted, our own ix:header-stripping regex would no longer need to run on EdgarTools-sourced text — requires the `text_representation` marker (already added) so the pipeline never double-normalizes. |
| Custom `_INDEX_ROW`/`_INDEX_CELL` HTML-table-cell regex (`discovery.py`) | **DEPRECATE_LATER** | Narrowly scoped; removable without touching the eligibility/scoring logic, contingent on the above being adopted first. |

## Recommended migration scope

**DISCOVERY_AND_TEXT** is defensible on the *evidence quality* dimension
alone (clean multi-item segmentation, clean exhibit lists, zero item-code
disagreements) — but is **not recommended yet**, because two of this
phase's own acceptance criteria (spec section 15: "without degrading...
reporting-period correctness... latency unacceptably") are not currently
met: (1) the XOM finding shows EdgarTools' company/CIK resolution can lag
SEC's own live authoritative mapping on a real reorg event — a genuine
reporting-period/entity-correctness risk, not hypothetical; (2) a ~9x
discovery-latency regression (0.5s → 4.6s mean, spiking to 24s for SAP) is
unacceptable for a synchronous path without a caching layer this phase did
not build or measure. Net recommendation: **NO_MIGRATION** for now, with the
adapter kept on file (spike-only, `FINANCE_SEC_PROVIDER` still defaulting to
`current`) as a foundation for a future phase that specifically addresses
CIK-resolution freshness and latency before reconsidering `DISCOVERY_AND_TEXT`.
