# Security review — SEC EDGAR filing-document access (Phase H.4)

Reviews the **new outbound network surface** added so SEC-filed management
guidance can be read: two EDGAR Archives operations, `filing_index` and
`filing_document`. Companion to `YAHOO_SEC_PROVIDER_REVIEW.md`, which covers
the three pre-existing SEC JSON endpoints and is unchanged by this phase.

## 1. Why a new surface was needed at all

Quantitative guidance ("full-year 2026 sales growth of 2% to 3%") does not
appear in SEC's XBRL company-facts API. It exists only in the earnings-release
exhibit attached to an item-2.02 8-K, which is a document under
`www.sec.gov/Archives/`. Reading it requires fetching that document.

The alternative — inferring guidance from any other source, or letting a model
supply it — was rejected. See §4.

## 2. What is reachable, exactly

Two operations, both in the same closed `_SEC_DATASETS` registry as the
existing three (`finance/sec_datasets.py`). There is **no generic SEC URL
fetcher**, and no argument through which one can be constructed.

| Operation | URL produced |
|---|---|
| `filing_index` | `https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/index.json` |
| `filing_document` | `https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<document>` |

The host is a **literal in the format string**, not an argument. `www.sec.gov`
is already on the pre-existing `_ALLOWED_HOSTS` allowlist, and `fetch()`
independently re-checks the resolved host against that allowlist before
calling `safe_get` — unchanged defence in depth from the earlier review.

Every request still goes through `tools/http_safety.py::safe_get`
(SSRF-safe, blocks private/loopback/non-routable addresses, `max_redirects=0`,
`allow_http=False`) and still carries the configured `SEC_USER_AGENT`.

## 3. Argument validation

These are the **only** SEC operations whose URL contains a caller-supplied
path *segment* rather than just a zero-padded CIK, so they are the only place
a malformed argument could try to escape the intended path. Both are validated
in `finance/sec_provider.py` **before** any string concatenation:

* `accession` — must match `^\d{10}-\d{2}-\d{6}$`. Anchored, digits and
  hyphens only. Rejects traversal segments, encoded separators, query
  strings, and fragments by construction.
* `document` — must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, must not
  contain `..`, and must end in `.htm`, `.html`, or `.txt`. EDGAR document
  names are flat filenames; no directory component is ever legitimate. The
  extension allowlist keeps the surface to text documents.

`tests/test_finance_sec_provider.py` covers 21 rejection cases including
`../../../etc/passwd`, `sub/dir/file.htm`, `..\windows\win.ini`,
`//evil.example.com/a.htm`, query/fragment injection, wrong extensions, and
an over-length name — plus a positive test asserting the resolved host is
always on the allowlist regardless of arguments.

## 4. The document is untrusted input to a parser, never to a model

This is the substantive risk in the phase, and the design point that
addresses it.

A filed exhibit is a document **the issuer wrote**. The obvious
implementation — hand it to the local model and ask for the guidance numbers —
was rejected for two reasons:

1. **Fabrication.** A model asked to find guidance in a document that contains
   none will supply some. A fabricated forward number flows straight into the
   DCF and is indistinguishable from a real one.
2. **Injection.** Text that reaches a model is text that can attempt to
   instruct it.

Instead, `finance/guidance.py` treats the document purely as parser input.
Nothing in it reaches a model. Extraction is reviewed regular expressions
only; anything not matched is **dropped**, never interpreted. Every extracted
value carries the exact excerpt supporting it, so any number in a report can
be traced back to the sentence in the filing that justifies it.

The provider marks the payload `untrusted_content: True`, consistent with
every other market-data response in this project.

Precision is bought structurally: a guidance figure must be expressed as a
**range**. Companies state guidance as ranges and actuals as single values, so
requiring two numbers separates them without interpreting anything. The cost
is recall on single-point guidance, accepted deliberately — a miss is visible
(guidance reads as unavailable, and the workflow is built to keep working that
way), a false positive is not.

## 5. Request volume and caching

Guidance ingestion costs **two additional SEC requests per analysis** beyond
the submissions index: one filing-index page and one exhibit. Both are cached
for 30 days (`ttl_seconds=30 * 24 * 3600`), which is safe because an accepted
filing is immutable — only the *list* of filings changes, and that lives in
`company_submissions` with its own 6-hour TTL.

The number of releases examined is bounded by `GUIDANCE_MAX_RELEASES`
(default 3; more than one is required so superseded guidance can be
identified as superseded rather than simply absent).

Response size is bounded by the existing `max_market_data_bytes()` limit via
`read_limited`, and the payload records `truncated` when that limit was hit.

## 6. Failure behaviour

Every guidance failure mode is **non-fatal**. A missing 8-K, a missing
exhibit, an unreachable document, a rate-limited request, or an unparseable
release all degrade to "guidance unavailable" with a warning. The workflow is
explicitly designed to run without guidance
(`finance/freshness.py`, and `GUIDANCE_INGESTION_ENABLED=false` is a supported
configuration verified by
`tests/test_finance_aos_regression.py::test_guidance_absence_does_not_fail_the_workflow`).

## 7. Not changed by this phase

* Provider routing and the Yahoo/SEC/Alpha Vantage architecture.
* The three pre-existing SEC endpoints and their URL construction.
* `_ALLOWED_HOSTS` — `www.sec.gov` was already on it.
* `tools/http_safety.py`.
* `router.py`, `tools/executor.py`.
