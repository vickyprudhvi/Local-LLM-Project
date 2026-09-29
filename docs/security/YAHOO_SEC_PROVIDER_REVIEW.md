# Yahoo Finance (yfinance) and SEC EDGAR provider review (Phase H.3)

**Date:** 2026-08-06
**Outcome:** Both providers **approved**, with Yahoo access explicitly labelled
unofficial/personal-use and SEC EDGAR treated as the authoritative fundamentals
source for supported US companies. Neither is an MCP server — both are trusted
local `BaseTool`s, matching the existing Alpha Vantage pattern (see
`docs/security/ALPHAVANTAGE_PROVIDER_REVIEW.md`).

Every finding below was reproduced against the real package/API in this
environment (Windows, Python 3.13, this project's venv) — nothing is quoted
from a README alone.

---

## 1. Yahoo Finance via `yfinance`

| Field | Value |
|---|---|
| Package | `yfinance` |
| Pinned version | **1.5.2** |
| Wheel SHA-256 | `197fc03485c246547a5a9184956c60150ea33b6f740d877e02a97f123d5cd2b9` (verified against the PyPI-downloaded wheel) |
| License | Apache-2.0 (verified via PyPI classifiers) |
| Repository | `github.com/ranaroussi/yfinance` |
| Installed cleanly | Yes — Windows 11, Python 3.13.13, zero `pip check` conflicts, pandas 3.0.5 pulled in as a genuinely new dependency (this project had no prior pandas pin) |

### 1.1 — This is unofficial scraping, not a sanctioned API. Confirmed in source, not assumed.

Yahoo Finance has no public, authenticated, ToS-covered market-data API.
`yfinance` reverse-engineers Yahoo's own web-app endpoints
(`query1/query2.finance.yahoo.com`). Read directly from `yfinance/data.py`:

- **`curl_cffi` dependency** — impersonates a real browser's TLS fingerprint.
  This exists specifically to get past Yahoo's bot detection; it is not
  needed for a legitimate authenticated API.
- **Cookie/CSRF "crumb" dance** — `YfData._get_cookie_and_crumb()` fetches a
  session cookie from `fc.yahoo.com`, then a CSRF "crumb" token from
  `query{1,2}.finance.yahoo.com/v1/test/getcrumb`, required on every data
  request. Two different cookie strategies (`'basic'`, `'csrf'`) exist and
  the client **automatically toggles between them** when one fails
  (`_set_cookie_strategy`), then retries the whole request once.
- **Auto-accepts Yahoo's cookie-consent HTML form.** `_accept_consent_form()`
  parses the actual GDPR/CMP consent page with BeautifulSoup and
  auto-submits an "agree" POST whenever a request is redirected to
  `consent.yahoo.com` (`_is_this_consent_url` runs on every `.get()`). This
  is UI-flow automation against a real consent page, not a documented API
  contract — and confirms the "Yahoo response schema change" failure mode
  this project's spec explicitly asked to plan for is a real, live risk, not
  a hypothetical.
- **Login-cookie support exists** (`Auth.set_login_cookies`, reading real
  Yahoo-account `T`/`Y` cookies obtained by manually logging into
  `finance.yahoo.com` in a browser) — **this project will never configure
  these.** Only the fully anonymous, unauthenticated path is used, which is
  the lowest-risk personal-use posture: no real Yahoo account is ever linked
  to or re-used by this integration.

**Personal-use / ToS risk (explicitly requested deliverable):** Yahoo's own
Terms of Service restrict automated/bulk access to its consumer site;
`yfinance`'s README has historically carried the same disclaimer this review
repeats: it is intended for **personal, non-commercial research use**, may
break without notice when Yahoo changes its internal endpoints or consent
flow, and carries no support or availability guarantee. This project uses it
accordingly — anonymously, at low/bounded volume, for one user's own research,
never resold or redistributed, with a kill switch (§5) and every dataset
labelled with its actual provider so nothing is presented as an authoritative
filing fact.

### 1.2 — Local disk state yfinance maintains on its own

`yfinance/cache.py` uses `platformdirs.user_cache_dir()` (Windows:
`%LOCALAPPDATA%\py-yfinance\py-yfinance`), **outside this project's
`app_data/` convention and outside git entirely** — a timezone cache and a
`pickle`-serialized cookie cache (`cache.get_cookie_cache()`,
`data.py::_save_cookie_curlCffi`/`_load_cookie_curlCffi`). This is yfinance's
own persistence, not something this project writes to directly. Residual risk,
documented rather than hidden: pickle deserialization of a file only this
local OS user can write is a low-severity risk in this single-user desktop
context, not a multi-tenant server — noted for completeness per this
project's "honest residual limitations" convention.

### 1.3 — Networking, retries, rate limits (verified in source)

- Built-in retry with exponential backoff (`YfConfig.network.retries`,
  `2 ** attempt` sleep) on transient errors (`_is_transient_error`).
- A real 429 response raises `YFRateLimitError` — a controllable, catchable
  signal this project's retry classifier maps to a `MARKET_DATA_RATE_LIMITED`-
  equivalent code, exactly like the existing Alpha Vantage classifier.
- `YfData` is a **process-wide singleton** (`SingletonMeta`) — one shared
  session/cookie jar for the whole process, thread-safe via an internal lock.
  This project's coordinator calls it exactly as it calls the existing
  Alpha Vantage client: through one injected client object.
- No `exec`/`eval`/dynamic code execution found anywhere in the reviewed
  source tree (13,298 lines, `grep`-checked).
- `websockets`/`pricing_pb2.py`/`live.py` implement a **live streaming quote
  feature over a persistent websocket** — **not used by this integration**.
  Only the plain request/response `Ticker.history()`/`.info`/`.fast_info`/
  `.dividends`/`.splits`/`.analyst_price_targets` methods are called; nothing
  in this project's client code imports or starts a websocket connection.

### 1.4 — Live verification (this session, AAPL, anonymous, no login cookies)

```
Ticker('AAPL').info            -> 180-key dict, 200 OK, no crumb/cookie failure
Ticker('AAPL').fast_info        -> lastPrice/previousClose/currency/exchange/marketCap
Ticker('AAPL').history(period='5d')  -> tz-aware DataFrame (America/New_York), OHLCV+Dividends+Splits
Ticker('AAPL').dividends        -> pandas Series, correct values
```

`.info` and `.fast_info` returned **slightly different `previousClose`**
values in this test (309.38 vs 310.54) — they are fetched from different
Yahoo endpoints under the hood and are not perfectly synchronized. This
project uses **`fast_info` as the sole source for the live quote tool** (it
is the field Yahoo itself designs for fast/current-quote access) and `.info`
only for descriptive company-profile fields, never re-deriving a quote from
it — avoiding exactly this kind of same-field-two-answers ambiguity inside a
single tool.

### 1.5 — What this project exposes

Six exact tools, no generic dispatcher (`finance.yahoo.symbol_search`,
`.stock_quote`, `.company_profile`, `.price_history`, `.corporate_actions`,
`.analyst_estimates`) — see `finance/yahoo_provider.py` /
`tools/finance_tools.py`.

---

## 2. SEC EDGAR

| Field | Value |
|---|---|
| Base hosts | `www.sec.gov` (ticker-to-CIK mapping, static), `data.sec.gov` (submissions, XBRL company facts) |
| Auth | **None** — no API key. A descriptive `User-Agent` header is REQUIRED. |
| License / ToS | US government work product; SEC's own fair-access policy governs automated use (below) |

### 2.1 — Official, documented, government-operated API — the opposite risk profile from Yahoo

Unlike Yahoo, SEC EDGAR's `data.sec.gov` JSON endpoints are an intentional,
documented, stable public API. Verified live in this session with a real
identifying User-Agent (`LocalLLMProject-PersonalAssistant/1.0 (research;
contact: <configured email>)`):

```
GET https://www.sec.gov/files/company_tickers.json          -> 200, 795,660 bytes
GET https://data.sec.gov/submissions/CIK0000320193.json      -> 200, 164,379 bytes (AAPL)
GET https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json -> 200, 3,789,099 bytes
```

Real response shapes confirmed (not assumed):

- **Ticker-to-CIK**: `{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}` —
  a flat, static-ish file, `symbol_search`/CIK resolution needs this fetched
  once and cached with a long TTL, not re-fetched per ticker.
- **Submissions**: `filings.recent` is a set of PARALLEL ARRAYS
  (`form`, `filingDate`, `accessionNumber`, ...) aligned by index, e.g.
  `form[0]="10-Q"`, `filingDate[0]="2026-07-31"`,
  `accessionNumber[0]="0000320193-26-000020"`.
- **Company facts (XBRL)**: `facts["us-gaap"][<concept>]["units"]["USD"]` is a
  list of fact objects, each with exactly
  `{start, end, val, accn, fy, fp, form, filed, frame}` — the
  `RevenueFromContractWithCustomerExcludingAssessedTax` concept was found on
  the FIRST attempt for AAPL, confirming this project's candidate-tag
  precedence list (Phase 4) against real data, not a guess.

### 2.2 — Fair-access rate limit (verified via SEC's own published guidance)

**10 requests/second maximum**, enforced per source IP regardless of how many
processes/machines are used; exceeding it produces a 403 and a roughly
10-minute IP-wide block on further EDGAR access. This project paces SEC
requests well under that ceiling (configured, not hardcoded — see §5) since
nothing about a single-user personal analysis needs to approach 10/s.
[SEC EDGAR Rate Limits: 10 Requests/Second Rule](https://dealcharts.org/blog/edgar-scraping-rate-limits-explained) ·
[SEC EDGAR API Rate Limit: 10 req/sec, User-Agent Header Required](https://tldrfiling.com/blog/sec-edgar-api-rate-limits-best-practices)

### 2.3 — User-Agent requirement

SEC rejects requests without a descriptive `User-Agent` naming the
application and a real contact (recommended form:
`AppName contact@email`). This project reads it from `SEC_USER_AGENT`
(configured, never hardcoded — the source tree contains no email address or
personal identifier; see `tools/config.py::sec_user_agent()`), and it is
**excluded from every log line, cache key, cache record, exception message,
evidence snapshot, and checkpoint** — the same discipline already applied to
`ALPHAVANTAGE_API_KEY` (`docs/security/ALPHAVANTAGE_PROVIDER_REVIEW.md` §5),
even though a User-Agent is not a secret in the same sense a key is; it is
still an identifying string that has no business appearing in cached data.

### 2.4 — What this project exposes

Five exact tools, no generic SEC URL fetcher (`finance.sec.resolve_company`,
`.company_submissions`, `.company_facts`, `.filing_metadata`,
`.financial_statements`) — restricted to the two documented hosts above via
this project's existing SSRF-safe fetch path
(`tools/http_safety.py::safe_get`), exactly like Alpha Vantage.

---

## 3. Dataset-specific provider policy

No single provider is authoritative for everything. Configured, not
hardcoded (`tools/config.py`):

| Capability | Provider | Why |
|---|---|---|
| Live/recent quote | Yahoo | SEC has no quote data at all; Alpha Vantage quotes are also fine but Yahoo needs no API key/quota |
| Price history / technicals input | Yahoo | Same reasoning; local OHLCV feeds `finance/metrics.py`'s technical indicators unchanged |
| Corporate actions (splits/dividends) | Yahoo | Native `Ticker.actions` |
| **US-company fundamentals (income/balance/cash-flow/shares)** | **SEC EDGAR** | Authoritative filed-with-the-government source, not a third party's re-derivation; this is the single biggest accuracy upgrade of this phase |
| Company profile (sector/industry/description) | Yahoo | SEC's submissions payload has only SIC code/description, not a business summary |
| Analyst estimates | Yahoo | Non-authoritative by nature either way; labelled as such regardless of source |
| News, non-US companies, optional validation | Alpha Vantage | Explicitly secondary — never a silent fallback (§5) |

For a non-US ticker (no CIK resolvable), fundamentals are reported as a
controlled missing-data state (`REDUCED`/omitted with a reason), never
silently substituted from Yahoo's summary financial fields or from Alpha
Vantage without explicit configuration.

---

## 4. Cross-provider reconciliation

When the same fact could come from more than one provider (e.g. Yahoo's
`marketCap` vs. a value derivable from SEC's `shares outstanding` × price),
the two are **kept as separate, separately-labelled evidence items** —
never silently overwritten. A material mismatch produces one of:
`PROVIDER_VALUE_CONFLICT`, `PERIOD_MISMATCH`, `UNIT_MISMATCH`,
`CURRENCY_MISMATCH`, `FILING_RESTATEMENT`, `STALE_PROVIDER_DATA`,
`MARKET_CAP_SHARE_COUNT_MISMATCH` — each naming both providers, both values,
both periods, the computed difference, which value was actually selected for
downstream use, and the selection policy that made that call (see
`finance/reconciliation.py`).

For the DCF specifically: SEC reported facts are preferred for supported US
companies; Yahoo's price is used **only** as the market-comparison value
against the modeled intrinsic value, never as a DCF input; Alpha Vantage
supplies a DCF input only under explicit configured fallback, with its
provenance visible in the report.

---

## 5. Configuration and kill switches

```
YAHOO_FINANCE_ENABLED=true
YAHOO_PERSONAL_USE_ACKNOWLEDGED=true   # must be explicitly true, or Yahoo tools are not registered
SEC_EDGAR_ENABLED=true
SEC_USER_AGENT=<AppName contact@email>  # required for SEC calls; missing -> controlled error, not a crash
SEC_MIN_REQUEST_INTERVAL_MS=<paced well under the 10 req/s ceiling>
FINANCE_QUOTE_PROVIDER=yahoo
FINANCE_PRICE_HISTORY_PROVIDER=yahoo
FINANCE_CORPORATE_ACTIONS_PROVIDER=yahoo
FINANCE_US_FUNDAMENTALS_PROVIDER=sec
FINANCE_COMPANY_PROFILE_PROVIDER=yahoo
FINANCE_ANALYST_ESTIMATES_PROVIDER=yahoo
FINANCE_NEWS_PROVIDER=alphavantage
FINANCE_SECONDARY_PROVIDER=alphavantage
FINANCE_AUTOMATIC_FALLBACK=false
```

Follows this project's existing flat-env-var convention in `tools/config.py`
(the bracketed `[finance.providers]` TOML-style syntax some drafts of this
spec suggested does not match this codebase's established pattern, per that
same spec's own "follow existing conventions" instruction).

Either provider can be disabled independently without affecting the other,
SEC, DCF, or the research pipeline (`YAHOO_FINANCE_ENABLED=false` /
`SEC_EDGAR_ENABLED=false`).

## 6. Honest residual limitations

- **Yahoo can break without notice.** It is not a supported API; a
  `consent.yahoo.com` HTML change or an internal endpoint rename could break
  quote/price retrieval at any time. Failure is controlled (a labelled
  `REDUCED` analysis, never a crash or a silent empty result), but there is
  no SLA to point to.
- **SEC XBRL tag coverage is issuer-specific.** The candidate-tag precedence
  list in Phase 4 covers common concepts verified against AAPL; an issuer
  using a genuinely custom/unlisted tag produces a controlled unresolved-field
  warning, not a guessed value.
- **yfinance's own local pickle cache is outside this project's control**
  (§1.2) — not a secret-leak risk, but not inspectable or clearable through
  this project's own cache-management surface either.
- **This is still not investment advice.** Unchanged from the existing
  guardrail — richer, more authoritative sourcing changes confidence in the
  facts, not the nature of the research output.

Sources consulted for SEC fair-access policy (page content, not endpoint
behavior — the live endpoint calls above are this project's own
verification):
[SEC EDGAR Rate Limits: 10 Requests/Second Rule](https://dealcharts.org/blog/edgar-scraping-rate-limits-explained),
[SEC EDGAR API Rate Limit: 10 req/sec, User-Agent Header Required](https://tldrfiling.com/blog/sec-edgar-api-rate-limits-best-practices)
