# Alpha Vantage provider review (Phase H.1)

**Date:** 2026-08-05
**Outcome:** Both candidate Alpha Vantage **MCP servers were reviewed and REJECTED**.
Market data is served by a **trusted local tool** over the project's existing
SSRF-safe fetch path instead.

This document records what was inspected, what was found, and why.

---

## 1. What was inspected

Both candidates were downloaded from PyPI, hash-verified against the PyPI-published
digest, unpacked, read, installed into a clean throwaway virtual environment
(CPython 3.13.13 / win_amd64, matching the project venv), imported, and — where it
got that far — actually launched over stdio with a real JSON-RPC handshake.

Nothing below is quoted from a README. Every finding was reproduced.

---

## 2. Candidate A — `marketdata-mcp-server` 0.3.1 (OFFICIAL)

| Field | Value |
|---|---|
| Repository | `github.com/alphavantage/alpha_vantage_mcp` (official Alpha Vantage org) |
| License | MIT |
| Version | 0.3.1 — **the only release that has ever existed** |
| Wheel SHA-256 | `50d60704028779e045d794701c9a7bb432c97b947d0c5a70aea344f27256cdbd` (verified) |
| Requires | Python >= 3.13 |
| Declared deps | `alphavantage-core`, `awslabs-mcp-lambda-handler>=0.1.8`, `click`, `loguru>=0.7.3`, `mcp>=1.12.3`, `python-dotenv>=1.1.1` |

### A.1 — Blocking: the stdio server exposes only a generic dispatcher

`av_mcp/stdio_server.py` registers exactly three tools:

```
TOOL_LIST   ()
TOOL_GET    (tool_name: str | list[str])
TOOL_CALL   (tool_name: str, arguments: object)
```

`TOOL_CALL` takes a **free-form tool-name string** chosen by the model and
dispatches it to any of 100+ Alpha Vantage endpoints.

This is structurally incompatible with this project's trust model. The trusted
catalog expresses authority as a **per-tool `default_tool_policy`**
(`mcp_management/catalog.py::_build_policy`), and `default_permission` is required
to be `denied` so undeclared tools get nothing. With this server the catalog can
only classify `TOOL_CALL` **as a whole** — it cannot distinguish `GLOBAL_QUOTE`
from any other endpoint. Enabling the server means enabling everything it can
reach.

That directly violates the Phase 2 requirements *"Do not expose every provider
tool globally"* and *"Map only reviewed provider tools to deterministic
capabilities."*

### A.2 — The endpoint registry is not in the reviewed package

`stdio_server.py` imports `av_api.context` and `av_api.registry`, but `av_api` is
**not in the wheel**. It comes from `alphavantage-core`, a separate distribution
with **one release (0.1.0)** and an **unpinned `httpx`** dependency. Reviewing
"the official server" would mean reviewing a package that is not the one named,
and whose version can change underneath the pin.

### A.3 — AWS Lambda machinery and an import-time S3 upload path

`awslabs-mcp-lambda-handler` pulls `boto3` + `botocore` into what is meant to be a
local stdio process. Worse, `av_mcp/common.py` executes at import time:

```python
set_response_processor(_server_response_processor)
```

and `_server_response_processor` calls `upload_to_object_storage()`, which
**uploads response data to S3** whenever a response exceeds `MAX_RESPONSE_TOKENS`.
It is gated on `CDN_BUCKET_NAME` / `CDN_DOMAIN` being set (we would not set them,
so it returns `None`), and `common.py` is not on the stdio import path — but an
import-time-installed data-exfiltration path shipping inside a local server is
exactly what this review exists to catch.

### A.4 — Credential handling is wrong for this project

`av_mcp/main.py`:

```python
@click.argument('api_key', required=False)
...
api_key = api_key or api_key_option or os.getenv('ALPHA_VANTAGE_API_KEY')
```

The documented invocation passes the key as a **positional argv element**. This
project's `configuration_generator` writes `command` + `args` into
`app_data/mcp_servers/<id>/server.json` and shows the argv in the approval plan,
so the key would be written to disk and displayed. The env fallback exists but
uses `ALPHA_VANTAGE_API_KEY` — a **different name** from the `ALPHAVANTAGE_API_KEY`
this project was asked to use.

**Verdict: rejected on A.1 alone; A.2–A.4 independently reinforce it.**

---

## 3. Candidate B — `alphavantage-mcp` 0.3.24 (third-party)

| Field | Value |
|---|---|
| Repository | `github.com/calvernaz/alphavantage` |
| License | Apache-2.0 |
| Releases | **0.3.23 and 0.3.24 only** |
| Wheel SHA-256 | `fe6ac686902b15a4cbcfdc38f55dc3dab72d2f7ba6cb057a340b81dcb1cc42b7` (verified) |
| Requires | Python >= 3.12 |

### What was good

It exposes **112 real, individually-named tools**, verified by importing
`tools_definitions()`. The nine this project would use, with their **exact
verified schemas**:

| Tool | Required | Properties |
|---|---|---|
| `stock_quote` | `symbol` | `symbol`, `datatype` |
| `company_overview` | `symbol` | `symbol` |
| `income_statement` | `symbol` | `symbol` |
| `balance_sheet` | `symbol` | `symbol` |
| `cash_flow` | `symbol` | `symbol` |
| `company_earnings` | `symbol` | `symbol` |
| `time_series_daily_adjusted` | `symbol` | `symbol`, `outputsize`, `datatype` |
| `symbol_search` | `keywords` | `keywords`, `datatype` |
| `news_sentiment` | `tickers` | `tickers`, `topics`, `time_from`, `time_to`, `sort`, `limit`, `datatype` |

It also reads the credential from `ALPHAVANTAGE_API_KEY` **and only from the
environment** (`api.py`: `API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")`), never
from argv — exactly the required handling.

### B.1 — BLOCKING: the stdio server cannot start from an installed package

`alphavantage_mcp_server/server.py`:

```python
def get_version():
    with open("pyproject.toml", "r") as f:      # relative to the CWD
        pyproject = toml.load(f)
        return pyproject["project"]["version"]

async def run_stdio_server():
    init_telemetry(start_metrics=True)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(..., server_version=get_version(), ...)
```

`get_version()` opens `pyproject.toml` **relative to the current working
directory**, with no fallback. An installed package has no `pyproject.toml` in
its CWD, so the server dies during startup:

```
FileNotFoundError: [Errno 2] No such file or directory: 'pyproject.toml'
exit code 1
```

Reproduced directly. **Both** published releases contain it. The server only works
when run from a source checkout.

### B.2 — Undeclared dependency

`api.py` does `from dotenv import load_dotenv`, but `python-dotenv` is **absent
from `requires_dist`**. A clean `pip install alphavantage-mcp==0.3.24` therefore
produces a broken install (`ModuleNotFoundError: No module named 'dotenv'`).

Note also that `load_dotenv()` runs at import and searches upward for a `.env`.

### B.3 — Unbounded `mcp` dependency resolves to an incompatible major version

The package declares `mcp>=1.9.4`. Today pip resolves `mcp==2.0.0`, in which
`Server.list_prompts` no longer exists:

```
AttributeError: 'Server' object has no attribute 'list_prompts'
```

It imports successfully only when pinned back to `mcp==1.29.0` (verified).

### B.4 — Telemetry is ON by default and binds a local network listener

`telemetry_bootstrap.py`:

```python
MCP_TELEMETRY_ENABLED = os.getenv("MCP_TELEMETRY_ENABLED", "true").lower() == "true"
MCP_METRICS_PORT      = int(os.getenv("MCP_METRICS_PORT", "9464"))
...
start_http_server(MCP_METRICS_PORT, addr="127.0.0.1")
```

`run_stdio_server()` calls `init_telemetry(start_metrics=True)` unconditionally, so
a "local stdio" server starts a **Prometheus HTTP listener on 127.0.0.1:9464**
exposing per-tool call metrics. Loopback-only, but unrequested.

### B.5 — Minor: an unreachable tool

The `AlphavantageTools` enum maps `COMPANY_SPLITS` to the value
`"company_dividends"`, duplicating `COMPANY_DIVIDENDS`. `company_splits` is
therefore not addressable by name.

### Why it was still rejected

B.1 is fatal on its own. It is *workaroundable* — the project controls the child's
`working_directory`, so a project-authored `pyproject.toml` shim could be written
into the isolated runtime workspace — but the full mitigation stack would have
been:

1. hash-lock pinning `mcp==1.29.0` and adding the undeclared `python-dotenv`;
2. a fabricated `pyproject.toml` shim so `get_version()` succeeds (meaning the
   server reports a version string **we** wrote);
3. a **new `environment_overrides` field** in `mcp_layer/config.py` and
   `mcp_layer/external.py` to force `MCP_TELEMETRY_ENABLED=false` — the child env
   is currently built from allowlisted *names* only
   (`mcp_layer/external.py:119`), so there is no existing way to set a literal.

That is three pieces of load-bearing scaffolding propping up an unmaintained
two-release package, one of which modifies a security-sensitive shared layer.

---

## 4. Chosen approach — trusted local provider

Market data is served by ordinary `BaseTool`s in `tools/finance_tools.py`.

**What this preserves**

- Per-endpoint authority: nine reviewed datasets in `finance/datasets.py`, and
  `resolve_dataset` **fails closed** on anything else. There is no dispatcher and
  no free-form endpoint string.
- `ToolExecutor` remains the sole execution authority. Every tool is
  `permission = READ` and `requires_internet = True` (except `finance.dcf_model`,
  which needs no network).
- Every request goes through `tools/http_safety.py::safe_get`, so scheme, port,
  embedded-credential, DNS and private-IP rules are enforced identically to
  `browser.fetch_page` — including on redirects (`max_redirects=0` here, since the
  documented endpoint never legitimately redirects).

**What it avoids**

No third-party process, no lock file, no candidate validation, no orphan-process
risk, no unreviewed transitive dependency tree, and no change to `mcp_layer`.

**What it gives up**

The project gains no new MCP provider from this phase. The MCP subsystem is
untouched and continues to work exactly as before.

---

## 5. Credential handling

`ALPHAVANTAGE_API_KEY` is read via `tools/config.py::alphavantage_api_key()` **at
call time**, so a missing key disables only market data (reported as
`MARKET_DATA_API_KEY_MISSING`) and never breaks startup.

The key appears in exactly one place: the query string of the outbound request,
built inside `AlphaVantageClient._build_url` and never returned to a caller.

It is provably absent from:

| Surface | Mechanism |
|---|---|
| Cache keys | `build_cache_key` hashes provider/dataset/symbol/arguments/freshness/schema only. Test: `test_cache_key_never_contains_the_api_key` |
| Cache records | Only the validated payload is stored. Tests: `test_api_key_never_appears_in_cache_contents` (checks the DB **and** the WAL sidecar) |
| Logs | `_log_meta` carries dataset, origin, cache status and call counts only |
| Exception messages | `provider._redact()` strips `apikey=`/`token=` fragments and whole URLs from every provider-derived string |
| Workflow output | Test: `test_no_secret_appears_anywhere_in_the_result` |
| Repository | `.env` is gitignored; `.env.example` ships an empty key |

## 6. Capability boundaries

- **No trading, order placement, brokerage write, or portfolio access.** Enforced
  by test `test_no_trading_or_order_capability_exists`, which scans every
  registered tool name.
- **No shell execution and no arbitrary Python** from workflow or skill text. The
  workflow holds no registry and calls tools only through `ToolExecutor`.
- **No provider tools on unrelated requests.** Market-data tools set
  `shortlist_requires_relevance = True`, so they are excluded from the Phase B
  shortlist when nothing in the request matches them.

## 7. Honest residual limitations

- **The DNS-rebinding TOCTOU window in `tools/http_safety.py` applies here too.**
  It is documented in that module and is not made worse by this phase.
- **OS-level outbound networking is not sandboxed.** The request is made in-process
  by `requests`; the project does not claim firewall-level isolation.
- **Alpha Vantage free-tier data is delayed.** Nothing in this phase labels any
  price as realtime; `normalize_quote` hardcodes `price_basis = "delayed"` and the
  workflow propagates it. Entitlement is not verified, so realtime is never claimed.
- **The daily quota limit is an estimate.** `ALPHAVANTAGE_DAILY_CALL_LIMIT`
  defaults to 25 but is not a claim about any plan; a provider rate-limit response
  is always treated as authoritative and stops further calls.
- **A DCF is a scenario model, not a prediction.** Outputs are research inputs and
  are explicitly labelled "not a trading instruction".
