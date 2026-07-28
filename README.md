# home-ai — local-first voice assistant

A local-first Windows voice assistant: mic in → a local model (via Ollama) answers out loud,
escalating hard or accuracy-critical questions to the Claude API. See `spec.md` for the full
design and `CAMERA.md` for the camera features.

## Running the assistant

```
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys/models
python assistant.py
```

At the prompt choose `t` (text), `p` (push-to-talk), or `q` (quit).

## Phase 1: local tool calling

The local model can now call a small set of **safe, built-in tools**, receive a structured JSON
result, and then write the final answer itself. The tool never writes the user-facing reply — the
local LLM always does. This runs **only on the local answering path**; Claude routing is unchanged.

### Available Phase 1 tools

- **`system.echo`** — echoes text back (diagnostics).
- **`math.calculate`** — evaluates a basic arithmetic expression (`+ - * / **`, parentheses,
  decimals, unary minus) using a restricted parser — never `eval`.

Internet access, GitHub search, plugins, MCP, filesystem, and shell tools are **not** included yet.

### Enabling / disabling

Configured via environment (safe defaults; the app starts fine if these are unset):

```
OLLAMA_URL=http://localhost:11434
TOOL_CALLING_ENABLED=true          # set false to restore original single-shot local chat
MAX_TOOL_STEPS=5                   # max tool calls per turn (every call counts)
DEFAULT_TOOL_TIMEOUT_SECONDS=10
```

### Example interaction

```
> Use the calculator to calculate (17 * 23) + 5.
The result is 396.
```

Internally: the router picks the local path → the local model requests `math.calculate` →
the executor returns `{"expression": "(17 * 23) + 5", "result": 396}` → that JSON is handed back
to the local model → the model replies "The result is 396."

### Running the tests

```
python -m pytest -q
```

The suite is deterministic and does not require a live Ollama endpoint.

### More detail

- `docs/phase1-tool-framework.md` — design, configuration, token overhead, limitations.
- `docs/phase1-tool-call-test.md` — the verified native Ollama tool-call/result message format.
- `docs/phase1-tool-framework-plan.md` — the implementation plan.

## Phase 2A: read-only internet & GitHub tools

The local model can now **search the public web and inspect public GitHub repositories** through
read-only tools, built on the same Phase 1 framework. Results always return to the local LLM, which
writes the final answer and cites its sources. **Repositories are never cloned or executed**, no
files are downloaded, nothing is installed, and no writes are made to GitHub.

### Available Phase 2A tools

- **`browser.search`** — search the public web (Tavily). Needs `TAVILY_API_KEY`.
- **`browser.fetch_page`** — fetch a public **HTTPS** page and extract readable text. Blocks
  local/private/loopback/metadata addresses (SSRF-guarded), enforces byte/char/redirect limits,
  and never executes JavaScript.
- **`github.search_repositories`**, **`github.get_repository`**, **`github.read_file`**,
  **`github.list_directory`**, **`github.list_releases`** — read-only GitHub REST API access.

Remote content is marked **untrusted**: the model is instructed never to follow instructions found
inside fetched pages or repository files (prompt-injection defense). Still **not** included: repo
cloning, code/shell execution, package installs, plugins, MCP, private repos, or any GitHub writes.

### Environment variables

```
INTERNET_TOOLS_ENABLED=true     # false → the 7 tools are not offered to the model
INTERNET_READ_ENABLED=true      # false → tools are visible but blocked at execution
SEARCH_PROVIDER=tavily
TAVILY_API_KEY=                 # without it, browser.search returns a controlled missing-key error
GITHUB_TOKEN=                   # optional; read-only token → higher rate limits + live tests
HTTP_USER_AGENT=Local-LLM-Project/1.0
ALLOW_HTTP_FETCH=false          # HTTPS only by default
SEARCH_MAX_RESULTS=10
MAX_SEARCH_QUERY_CHARS=500
BROWSER_CONNECT_TIMEOUT_SECONDS=5
BROWSER_READ_TIMEOUT_SECONDS=20
BROWSER_MAX_REDIRECTS=5
MAX_PAGE_BYTES=2000000
MAX_PAGE_CHARS=30000
GITHUB_TIMEOUT_SECONDS=20
GITHUB_MAX_FILE_BYTES=1000000
GITHUB_MAX_FILE_CHARS=30000
GITHUB_MAX_DIRECTORY_ENTRIES=200
GITHUB_MAX_RELEASES=10
```

Missing optional credentials never break startup — only the specific capability is disabled.

### Example interactions

GitHub:
```
> Find public GitHub repositories for financial analysis tools and compare the top two.
The local model calls github.search_repositories, github.get_repository, and github.read_file,
then answers with each repo's full name, URL, stars, and what it read — citing only real sources.
```

Browser:
```
> Search for the official page of <project> and summarize it.
The local model calls browser.search, then browser.fetch_page on a result, then summarizes the
page and names the final URL. (A search snippet alone is not treated as having read the page.)
```

### Security limitations

`browser.fetch_page` resolves and validates every hop against an IP blocklist, but a small
DNS-rebinding (TOCTOU) window exists between resolution and connection. Text extraction is
best-effort (no JavaScript rendering). See `docs/phase2a-internet-github-tools.md` for the full
security model, SSRF protections, rate-limit handling, and token-overhead numbers.

### Running the tests (Phase 1 + Phase 2A)

```
python -m pytest -q
```

Automated tests mock all network access — no internet or API keys required. Optional live tests
(GitHub / Tavily / a public page) run only when credentials and connectivity are available and are
reported honestly as skipped otherwise.
