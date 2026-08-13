"""Phase 2A configuration.

Getter functions read os.environ on each call (with safe defaults), so tests can
toggle behavior via monkeypatch.setenv without import-time binding or reloads.
Missing optional credentials must never break startup — they only disable the
specific capability that needs them.
"""

import os

from dotenv import load_dotenv

# Load .env so the tools are self-contained: config works even when a tool is used
# without importing brain first. Idempotent and non-overriding (real process env and
# test monkeypatches win over .env values).
load_dotenv()


def _bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _str(name, default):
    val = os.environ.get(name)
    return val if val not in (None, "") else default


# ---- Phase B: tool selection budget ----
# Bound the candidate tool set (and each description) placed in the local model's
# selection prompt, so prompt size stays ~constant as the registry grows.
def max_shortlist_tools():
    return _int("MAX_SHORTLIST_TOOLS", 5)


def max_tool_description_chars():
    return _int("MAX_TOOL_DESCRIPTION_CHARS", 300)


def max_selection_prompt_chars():
    return _int("MAX_SELECTION_PROMPT_CHARS", 8000)


# ---- Capability toggles ----
def internet_tools_enabled():
    return _bool("INTERNET_TOOLS_ENABLED", True)


def internet_read_enabled():
    return _bool("INTERNET_READ_ENABLED", True)


# ---- Search provider ----
def search_provider():
    return _str("SEARCH_PROVIDER", "tavily")


def tavily_api_key():
    return _str("TAVILY_API_KEY", None)


def search_max_results():
    return _int("SEARCH_MAX_RESULTS", 10)


def max_search_query_chars():
    return _int("MAX_SEARCH_QUERY_CHARS", 500)


# ---- HTTP / browser ----
def http_user_agent():
    return _str("HTTP_USER_AGENT", "Local-LLM-Project/1.0")


def allow_http_fetch():
    return _bool("ALLOW_HTTP_FETCH", False)


def browser_connect_timeout():
    return _int("BROWSER_CONNECT_TIMEOUT_SECONDS", 5)


def browser_read_timeout():
    return _int("BROWSER_READ_TIMEOUT_SECONDS", 20)


def browser_max_redirects():
    return _int("BROWSER_MAX_REDIRECTS", 5)


def max_page_bytes():
    return _int("MAX_PAGE_BYTES", 2_000_000)


def max_page_chars():
    return _int("MAX_PAGE_CHARS", 30_000)


# ---- GitHub ----
def github_token():
    return _str("GITHUB_TOKEN", None)


def github_timeout():
    return _int("GITHUB_TIMEOUT_SECONDS", 20)


def github_max_file_bytes():
    return _int("GITHUB_MAX_FILE_BYTES", 1_000_000)


def github_max_file_chars():
    return _int("GITHUB_MAX_FILE_CHARS", 30_000)


def github_max_directory_entries():
    return _int("GITHUB_MAX_DIRECTORY_ENTRIES", 200)


def github_max_releases():
    return _int("GITHUB_MAX_RELEASES", 10)


# ---- Phase 2B: clone + static repository inspection ----
def repository_clone_enabled():
    return _bool("REPOSITORY_CLONE_ENABLED", False)


def repository_inspection_enabled():
    # Enabled implicitly whenever cloning is on, or explicitly via its own flag.
    return _bool("REPOSITORY_INSPECTION_ENABLED", False) or repository_clone_enabled()


def repository_root():
    return _str("REPOSITORY_ROOT", "data/repositories")


def git_executable():
    return _str("GIT_EXECUTABLE", "git")


def git_clone_timeout():
    return _int("GIT_CLONE_TIMEOUT_SECONDS", 120)


def max_repository_preflight_size_kb():
    return _int("MAX_REPOSITORY_PREFLIGHT_SIZE_KB", 200_000)


def max_cloned_repository_size_mb():
    return _int("MAX_CLONED_REPOSITORY_SIZE_MB", 250)


def max_cloned_repository_files():
    return _int("MAX_CLONED_REPOSITORY_FILES", 25_000)


def repo_max_list_entries():
    return _int("REPO_MAX_LIST_ENTRIES", 500)


def repo_max_list_depth():
    return _int("REPO_MAX_LIST_DEPTH", 5)


def repo_max_read_bytes():
    return _int("REPO_MAX_READ_BYTES", 1_000_000)


def repo_max_read_chars():
    return _int("REPO_MAX_READ_CHARS", 30_000)


def repo_scan_max_files():
    return _int("REPO_SCAN_MAX_FILES", 5_000)


def repo_scan_max_file_bytes():
    return _int("REPO_SCAN_MAX_FILE_BYTES", 500_000)


def repo_scan_max_total_bytes():
    return _int("REPO_SCAN_MAX_TOTAL_BYTES", 50_000_000)


def repo_scan_max_depth():
    return _int("REPO_SCAN_MAX_DEPTH", 20)


def repo_scan_max_findings():
    return _int("REPO_SCAN_MAX_FINDINGS", 500)


# ---- Phase C: untrusted repository text ----
# Hard cap on any raw repository text placed in the model prompt. Repository
# content is untrusted; keep excerpts small, bounded, and clearly labeled.
def max_untrusted_repo_text_chars():
    return _int("MAX_UNTRUSTED_REPO_TEXT_CHARS", 4000)


# ---- Phase D: MCP layer (internal test server only) ----
def mcp_test_server_enabled():
    return _bool("MCP_TEST_SERVER_ENABLED", True)


def mcp_test_workspace():
    return _str("MCP_TEST_WORKSPACE", "test_workspace")


def mcp_startup_timeout():
    return _int("MCP_STARTUP_TIMEOUT_SECONDS", 15)


def mcp_call_timeout():
    return _int("MCP_CALL_TIMEOUT_SECONDS", 20)


# ---- Phase E: external single-server MCP configuration ----
def mcp_config_path():
    return _str("MCP_CONFIG_PATH", "config/mcp_server.json")


def mcp_workspaces_root():
    return _str("MCP_WORKSPACES_ROOT", "mcp_workspaces")


# ---- Phase F: automatic MCP provisioning ----
def mcp_catalog_path():
    return _str("MCP_CATALOG_PATH", "config/mcp_catalog.json")


def mcp_managed_root():
    """Root of the managed installation area. Never the repo root or a venv."""
    return _str("MCP_MANAGED_ROOT", "app_data/mcp_servers")


def mcp_install_timeout():
    return _int("MCP_INSTALL_TIMEOUT_SECONDS", 300)


def mcp_provisioning_enabled():
    return _bool("MCP_PROVISIONING_ENABLED", True)


# ---- Phase H.1: market data, cache, quota, DCF, stock analysis ----
# The Alpha Vantage provider is a TRUSTED LOCAL tool over tools/http_safety.py.
# The API key is read here, at call time, and is NEVER written to a cache key,
# a cache record, a log line, an exception message, or a test snapshot.
def market_data_enabled():
    return _bool("MARKET_DATA_ENABLED", True)


def alphavantage_api_key():
    """The provider credential. Read at call time so a missing key disables only
    market data (reported as MARKET_DATA_API_KEY_MISSING) and never breaks startup."""
    return _str("ALPHAVANTAGE_API_KEY", None)


def alphavantage_endpoint():
    return _str("ALPHAVANTAGE_ENDPOINT", "https://www.alphavantage.co/query")


def market_data_connect_timeout():
    return _int("MARKET_DATA_CONNECT_TIMEOUT_SECONDS", 5)


def market_data_read_timeout():
    return _int("MARKET_DATA_READ_TIMEOUT_SECONDS", 20)


def max_market_data_bytes():
    return _int("MAX_MARKET_DATA_BYTES", 8_000_000)


# ---- cache ----
def market_data_cache_path():
    return _str("MARKET_DATA_CACHE_PATH", "app_data/finance_cache/market_data.sqlite3")


def market_data_cache_enabled():
    return _bool("MARKET_DATA_CACHE_ENABLED", True)


def market_data_stale_if_error_seconds():
    """How far past expiry a cached record may still be served when a refresh
    fails. Serving one is always reported as stale — never as fresh."""
    return _int("MARKET_DATA_STALE_IF_ERROR_SECONDS", 7 * 24 * 3600)


def market_data_negative_ttl_seconds():
    """TTL for a CONTROLLED negative result (e.g. a provider-confirmed unknown
    symbol). Never used for transport failures."""
    return _int("MARKET_DATA_NEGATIVE_TTL_SECONDS", 3600)


# ---- quota ----
def alphavantage_daily_call_limit():
    """Estimated daily external-call budget. NOT a claim about the user's plan —
    provider rate-limit responses are always treated as authoritative."""
    return _int("ALPHAVANTAGE_DAILY_CALL_LIMIT", 25)


def market_data_max_retries():
    return _int("MARKET_DATA_MAX_RETRIES", 2)


def market_data_retry_base_delay_ms():
    return _int("MARKET_DATA_RETRY_BASE_DELAY_MS", 500)


def market_data_use_adjusted_prices():
    """Whether to request split/dividend-ADJUSTED price history.

    TIME_SERIES_DAILY_ADJUSTED is a premium Alpha Vantage endpoint; on a free key
    it fails with MARKET_DATA_ENTITLEMENT_REQUIRED. Defaults off so a free key
    never spends a call on a guaranteed failure. Turn on if your plan includes it.
    """
    return _bool("MARKET_DATA_USE_ADJUSTED_PRICES", False)


def market_data_min_request_interval_ms():
    """Minimum spacing between EXTERNAL provider calls, process-wide.

    Alpha Vantage's free tier allows roughly one request per second. Pacing is
    cheaper than tripping the throttle and retrying, and cache hits bypass it
    entirely. Set to 0 to disable (tests do this).
    """
    return _int("MARKET_DATA_MIN_REQUEST_INTERVAL_MS", 1200)


# ---- stock analysis workflow ----
def stock_analysis_enabled():
    return _bool("STOCK_ANALYSIS_ENABLED", True)


# ---- compact synthesis payload (bounds what reaches the local LLM) ----
def stock_analysis_compact_history_years():
    """How many of the most recent ANNUAL periods per statement are included
    in the synthesis payload. Quarterly periods are never included in
    synthesis at all — nothing that calculates from them needs them there."""
    return _int("STOCK_ANALYSIS_COMPACT_HISTORY_YEARS", 5)


def stock_analysis_compact_price_observations():
    """How many of the most recent daily closes are included as illustrative
    price points, separate from and much smaller than the full OHLCV history
    used to calculate technical indicators (which is never sent to the LLM)."""
    return _int("STOCK_ANALYSIS_COMPACT_PRICE_OBSERVATIONS", 10)


def stock_analysis_compact_max_warnings():
    return _int("STOCK_ANALYSIS_COMPACT_MAX_WARNINGS", 20)


def stock_analysis_compact_max_provenance_entries():
    return _int("STOCK_ANALYSIS_COMPACT_MAX_PROVENANCE_ENTRIES", 20)


def stock_analysis_earnings_max_annual():
    return _int("STOCK_ANALYSIS_EARNINGS_MAX_ANNUAL", 5)


def stock_analysis_earnings_max_quarterly():
    return _int("STOCK_ANALYSIS_EARNINGS_MAX_QUARTERLY", 8)


# ---- staged research pipeline (bull/bear/rebuttal/manager/risk/final) ----
# Adapted from TauricResearch/TradingAgents' role structure (bull researcher,
# bear researcher, research manager, risk reviewer) at commit
# a33fd4c0f134485a43553a2c23a63cb14adbd88f — NOT imported as a runtime
# dependency; see finance/research_pipeline.py for what was adapted vs.
# deliberately excluded (its trader/portfolio-manager/order-execution
# behavior, its unbounded free-text debate loop, and its 3-way risk debate
# over a trade proposal).
def stock_analysis_research_pipeline_enabled():
    return _bool("STOCK_ANALYSIS_RESEARCH_PIPELINE_ENABLED", True)


def research_stage_max_output_tokens(stage=None):
    """Per-stage output token budget, passed to Ollama as options.num_predict.

    COR corrective patch (Phase 1/live verification): 700 was tuned for the
    old, shorter {point, evidence_cited} researcher schema. The Phase 3
    per-claim rewrite ({claim_id, claim, evidence_ids, claim_type,
    assumptions, confidence} x 2-5 claims) needs meaningfully more room --
    confirmed live against real COR data, where 700 reliably truncated
    bull_researcher's JSON mid-object (surfacing as "response did not
    contain a parseable JSON object", NOT a content-policy violation) and
    2500 reliably did not, across repeated trials. Raised to 3000 for
    headroom above that confirmed-working figure.

    MLI corrective patch: raised 3000 -> 8000. A reasoning model spends
    output budget on THINKING tokens that never appear in
    `message.content`, and that spend scales with prompt size. On MLI (119
    evidence items, ~5.4k prompt tokens) bull_researcher reproducibly
    returned completion_tokens=3002 against a 3000 cap with an EMPTY body --
    the entire budget consumed before any JSON was emitted, surfacing as
    "response did not contain a parseable JSON object" (a parse error, which
    by design gets NO repair attempt) and taking the whole pipeline down.
    The same call at 8000 returns valid JSON in ~4.4k tokens. Note this is
    a CEILING, not a target: stages that need less still use less, so the
    higher bound costs nothing on smaller companies.

    WM corrective patch: raised 8000 -> 16000, and made PER-STAGE.
    8000 was measured, not guessed, and it was too tight. Four consecutive
    live WM runs produced research_manager completion_tokens of 7719, 7587,
    7691 against the 8000 cap -- within ~300 tokens of truncation every
    single time. That is the mechanism behind "research_manager fails for a
    different stock every time": it is not ticker-specific, it is a stage
    running permanently at the edge of its budget, where any company whose
    evidence produces slightly more text tips over. And because a truncated
    response is indistinguishable from a malformed one without
    `done_reason`, the failure did not even look like a budget problem.

    Per-stage because the stages are not alike: research_manager reconciles
    two full researcher outputs into a ten-field schema with seven lists,
    and the final synthesizer carries the recommendation plus its
    justification. `rebuttal_round` needs a fraction of that.
    """
    base = _int("RESEARCH_STAGE_MAX_OUTPUT_TOKENS", 16000)
    if not stage:
        return base
    override = _int(f"RESEARCH_STAGE_MAX_OUTPUT_TOKENS_{stage.upper()}", 0)
    if override > 0:
        return override
    return int(base * _STAGE_OUTPUT_BUDGET_SCALE.get(stage, 1.0))


# Relative output budget per stage, applied to
# `RESEARCH_STAGE_MAX_OUTPUT_TOKENS`. A ceiling, never a target -- a stage
# that needs less still uses less. Ratios reflect measured live usage:
# research_manager and the two researchers are the heavy stages.
_STAGE_OUTPUT_BUDGET_SCALE = {
    "bull_researcher": 1.0,
    "bear_researcher": 1.0,
    "rebuttal_round": 0.6,
    "research_manager": 1.25,
    "risk_reviewer": 0.75,
    "final_investment_synthesizer": 1.0,
}


def research_stage_timeout_seconds():
    """COR corrective patch: raised alongside research_stage_max_output_tokens
    -- a larger output budget needs more wall-clock time to generate, and a
    live run observed an actual read timeout at the old 90s on a large
    (research_manager-sized) prompt even before this change."""
    return _int("RESEARCH_STAGE_TIMEOUT_SECONDS", 180)


def stock_analysis_synthesis_timeout_seconds():
    """GE corrective patch: the SAME class of bug `research_stage_timeout_
    seconds` was raised to fix, found in a DIFFERENT call site that never got
    the same treatment -- finance/workflow.py::_ask_local_with_content_policy
    (the report_detail="full" single-shot narrative call, and its one repair
    attempt) called `ask_local_fn(messages)` with no explicit timeout at all,
    silently inheriting `brain.ask_local_raw`'s bare 120-second default. That
    call's prompt is the FULL compact payload (larger than any one research-
    pipeline stage's prompt) and its completion has no `options.num_predict`
    cap at all (unlike a pipeline stage, capped by
    `research_stage_max_output_tokens`) -- a live GE run hit "Read timed out.
    (read timeout=120)" on exactly this path. Defaults higher than
    `research_stage_timeout_seconds` for that reason, not copied from it."""
    return _int("STOCK_ANALYSIS_SYNTHESIS_TIMEOUT_SECONDS", 240)


def research_material_scenario_spread():
    """MLI corrective patch: bull-to-bear scenario range, as a fraction of
    the base modeled value, at or above which research readiness is capped at
    LIMITED (finance/workflow.py::_assumption_quality_limitations).

    Needs calibration against this DCF's OWN typical spread, not an abstract
    notion of "wide" -- observed live: MLI 97%, AMZN 127%, DIS 96%. A
    threshold much below ~0.9 would mark essentially every analysis LIMITED
    and make the signal worthless. 0.90 flags a bull-to-bear range wider than
    the base value itself; tune as more tickers are observed."""
    return float(_str("RESEARCH_MATERIAL_SCENARIO_SPREAD", "0.90"))


def stock_analysis_include_news():
    return _bool("STOCK_ANALYSIS_INCLUDE_NEWS", False)


def stock_analysis_forecast_years():
    return _int("STOCK_ANALYSIS_FORECAST_YEARS", 5)


# ---- H.4 corrective patch: report length/detail mode ----
# The full, historically-detailed report (bull/bear/rebuttal transcript,
# complete assumption provenance, full sensitivity grid, ...) stays available
# unchanged behind report_detail="full" -- see finance/workflow.py::
# render_compact_report / synthesize_report. Nothing internal (deterministic
# facts, DCF detail, evidence) is ever reduced by this setting -- only what
# is RENDERED to the user.
def stock_analysis_report_detail_default():
    """'compact' (default) or 'full'. A user request containing an explicit
    "detailed"/"full report" phrase overrides this for that one request
    (finance/workflow.py::detect_report_detail) regardless of this setting."""
    return _str("STOCK_ANALYSIS_REPORT_DETAIL", "compact")


def stock_analysis_compact_report_target_words():
    """Soft target for the compact report's rendered length. Not a hard
    truncation limit -- the renderer's fixed section/bullet caps are what
    actually bound the length; this is only the documented target tests
    check against, mirroring the compact-synthesis-payload token target."""
    return _int("STOCK_ANALYSIS_COMPACT_REPORT_TARGET_WORDS", 1500)


def dcf_min_forecast_years():
    return _int("DCF_MIN_FORECAST_YEARS", 1)


def dcf_max_forecast_years():
    return _int("DCF_MAX_FORECAST_YEARS", 15)


def dcf_min_discount_rate():
    return float(_str("DCF_MIN_DISCOUNT_RATE", "0.01"))


def dcf_max_discount_rate():
    return float(_str("DCF_MAX_DISCOUNT_RATE", "0.60"))


def dcf_net_debt_policy():
    """The default net-debt policy. Conservative by design: 'cash_only' never
    assumes an unverified security is liquid. Switch to
    'cash_and_marketable_securities' only alongside
    dcf_short_term_investments_eligible=true — see finance.dcf.NetDebtPolicy."""
    return _str("DCF_NET_DEBT_POLICY", "cash_only")


def dcf_short_term_investments_eligible():
    """Whether short-term investments are ELIGIBLE to net against debt under
    the 'cash_and_marketable_securities' policy. False by default — this
    project does not verify per-security liquidity, so eligibility is an
    explicit operator opt-in, never inferred."""
    return _bool("DCF_SHORT_TERM_INVESTMENTS_ELIGIBLE", False)


# ---- Phase H.3 corrective patch: CapEx/D&A/NWC assumption hierarchy ----
# Explicitly configured LAST-RESORT defaults only — used when NO reported
# historical ratio exists at all. Never derived from a margin difference or
# any other proxy; see finance/workflow.py::propose_assumptions and
# docs/PHASE_H1_STOCK_ANALYSIS.md's DCF assumption section for the full
# reported-history-first hierarchy this backs.
def dcf_default_capex_pct_revenue():
    return float(_str("DCF_DEFAULT_CAPEX_PCT_REVENUE", "0.05"))


def dcf_default_depreciation_pct_revenue():
    return float(_str("DCF_DEFAULT_DEPRECIATION_PCT_REVENUE", "0.04"))


def dcf_default_working_capital_pct_revenue():
    return float(_str("DCF_DEFAULT_WORKING_CAPITAL_PCT_REVENUE", "0.02"))


def dcf_assumption_history_max_years():
    """How many reported annual periods the CapEx/D&A/NWC historical-ratio
    hierarchy may look back across (average of up to this many years)."""
    return _int("DCF_ASSUMPTION_HISTORY_MAX_YEARS", 5)


# ---- Phase H.4: forward assumptions ----
def dcf_long_run_growth():
    """The rate a forecast growth path FADES TOWARD across the horizon.

    Not a terminal-growth assumption (that is `terminal_growth`, applied to
    the perpetuity) — this is the year-5 end of the explicit forecast. It
    exists because the previous behaviour held one historical CAGR flat
    across every forecast year, which is not a neutral default but an
    aggressive one: it asserts a company's current growth persists unchanged
    for the whole horizon. Defaulted to a broad-economy nominal rate."""
    return float(_str("DCF_LONG_RUN_GROWTH", "0.03"))


def dcf_default_revenue_growth():
    """LAST-RESORT growth, used only when NO growth evidence of any kind
    exists — no guidance, no TTM trend, no reported year-over-year, no
    history. Deliberately conservative; a company we know nothing about is
    not assumed to grow faster than the economy."""
    return float(_str("DCF_DEFAULT_REVENUE_GROWTH", "0.03"))


def dcf_default_operating_margin():
    """LAST-RESORT operating margin, used only when none was ever reported."""
    return float(_str("DCF_DEFAULT_OPERATING_MARGIN", "0.10"))


def forward_assumption_model_enabled():
    """Whether the local model may PROPOSE a forward path at all.

    When false the deterministic baseline is used directly. The model's
    proposal is always validated and can always be rejected, so this is a
    performance/latency switch rather than a safety one — the safety comes
    from `finance/forward_assumptions.py::validate_proposal`, not from here."""
    return _bool("FORWARD_ASSUMPTION_MODEL_ENABLED", True)


def guidance_ingestion_enabled():
    """Whether SEC-filed management guidance is fetched and extracted.

    Costs two extra SEC requests per analysis (the filing index and the
    earnings-release exhibit). Disabling it is a supported configuration:
    the whole workflow is built to run with guidance unavailable (see
    finance/freshness.py and section 20 of the phase spec)."""
    return _bool("GUIDANCE_INGESTION_ENABLED", True)


def research_run_artifacts_enabled():
    """Whether each research-pipeline run is written to disk for replay.

    Phase H.5, Phase 0. OFF by default: this is diagnostic capture, not a
    product feature, and an artifact holds every stage's full output
    (~100 KB per run). Turn it on when investigating a validation failure --
    without it there is nothing to replay, which is exactly the position the
    WM investigation was in (the 1-to-4-to-3 escalation existed only in a
    transient console buffer and could not be re-examined)."""
    return _bool("RESEARCH_RUN_ARTIFACTS_ENABLED", False)


def research_run_artifacts_dir():
    return _str("RESEARCH_RUN_ARTIFACTS_DIR", "logs/research_runs")


def research_stage_max_attempts():
    """How many times ONE research-pipeline stage may be attempted before it
    fails closed, counting the first attempt.

    WM corrective patch. Previously fixed at 1 attempt, plus a single repair
    for content-policy violations only — so a stage that got cut off at the
    token limit, emitted one malformed brace, or cited one wrong evidence id
    died outright and cascaded through every stage downstream. Those failures
    are mechanical and stochastic, which is why `research_manager` seemed to
    fail on a different ticker each run rather than on a specific one.

    3 is deliberately small: each attempt is a full local-model call
    (~60s on the measured WM run), and a stage that cannot produce valid JSON
    in three tries has a real problem worth surfacing rather than grinding
    on. Set to 1 to restore the old single-attempt behaviour."""
    return _int("RESEARCH_STAGE_MAX_ATTEMPTS", 3)


def guidance_max_releases():
    """How many recent item-2.02 8-K earnings releases to examine. More than
    one is needed so superseded guidance can be identified as superseded
    rather than simply absent."""
    return _int("GUIDANCE_MAX_RELEASES", 3)


# ---- TSLA DCF validation patch: post-hoc result validation ----
def dcf_allow_negative_terminal_fcff():
    """False by default (fail closed): a NEGATIVE terminal-year FCFF fed into
    the Gordon-growth perpetuity (TV = FCFF_(n+1) / (WACC - g)) does not
    represent a going concern that grows forever -- it silently produces a
    perpetuity value with the WRONG economic meaning (see finance/dcf.py's
    `DcfValidationStatus.NEGATIVE_TERMINAL_FCFF`). Flip to true only if a
    reviewed policy decision explicitly wants a negative-terminal-FCFF
    perpetuity valued anyway rather than flagged invalid."""
    return _bool("DCF_ALLOW_NEGATIVE_TERMINAL_FCFF", False)


def dcf_scenario_monotonicity_check_enabled():
    """Whether run_dcf() checks bull >= base >= bear ordering when scenario
    NAMES are exactly {'base','bull','bear'} AND their assumptions are
    constructed so that ordering is expected (see finance/dcf.py::
    _scenario_monotonicity_check). On by default; the check only ever
    DETECTS and reports a violation -- it never reorders or clamps a value."""
    return _bool("DCF_SCENARIO_MONOTONICITY_CHECK_ENABLED", True)


# ---- Phase H.3 corrective patch (Problem 10): reduced-mode confidence caps ----
# The research-pipeline prompts already INSTRUCT the model to lower its own
# self-reported confidence when material datasets were omitted (see
# finance/research_pipeline.py's _final_synthesizer_prompt and _researcher_
# prompt) -- these two settings are the deterministic BACKSTOP, exactly the
# same "instruction alone is not enough" pattern as the content-policy scan:
# whatever the model reports, it is clamped DOWN (never up, never rejected)
# to these ceilings whenever finance.claim_validation-visible
# 'plan.omitted.*' evidence entries exist in the index for that analysis.
def research_reduced_mode_confidence_cap():
    """Ceiling for the FinalInvestmentSynthesizer's numeric 'confidence'
    (0.0-1.0) when any dataset was omitted from the analysis."""
    return float(_str("RESEARCH_REDUCED_MODE_CONFIDENCE_CAP", "0.6"))


def research_reduced_mode_confidence_enum_cap():
    """Ceiling for the Bull/Bear Researchers' enum 'confidence'
    (low/medium/high) when any dataset was omitted from the analysis."""
    return _str("RESEARCH_REDUCED_MODE_CONFIDENCE_ENUM_CAP", "medium")


# ---- Phase H.3: Yahoo Finance (yfinance) + SEC EDGAR providers ----
# See docs/security/YAHOO_SEC_PROVIDER_REVIEW.md for the full review: yfinance
# is UNOFFICIAL personal-use scraping (no sanctioned API, no key) and requires
# explicit acknowledgement; SEC EDGAR is an official, documented, keyless
# government API gated only by a required identifying User-Agent and a
# published 10 req/s fair-access ceiling. Alpha Vantage remains available but
# is demoted to an explicitly-configured secondary/news provider — it is
# NEVER a silent fallback (automatic_fallback below defaults to false).

def yahoo_finance_enabled():
    return _bool("YAHOO_FINANCE_ENABLED", True)


def yahoo_personal_use_acknowledged():
    """Yahoo tools are not registered at all unless this is explicitly true —
    see docs/security/YAHOO_SEC_PROVIDER_REVIEW.md §1.1. Off by default:
    unlike a reviewed official API, unofficial scraping needs an explicit
    opt-in, not just 'not disabled'."""
    return _bool("YAHOO_PERSONAL_USE_ACKNOWLEDGED", False)


def sec_edgar_enabled():
    return _bool("SEC_EDGAR_ENABLED", True)


def sec_user_agent():
    """SEC rejects requests with no descriptive User-Agent. Read at call time,
    like alphavantage_api_key() — a missing value disables only SEC tools
    (a controlled error), never breaks startup. Never hardcoded: this
    deliberately carries a real contact per SEC's own guidance, so it must
    come from configuration, never source."""
    return _str("SEC_USER_AGENT", None)


def sec_min_request_interval_ms():
    """Minimum spacing between EXTERNAL SEC calls. SEC's fair-access ceiling
    is 10 req/s (100ms); this project paces well under that by default since
    nothing about a single-user analysis needs to approach the limit — see
    docs/security/YAHOO_SEC_PROVIDER_REVIEW.md §2.2."""
    return _int("SEC_MIN_REQUEST_INTERVAL_MS", 250)


def sec_ticker_cik_cache_ttl_seconds():
    """The ticker->CIK mapping file changes rarely; a long TTL avoids
    re-fetching the ~800KB file on every symbol lookup."""
    return _int("SEC_TICKER_CIK_CACHE_TTL_SECONDS", 7 * 24 * 3600)


# ---- Phase H.3: dataset-specific provider selection ----
# Explicit, per-capability, never a single "the" provider — see
# docs/security/YAHOO_SEC_PROVIDER_REVIEW.md §3. Each is independently
# configurable so e.g. disabling Yahoo can be paired with routing quotes to
# Alpha Vantage instead, without code changes.

def finance_quote_provider():
    return _str("FINANCE_QUOTE_PROVIDER", "yahoo")


def finance_price_history_provider():
    return _str("FINANCE_PRICE_HISTORY_PROVIDER", "yahoo")


def finance_corporate_actions_provider():
    return _str("FINANCE_CORPORATE_ACTIONS_PROVIDER", "yahoo")


def finance_us_fundamentals_provider():
    return _str("FINANCE_US_FUNDAMENTALS_PROVIDER", "sec")


def finance_company_profile_provider():
    return _str("FINANCE_COMPANY_PROFILE_PROVIDER", "yahoo")


def finance_analyst_estimates_provider():
    return _str("FINANCE_ANALYST_ESTIMATES_PROVIDER", "yahoo")


def finance_news_provider():
    return _str("FINANCE_NEWS_PROVIDER", "alphavantage")


def finance_secondary_provider():
    return _str("FINANCE_SECONDARY_PROVIDER", "alphavantage")


def finance_automatic_fallback():
    """Whether a primary-provider failure may silently try the secondary
    provider. False by default: per-dataset provider is always reported, and
    a fallback that happened is always visible in the report rather than
    invisible — see docs/security/YAHOO_SEC_PROVIDER_REVIEW.md §3."""
    return _bool("FINANCE_AUTOMATIC_FALLBACK", False)


# ---- Phase G.1: MCP capability detection / server selection ----
def mcp_capability_debug_enabled():
    """Verbose per-request capability/selection logging — off by default so a
    normal request never prints extra MCP diagnostics."""
    return _bool("MCP_CAPABILITY_DEBUG", False)
