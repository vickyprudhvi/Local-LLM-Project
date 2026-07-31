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


# ---- Phase G.1: MCP capability detection / server selection ----
def mcp_capability_debug_enabled():
    """Verbose per-request capability/selection logging — off by default so a
    normal request never prints extra MCP diagnostics."""
    return _bool("MCP_CAPABILITY_DEBUG", False)
