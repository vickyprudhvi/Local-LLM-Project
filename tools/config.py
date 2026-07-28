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
