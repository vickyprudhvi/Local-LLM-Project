"""Web-search provider abstraction with a single Phase 2A implementation: Tavily.

The provider is dumb transport: given a query + limit it calls the Tavily REST API
and returns a normalized list of result dicts. It raises ToolFailure with a
controlled SEARCH_* code on any problem. The API key is read from config at call
time (so a missing key is reported as SEARCH_API_KEY_MISSING, and the rest of the
assistant keeps working).
"""

from urllib.parse import urlsplit

import requests

import tools.config as config
from tools.base import ToolFailure
from tools.models import (
    SEARCH_API_KEY_MISSING,
    SEARCH_AUTHENTICATION_FAILED,
    SEARCH_PROVIDER_ERROR,
    SEARCH_RATE_LIMITED,
)

TAVILY_ENDPOINT = "https://api.tavily.com/search"


def _source_of(url):
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


class TavilyProvider:
    name = "tavily"

    def __init__(self, session=None, timeout=15):
        # Session is injectable for tests / connection pooling; created lazily.
        self._session = session
        self.timeout = timeout

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def search(self, query, limit):
        api_key = config.tavily_api_key()
        if not api_key:
            raise ToolFailure(SEARCH_API_KEY_MISSING,
                              "Web search is unavailable: no search API key is configured.")

        payload = {"api_key": api_key, "query": query, "max_results": limit,
                   "search_depth": "basic"}
        try:
            resp = self.session.post(TAVILY_ENDPOINT, json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise ToolFailure(SEARCH_PROVIDER_ERROR, "The search provider timed out.", retryable=True)
        except requests.exceptions.RequestException as e:
            raise ToolFailure(SEARCH_PROVIDER_ERROR, f"The search request failed ({type(e).__name__}).")

        if resp.status_code in (401, 403):
            raise ToolFailure(SEARCH_AUTHENTICATION_FAILED, "Search authentication failed.")
        if resp.status_code == 429:
            raise ToolFailure(SEARCH_RATE_LIMITED, "The search provider is rate limiting requests.",
                              retryable=True)
        if resp.status_code >= 400:
            raise ToolFailure(SEARCH_PROVIDER_ERROR,
                              f"The search provider returned status {resp.status_code}.")

        try:
            data = resp.json()
        except ValueError:
            raise ToolFailure(SEARCH_PROVIDER_ERROR, "The search provider returned an invalid response.")
        if not isinstance(data, dict):
            raise ToolFailure(SEARCH_PROVIDER_ERROR, "The search provider returned an invalid response.")

        raw_results = data.get("results")
        if raw_results is None:
            raw_results = []
        if not isinstance(raw_results, list):
            raise ToolFailure(SEARCH_PROVIDER_ERROR, "The search provider returned an invalid response.")

        results = []
        for item in raw_results[:limit]:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or ""
            results.append({
                "title": (item.get("title") or "").strip(),
                "url": url,
                "snippet": (item.get("content") or "").strip(),
                "source": _source_of(url),
            })
        return results


def get_provider(session=None):
    """Return the configured search provider. Phase 2A supports Tavily only."""
    name = (config.search_provider() or "tavily").lower()
    if name != "tavily":
        # Unknown provider names fall back to Tavily rather than breaking startup.
        pass
    return TavilyProvider(session=session)
