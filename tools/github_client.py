"""Shared, read-only GitHub REST API client for the Phase 2A github.* tools.

One requests.Session with a User-Agent and Accept header, an optional read-only
Bearer token loaded ONLY from GITHUB_TOKEN, a request timeout, and conversion of
HTTP/network failures into controlled ToolFailure codes. The token is never
logged, never returned, never placed in a URL, and never accepted as a tool
argument.
"""

import requests

import tools.config as config
from tools.base import ToolFailure
from tools.models import (
    GITHUB_API_ERROR,
    GITHUB_AUTHENTICATION_FAILED,
    GITHUB_RATE_LIMITED,
    GITHUB_REPOSITORY_NOT_FOUND,
    INVALID_RESPONSE,
)

API_BASE = "https://api.github.com"


class GitHubResponse:
    """A validated GitHub response: parsed JSON plus safe rate-limit metadata."""

    def __init__(self, status_code, data, rate_limit):
        self.status_code = status_code
        self.data = data
        self.rate_limit = rate_limit  # {"remaining": int|None, "reset_at": iso|None}


class GitHubClient:
    def __init__(self, session=None, token=None):
        self._session = session
        # Token resolved at construction if not given; None => unauthenticated.
        self._token = token

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _headers(self):
        headers = {
            "User-Agent": config.http_user_agent(),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = self._token if self._token is not None else config.github_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _rate_limit(resp):
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        reset_at = None
        if reset is not None:
            try:
                from datetime import datetime, timezone
                reset_at = datetime.fromtimestamp(int(reset), tz=timezone.utc).isoformat()
            except (ValueError, OverflowError, OSError):
                reset_at = None
        try:
            remaining = int(remaining) if remaining is not None else None
        except ValueError:
            remaining = None
        return {"remaining": remaining, "reset_at": reset_at}

    def get(self, path, params=None):
        """GET {API_BASE}{path}. Returns GitHubResponse or raises ToolFailure.

        404 is returned (not raised) so callers can map it to a resource-specific
        not-found code; auth/rate-limit/server errors raise ToolFailure.
        """
        url = f"{API_BASE}{path}"
        try:
            resp = self.session.get(url, headers=self._headers(), params=params,
                                    timeout=config.github_timeout())
        except requests.exceptions.Timeout:
            raise ToolFailure(GITHUB_API_ERROR, "The GitHub request timed out.", retryable=True)
        except requests.exceptions.RequestException as e:
            raise ToolFailure(GITHUB_API_ERROR, f"The GitHub request failed ({type(e).__name__}).")

        rate_limit = self._rate_limit(resp)

        if resp.status_code == 401:
            raise ToolFailure(GITHUB_AUTHENTICATION_FAILED, "GitHub authentication failed.")
        if resp.status_code == 403:
            # 403 with remaining==0 is a rate limit; otherwise a forbidden/abuse response.
            if rate_limit["remaining"] == 0:
                raise ToolFailure(GITHUB_RATE_LIMITED,
                                  "GitHub API rate limit exceeded.", retryable=True,
                                  log_meta={"rate_limit_remaining": 0,
                                            "rate_limit_reset_at": rate_limit["reset_at"]})
            raise ToolFailure(GITHUB_RATE_LIMITED,
                              "GitHub denied the request (rate limit or abuse detection).",
                              retryable=True)
        if resp.status_code == 404:
            return GitHubResponse(404, None, rate_limit)
        if resp.status_code >= 400:
            raise ToolFailure(GITHUB_API_ERROR,
                              f"GitHub returned status {resp.status_code}.")

        try:
            data = resp.json()
        except ValueError:
            raise ToolFailure(INVALID_RESPONSE, "GitHub returned an invalid response.")

        return GitHubResponse(resp.status_code, data, rate_limit)


def not_found(message):
    return ToolFailure(GITHUB_REPOSITORY_NOT_FOUND, message)
