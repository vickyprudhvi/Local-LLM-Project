"""SSRF-safe HTTP fetching for browser.fetch_page.

Every network hop (initial request and every redirect) is validated the same way:
scheme/port/credential checks on the URL, DNS resolution, and an IP blocklist that
rejects loopback/private/link-local/multicast/reserved/unspecified addresses,
IPv4-mapped-private IPv6, cloud-metadata endpoints, and anything not globally
routable. Reads are bounded by byte limit and timeouts; JavaScript is never
executed and no headless browser is used.

Known limitation: there is a small TOCTOU window between DNS resolution and the
connection (DNS rebinding). Acceptable for Phase 2A read-only fetches; documented.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import requests

from tools.base import ToolFailure
from tools.models import (
    FETCH_TIMEOUT,
    INVALID_RESPONSE,
    INVALID_URL,
    PRIVATE_NETWORK_BLOCKED,
    REDIRECT_BLOCKED,
    TOO_MANY_REDIRECTS,
    UNSUPPORTED_URL_SCHEME,
)

# Schemes we will never fetch, even if someone flips a config flag.
_FORBIDDEN_SCHEMES = {"file", "ftp", "data", "javascript", "gopher", "ssh", "ws", "wss", "mailto"}
_ALLOWED_PORTS = {80, 443}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class FetchError(ToolFailure):
    """A controlled fetch failure carrying a ToolError code."""

    def __init__(self, code: str, message: str):
        super().__init__(code, message)


def _blocked_ip(ip_str: str) -> bool:
    """True if the address must not be contacted (SSRF blocklist)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable -> treat as unsafe
    # Unwrap IPv4-mapped IPv6 (e.g. ::ffff:10.0.0.1) and judge the embedded v4.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified):
        return True
    # Belt and suspenders: anything not globally routable is blocked.
    return not ip.is_global


def _resolve(host: str):
    """Resolve a hostname to a list of IP strings. Patchable in tests."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    return list({info[4][0] for info in infos})


def validate_url(url: str, allow_http: bool):
    """Validate scheme/credentials/host/port. Returns urlsplit result or raises FetchError."""
    if not isinstance(url, str) or not url.strip():
        raise FetchError(INVALID_URL, "A non-empty URL string is required.")
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "").lower()

    if not scheme:
        raise FetchError(INVALID_URL, "The URL must include a scheme (https://).")
    if scheme in _FORBIDDEN_SCHEMES:
        raise FetchError(UNSUPPORTED_URL_SCHEME, f"The URL scheme {scheme!r} is not allowed.")
    if scheme == "http" and not allow_http:
        raise FetchError(UNSUPPORTED_URL_SCHEME, "HTTP is disabled; use HTTPS.")
    if scheme not in ("http", "https"):
        raise FetchError(UNSUPPORTED_URL_SCHEME, f"Only HTTP(S) is supported, not {scheme!r}.")
    if parts.username or parts.password:
        raise FetchError(INVALID_URL, "URLs with embedded credentials are not allowed.")
    if not parts.hostname:
        raise FetchError(INVALID_URL, "The URL has no host.")
    try:
        port = parts.port
    except ValueError:
        raise FetchError(INVALID_URL, "The URL has an invalid port.")
    if port is not None and port not in _ALLOWED_PORTS:
        raise FetchError(INVALID_URL, f"Port {port} is not permitted (only 80/443).")
    return parts


def _guard_host(host: str, is_redirect: bool):
    """Resolve and block private/loopback/etc. addresses before connecting."""
    ips = _resolve(host)
    if not ips:
        raise FetchError(INVALID_URL, f"Could not resolve host {host!r}.")
    for ip in ips:
        if _blocked_ip(ip):
            code = REDIRECT_BLOCKED if is_redirect else PRIVATE_NETWORK_BLOCKED
            raise FetchError(code, "Access to private, local, or non-public addresses is blocked.")


def safe_get(url, session, user_agent, connect_timeout, read_timeout, max_redirects, allow_http):
    """Fetch a URL with SSRF checks on every hop. Returns (response, final_url).

    The returned response is a streaming requests.Response (stream=True) — the
    caller is responsible for bounded reading and closing it.
    """
    current = url
    headers = {"User-Agent": user_agent, "Accept": "text/html,text/plain,application/xhtml+xml,*/*"}

    for hop in range(max_redirects + 1):
        parts = validate_url(current, allow_http)
        _guard_host(parts.hostname, is_redirect=(hop > 0))
        try:
            resp = session.get(
                current,
                headers=headers,
                allow_redirects=False,
                stream=True,
                timeout=(connect_timeout, read_timeout),
            )
        except requests.exceptions.Timeout:
            raise FetchError(FETCH_TIMEOUT, "The request timed out.")
        except requests.exceptions.RequestException as e:
            raise FetchError(INVALID_RESPONSE, f"The request failed ({type(e).__name__}).")

        if resp.status_code in _REDIRECT_STATUSES:
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise FetchError(INVALID_RESPONSE, "Redirect response had no Location header.")
            current = urljoin(current, location)
            continue

        return resp, current

    raise FetchError(TOO_MANY_REDIRECTS, f"Exceeded the maximum of {max_redirects} redirects.")


def read_limited(resp, max_bytes):
    """Read a streaming response body up to max_bytes. Raises RESPONSE_TOO_LARGE if exceeded."""
    from tools.models import RESPONSE_TOO_LARGE

    declared = resp.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                resp.close()
                raise FetchError(RESPONSE_TOO_LARGE, "The response exceeds the maximum allowed size.")
        except ValueError:
            pass  # ignore a malformed Content-Length; the streaming cap still applies

    chunks = []
    total = 0
    try:
        for chunk in resp.iter_content(chunk_size=16384):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise FetchError(RESPONSE_TOO_LARGE, "The response exceeds the maximum allowed size.")
            chunks.append(chunk)
    except requests.exceptions.Timeout:
        raise FetchError(FETCH_TIMEOUT, "The request timed out while reading the body.")
    except requests.exceptions.RequestException as e:
        raise FetchError(INVALID_RESPONSE, f"Reading the response failed ({type(e).__name__}).")
    finally:
        resp.close()
    return b"".join(chunks), total
