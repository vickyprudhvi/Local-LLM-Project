"""Phase E — build a minimal, isolated child environment for the MCP subprocess.

The full parent environment is NEVER inherited. The child gets only:
  - a small set of platform-required process variables,
  - variables the config explicitly allowlists (by NAME; values are read from the
    parent at runtime and never stored in config),
  - launcher-controlled extras (e.g. PYTHONPATH so `python -m <pkg>` resolves).

Known-secret variables are never copied unless explicitly allowlisted by the
operator. Only variable NAMES are ever logged — never values.
"""

import os

# Process variables commonly required for a child to run correctly across platforms.
_PLATFORM_KEEP = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "ComSpec",
    "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
    "HOME", "USERPROFILE", "HOMEPATH", "HOMEDRIVE",
)

# Never copied automatically; only ever present if explicitly allowlisted by the operator.
SECRET_DENYLIST = frozenset({
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID", "AZURE_CLIENT_SECRET", "DATABASE_URL", "SSH_AUTH_SOCK",
    "GOOGLE_APPLICATION_CREDENTIALS", "TAVILY_API_KEY",
})


def build_child_environment(allowlist, extra=None, parent=None):
    """Return the minimal env dict for the MCP subprocess and the names it contains."""
    parent = os.environ if parent is None else parent
    env = {}
    for key in _PLATFORM_KEEP:
        if key in parent:
            env[key] = parent[key]
    for name in allowlist or ():
        if name in parent:
            env[name] = parent[name]
    if extra:
        env.update(extra)
    return env


def allowlisted_secret_names(allowlist):
    """Names in the allowlist that are known secrets — surfaced for an operator warning."""
    return [n for n in (allowlist or ()) if n in SECRET_DENYLIST]
