"""Pure routing logic, no I/O. route_intent(text) -> RouteDecision.

Precedence: manual overrides > tool patterns > Claude domain patterns > default local.
"""

import re
from dataclasses import dataclass
from typing import Optional

OVERRIDE_CLAUDE_RE = re.compile(r"^\s*ask\s+claude\b\s*[:,]?\s*(.*)$", re.IGNORECASE)
OVERRIDE_LOCAL_RE = re.compile(r"^\s*use\s+local\b\s*[:,]?\s*(.*)$", re.IGNORECASE)

REMEMBER_RE = re.compile(r"^(?:remember|note|save this)\b\s*:?\s*(.+)$", re.IGNORECASE)

RECALL_PHRASES = ("what do you remember", "what do you know about me")
TIME_RE = re.compile(r"\bwhat('s| is)?\s+(the\s+)?(time|date)\b|\btoday'?s\s+date\b", re.IGNORECASE)
LOOK_PHRASES = (
    "look at this",
    "take a look",
    "what do you see",
    "check the camera",
    "take a picture",
)

CLAUDE_DOMAIN_RE = re.compile(
    r"\b("
    r"money|invest|investing|investment|investments|investor|investors"
    r"|tax|taxes|mortgage|mortgages|insurance"
    r"|medical|doctor|diagnosis|diagnose|symptom|symptoms"
    r"|legal|lawyer|attorney|lawsuit|contract"
    r"|resume|cv|salary|career|careers"
    r"|should i|think hard|reason carefully"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    mode: str  # "tool" | "claude" | "local"
    tool: Optional[str] = None
    payload: Optional[str] = None


def route_intent(text: str) -> RouteDecision:
    stripped = text.strip()
    lowered = stripped.lower()
    
    override = OVERRIDE_CLAUDE_RE.match(stripped)
    if override:
        return RouteDecision(mode="claude", payload=override.group(1).strip() or stripped)

    override = OVERRIDE_LOCAL_RE.match(stripped)
    if override:
        return RouteDecision(mode="local", payload=override.group(1).strip() or stripped)

    remember = REMEMBER_RE.match(stripped)
    if remember:
        return RouteDecision(mode="tool", tool="remember", payload=remember.group(1).strip())

    if any(phrase in lowered for phrase in RECALL_PHRASES):
        return RouteDecision(mode="tool", tool="recall", payload=stripped)
    
    if TIME_RE.search(stripped):
        return RouteDecision(mode="tool", tool="time", payload=stripped)

    if any(phrase in lowered for phrase in LOOK_PHRASES):
        return RouteDecision(mode="tool", tool="look", payload=stripped)

    if CLAUDE_DOMAIN_RE.search(stripped):
        return RouteDecision(mode="claude", payload=stripped)

    return RouteDecision(mode="local", payload=stripped)
