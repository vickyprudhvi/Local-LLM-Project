"""Phase H.5 (validation rework), Phase 0 — persist a research-pipeline run.

Prerequisite infrastructure, not a rework step. Two later acceptance criteria
("replay existing runs against the new schema", "replay the WM case that went
1 to 4 to 3") assume stored stage output, and nothing stored it:
`logs/interactions.jsonl` records tool calls only, so a search for
`research_manager` across 4,117 records returns zero.

That gap has a cost. The WM escalation this whole rework responds to was
observed once, in a transient console buffer, and could never be re-examined
— every subsequent question about it had to be answered by re-running the
model and hoping it recurred.

FAILED runs are captured too, including the text that failed. A run that
completed cleanly is the least interesting thing to keep; the failing field
is the artifact anyone actually wants.

OFF by default (`RESEARCH_RUN_ARTIFACTS_ENABLED`). When off, nothing here
touches the filesystem and no validation path changes.
"""

import datetime
import json
import os
import re
from typing import Optional

import tools.config as config

# Filenames are built from a symbol and a timestamp, both of which reach this
# module from callers. Constrained to a conservative character set so a
# malformed symbol cannot produce a path outside the artifact directory.
_SAFE_SYMBOL = re.compile(r"[^A-Za-z0-9._-]")


def _safe_symbol(symbol) -> str:
    cleaned = _SAFE_SYMBOL.sub("", str(symbol or "UNKNOWN"))
    return cleaned[:16] or "UNKNOWN"


def artifact_path(symbol, timestamp: Optional[str] = None, directory: Optional[str] = None) -> str:
    """`logs/research_runs/AOS-20260813T142530Z.json`."""
    directory = directory or config.research_run_artifacts_dir()
    stamp = timestamp or datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(directory, f"{_safe_symbol(symbol)}-{stamp}.json")


def build_artifact(symbol, pipeline_result, evidence_index_size=None) -> dict:
    """The serializable record. Pure -- no filesystem access, so it can be
    asserted on directly in tests without writing anything."""
    checkpoints = [c.to_dict() for c in (pipeline_result.checkpoints or [])]
    return {
        "schema": "research_run_artifact/1",
        "symbol": symbol,
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "available": bool(getattr(pipeline_result, "available", False)),
        "evidence_index_size": (evidence_index_size
                                if evidence_index_size is not None
                                else getattr(pipeline_result, "evidence_index_size", None)),
        "total_prompt_tokens": getattr(pipeline_result, "total_prompt_tokens", None),
        "total_completion_tokens": getattr(pipeline_result, "total_completion_tokens", None),
        # Every checkpoint verbatim: status, output, error, quarantines,
        # findings, failure_detail, tokens, duration. This IS the replay
        # surface -- trimming it here would recreate the gap this exists for.
        "checkpoints": checkpoints,
    }


def write_artifact(symbol, pipeline_result, evidence_index_size=None,
                   directory: Optional[str] = None) -> Optional[str]:
    """Write the artifact when enabled; return its path, or None.

    Never raises: a diagnostic capture failing must not take down an analysis
    that otherwise succeeded. A write error is swallowed deliberately -- the
    caller has no useful recovery, and the alternative is losing a completed
    report because a log directory was read-only.
    """
    if not config.research_run_artifacts_enabled():
        return None
    if pipeline_result is None:
        return None
    directory = directory or config.research_run_artifacts_dir()
    path = artifact_path(symbol, directory=directory)
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(build_artifact(symbol, pipeline_result, evidence_index_size),
                      handle, indent=1, sort_keys=True, default=str)
    except OSError:
        return None
    return path


def load_artifact(path) -> dict:
    """Read one artifact back. The replay entry point."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
