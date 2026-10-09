"""Phase H.35 — the runtime skill loader.

`finance/research_pipeline.py`'s local-LLM stages previously received a
hand-copied paraphrase of the financial-evidence-analysis discipline, baked
into `_SHARED_GUARDRAILS` as a Python string literal. That is not "the
runtime skill" in any meaningful sense: the actual, maintained document lived
at `.claude/skills/financial-evidence-analysis/SKILL.md`, read only by an
interactive Claude Code session, never by the pipeline that actually talks to
the local model.

This module makes `skills/financial-evidence-analysis/SKILL.md` (tracked in
git, moving with the repository) the single source of truth, and reads it
verbatim into the research pipeline's stage prompts. Deliberately minimal:
a deterministic path, a read, a cache, an explicit exception on failure.
No new registry, no network call, no change to router.py or tools/executor.py.
"""

import functools
from pathlib import Path
from typing import Optional

# finance/skill_loader.py -> finance/ -> <repo root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SKILLS_DIR = _PROJECT_ROOT / "skills"


class SkillLoadError(Exception):
    """The runtime skill could not be read.

    Deliberately its own type, never swallowed inside this module -- the
    caller (finance/research_pipeline.py) decides what "cannot load the
    skill" means for the pipeline (fail the interpretive stages closed), and
    that decision must not be made silently in here.
    """


def skill_path(name: str) -> Path:
    """The deterministic, read-only path for a named runtime skill.

    No network, no search path, no environment-variable override -- exactly
    one location, so "did it load" is never ambiguous about WHICH file.
    """
    return _SKILLS_DIR / name / "SKILL.md"


@functools.lru_cache(maxsize=None)
def load_runtime_skill(name: str = "financial-evidence-analysis") -> str:
    """Read a runtime skill's full Markdown text, verbatim.

    Cached after the first successful read for the lifetime of the process --
    this is static project content, not something that changes mid-run, and
    re-reading it once per stage call would be pure overhead across a
    six-stage pipeline. A failed read is NOT cached (`lru_cache` never caches
    an exception), so a transient problem (e.g. a misconfigured working
    directory in a test) does not permanently wedge a process that could
    otherwise recover.

    Raises `SkillLoadError` on any failure (missing file, unreadable file,
    empty file) rather than returning a fallback string -- see this module's
    docstring and `SkillLoadError`'s for why the caller must decide what a
    missing skill means, rather than this function silently degrading.
    """
    path = skill_path(name)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillLoadError(f"could not read runtime skill {name!r} at {path}: {exc}") from exc
    if not text.strip():
        raise SkillLoadError(f"runtime skill {name!r} at {path} is empty")
    return text


def try_load_runtime_skill(name: str = "financial-evidence-analysis") -> Optional[str]:
    """`load_runtime_skill`, but None instead of raising.

    For a caller that wants fail-closed behavior as DATA (a None it then acts
    on by skipping stages) rather than as CONTROL FLOW (a try/except at every
    call site). `finance/research_pipeline.py` uses this at the top of
    `run_research_pipeline`, once per run.
    """
    try:
        return load_runtime_skill(name)
    except SkillLoadError:
        return None
