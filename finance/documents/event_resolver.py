"""Combining validated events into one summary. Deterministic, and thin.

Everything that decides whether ONE event is funded, committed, or what type
it is already happened in `event_validator.py`. This module only aggregates
a package's validated events for diagnostics and for the report -- it is the
boundary named in the package docstring: Phase 1 surfaces events as evidence,
it does not adjust `finance/net_debt.py` or any DCF input from them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from finance.documents.event_schema import ResolvedFinancingEvent


@dataclass
class FinancingEventSummary:
    events: List[ResolvedFinancingEvent] = field(default_factory=list)
    funded_events: List[ResolvedFinancingEvent] = field(default_factory=list)
    committed_only_events: List[ResolvedFinancingEvent] = field(default_factory=list)
    by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_count": len(self.events),
            "funded_count": len(self.funded_events),
            "committed_only_count": len(self.committed_only_events),
            "by_type": dict(self.by_type),
            "events": [e.to_dict() for e in self.events],
        }


def summarize_events(events: Sequence[ResolvedFinancingEvent]) -> FinancingEventSummary:
    summary = FinancingEventSummary(events=list(events))
    for event in events:
        summary.by_type[event.event_type] = summary.by_type.get(event.event_type, 0) + 1
        if event.funded:
            summary.funded_events.append(event)
        elif event.committed:
            summary.committed_only_events.append(event)
    return summary
