"""One unified diagnostic object for compare/debug mode. Spec section 23.

Structured diagnostics only -- no full filing text, no secrets. Mirrors the
shape `finance.extraction.runtime.ExtractionObservation` and
`finance.reported_actuals.runtime.ReportedActualsObservation` already use,
so a reader who knows those two knows this one.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DocumentPipelineDiagnostics:
    mode: str = "v1"
    documents_selected: Dict[str, int] = field(default_factory=dict)
    periods_found: List[str] = field(default_factory=list)
    structured_facts_available: bool = False
    llm_actuals_attempted: bool = False
    llm_actuals_candidates_proposed: int = 0
    llm_actuals_accepted: int = 0
    llm_actuals_rejection_codes: Dict[str, int] = field(default_factory=dict)
    llm_events_attempted: bool = False
    llm_events_candidates_proposed: int = 0
    llm_events_accepted: int = 0
    llm_events_rejection_codes: Dict[str, int] = field(default_factory=dict)
    current_actual_period: Optional[str] = None
    current_guidance_count: Optional[int] = None
    events_funded_count: int = 0
    events_committed_only_count: int = 0
    reporting_currency: Optional[str] = None
    ttm_end: Optional[str] = None
    dcf_base_period: Optional[str] = None
    failure_code: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "documents_selected": dict(self.documents_selected),
            "periods_found": list(self.periods_found),
            "structured_facts_available": self.structured_facts_available,
            "llm_actuals_attempted": self.llm_actuals_attempted,
            "llm_events_attempted": self.llm_events_attempted,
            "events_funded_count": self.events_funded_count,
            "events_committed_only_count": self.events_committed_only_count,
        }
        if self.llm_actuals_attempted:
            payload["llm_actuals_candidates_proposed"] = self.llm_actuals_candidates_proposed
            payload["llm_actuals_accepted"] = self.llm_actuals_accepted
            payload["llm_actuals_rejection_codes"] = dict(self.llm_actuals_rejection_codes)
        if self.llm_events_attempted:
            payload["llm_events_candidates_proposed"] = self.llm_events_candidates_proposed
            payload["llm_events_accepted"] = self.llm_events_accepted
            payload["llm_events_rejection_codes"] = dict(self.llm_events_rejection_codes)
        if self.current_actual_period:
            payload["current_actual_period"] = self.current_actual_period
        if self.current_guidance_count is not None:
            payload["current_guidance_count"] = self.current_guidance_count
        if self.reporting_currency:
            payload["reporting_currency"] = self.reporting_currency
        if self.ttm_end:
            payload["ttm_end"] = self.ttm_end
        if self.dcf_base_period:
            payload["dcf_base_period"] = self.dcf_base_period
        if self.failure_code:
            payload["failure_code"] = self.failure_code
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload
