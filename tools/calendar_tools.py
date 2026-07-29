"""Router-dispatched built-in: calendar.read (read-only Google Calendar).

Wraps the existing calendar_reader.get_events. Read-only — no write path and no
new confirmation gate is introduced here (permissions are a later phase). The
tool returns raw events wrapped in a summarize directive; the ask_local phrasing
step stays in the orchestrator. llm_callable=False: router-selected only.
"""

import calendar_reader
from tools.base import BaseTool

# Moved verbatim from the former assistant.dispatch calendar branch.
CALENDAR_SUMMARY_PROMPT = (
    "You're telling someone what's on their calendar, out loud. Summarize the events in a "
    "natural, conversational sentence or two — no bullet points, no markdown, no raw "
    "timestamps. Use plain phrasing for dates and times, like 'today at 2pm' or 'next "
    "Wednesday'. Be concise."
)


class CalendarReadTool(BaseTool):
    name = "calendar.read"
    description = "Read upcoming or past events from the user's Google Calendar."
    input_schema = {
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "Earliest date, YYYY-MM-DD."},
            "end_date": {"type": "string", "description": "Latest date, YYYY-MM-DD (inclusive)."},
        },
    }
    timeout_seconds = 30.0
    llm_callable = False

    def execute(self, arguments: dict) -> dict:
        try:
            events = calendar_reader.get_events(
                start_date=arguments.get("start_date"), end_date=arguments.get("end_date"), n=10
            )
        except Exception as e:  # noqa: BLE001 — surface a safe, credential-free message
            return {"render": "speak", "text": f"Sorry, I couldn't reach your calendar: {e}"}
        if not events:
            return {"render": "speak", "text": "Nothing found on your calendar for that range."}
        content = "Here's what's on your calendar: " + "; ".join(
            f"{e['start']} - {e['summary']}" for e in events
        )
        return {"render": "summarize", "content": content, "instructions": CALENDAR_SUMMARY_PROMPT}
