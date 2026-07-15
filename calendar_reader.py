"""Read-only Google Calendar access. get_events(...) returns events across all
of the user's calendars within an optional [start_date, end_date] window
(YYYY-MM-DD, inclusive). Callers pass concrete dates rather than a category
like "today"/"upcoming" — the router LLM resolves the user's relative date
("next Tuesday", "June 15") into dates itself, since only it has the
conversational context to do that; this module just executes the query.

Auth: uses the OAuth "installed app" flow against credentials.json (downloaded
from Google Cloud Console). The first call opens a browser for consent; the
resulting token is cached in token.json so later calls don't prompt again.
Both files hold secrets and are gitignored — never commit them.
"""

import datetime
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.json"
PAST_WINDOW_DAYS = 30  # default lookback when only end_date is given, so the query stays bounded


def _get_credentials():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return creds


def _time_bounds(start_date, end_date):
    now = datetime.datetime.now(datetime.timezone.utc)

    if start_date:
        time_min = datetime.datetime.fromisoformat(start_date).replace(tzinfo=datetime.timezone.utc)
    elif end_date:
        end_dt = datetime.datetime.fromisoformat(end_date)
        time_min = (end_dt - datetime.timedelta(days=PAST_WINDOW_DAYS)).replace(tzinfo=datetime.timezone.utc)
    else:
        time_min = now

    if end_date:
        time_max = datetime.datetime.fromisoformat(end_date).replace(
            tzinfo=datetime.timezone.utc
        ) + datetime.timedelta(days=1)
    else:
        time_max = None

    return time_min.isoformat(), (time_max.isoformat() if time_max else None)


def get_events(start_date=None, end_date=None, n=10):
    """Return up to n events across all calendars within [start_date, end_date]
    (YYYY-MM-DD, end_date inclusive), ordered by start time ascending.

    Both bounds are optional:
      - neither given: open-ended, from now into the future
      - only start_date: from that date onward, open-ended future
      - only end_date: from PAST_WINDOW_DAYS days before end_date through end_date
      - both given: exactly that window

    Each event is {"summary", "start", "end", "calendar"}.
    """
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    calendar_list = service.calendarList().list().execute().get("items", [])
    time_min, time_max = _time_bounds(start_date, end_date)

    events = []
    for cal in calendar_list:
        cal_id = cal["id"]
        list_kwargs = {
            "calendarId": cal_id,
            "maxResults": n,
            "singleEvents": True,
            "orderBy": "startTime",
            "timeMin": time_min,
        }
        if time_max:
            list_kwargs["timeMax"] = time_max

        result = service.events().list(**list_kwargs).execute()
        for event in result.get("items", []):
            start = event["start"].get("dateTime", event["start"].get("date"))
            end = event["end"].get("dateTime", event["end"].get("date"))
            events.append(
                {
                    "summary": event.get("summary", "(no title)"),
                    "start": start,
                    "end": end,
                    "calendar": cal.get("summary", cal_id),
                }
            )

    events.sort(key=lambda e: e["start"])
    return events[:n]


if __name__ == "__main__":
    import sys

    start_arg = sys.argv[1] if len(sys.argv) > 1 else None
    end_arg = sys.argv[2] if len(sys.argv) > 2 else None
    for e in get_events(start_arg, end_arg):
        print(f"{e['start']}  {e['summary']}  [{e['calendar']}]")
