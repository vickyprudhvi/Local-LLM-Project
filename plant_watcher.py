"""Standalone scheduled snapshot script. Every 30 min: snapshot. Once/day: compare yesterday's first and last photo."""

import argparse
import glob
import os
import time
from datetime import datetime, timedelta

from rich.console import Console

import brain
import eyes

console = Console()

SNAPSHOT_DIR = "snapshots"
LOG_PATH = os.path.join(SNAPSHOT_DIR, "plant_log.txt")
INTERVAL_SECONDS = 30 * 60

DESCRIBE_PROMPT = "Describe this plant's appearance in detail: size, color, leaves, any wilting or new growth."
COMPARE_SYSTEM_PROMPT = "You are a plant-watching assistant. Be concise and factual."
COMPARE_PROMPT = (
    "Here are two descriptions of the same plant, taken earlier and later on the same day.\n\n"
    "Earlier: {early}\n\n"
    "Later: {late}\n\n"
    "Compare them and summarize what, if anything, changed."
)


def _snapshot_path(now=None):
    now = now or datetime.now()
    return os.path.join(SNAPSHOT_DIR, f"plant_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")


def take_snapshot():
    path = _snapshot_path()
    eyes.snapshot(path=path)
    console.print(f"[dim]snapshot saved: {path}[/dim]")
    return path


def _photos_for_date(date_str):
    pattern = os.path.join(SNAPSHOT_DIR, f"plant_{date_str}_*.jpg")
    return sorted(glob.glob(pattern))


def _most_recent_date_with_photos():
    files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "plant_*.jpg")))
    dates = sorted({os.path.basename(f).split("_")[1] for f in files})
    return dates[-1] if dates else None


def run_daily_compare(date_str=None):
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    photos = _photos_for_date(date_str)
    if not photos:
        date_str = _most_recent_date_with_photos()
        if date_str is None:
            console.print("[red]No plant photos found to compare.[/red]")
            return
        photos = _photos_for_date(date_str)

    if len(photos) < 2:
        console.print(f"[yellow]Only {len(photos)} photo(s) for {date_str} — need at least 2 to compare.[/yellow]")
        return

    first_photo, last_photo = photos[0], photos[-1]
    console.print(f"[dim]comparing {first_photo} -> {last_photo}[/dim]")

    early_desc = eyes.describe_local(first_photo, DESCRIBE_PROMPT)
    late_desc = eyes.describe_local(last_photo, DESCRIBE_PROMPT)

    prompt = COMPARE_PROMPT.format(early=early_desc, late=late_desc)
    summary, _metrics = brain.ask_local(prompt, [], COMPARE_SYSTEM_PROMPT)

    console.print(f"[green]{summary}[/green]")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} ({date_str}): {summary}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-now", action="store_true", help="Run the daily-compare path immediately and exit.")
    args = parser.parse_args()

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    if args.compare_now:
        run_daily_compare()
        return

    take_snapshot()
    last_compare_date = None

    while True:
        time.sleep(INTERVAL_SECONDS)
        take_snapshot()

        today = datetime.now().date()
        if last_compare_date != today:
            run_daily_compare()
            last_compare_date = today


if __name__ == "__main__":
    main()
