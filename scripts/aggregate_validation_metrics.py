"""Phase H.5, Phase 5b — which validation patterns actually earn their place.

    venv/Scripts/python.exe scripts/aggregate_validation_metrics.py [LOG_PATH]

Reads the `research_validation` records finance/workflow.py writes after every
research-pipeline run and reports, per rule: how often it fired, on how many
distinct tickers, in which fields, on what text, and whether the runs it fired
on survived anyway.

WHY THIS EXISTS

47 patterns accumulated over five rounds of live failures, and nobody could
say which ones mattered. Gating behaviour cannot answer that: a pattern that
never fires and a pattern that fires constantly look identical from outside --
both just look like "no failures". The only way to tell them apart is to count.

The most important column is the one at the bottom: rules that have NEVER
fired. Those are the deletion candidates. Nothing is deleted by this script;
it reports, and the criterion below decides when a report is trustworthy
enough to act on.

DELETION CRITERION (amendment A5)

A pattern is deletable when it has fired zero times across at least
MIN_RUNS runs covering at least MIN_TICKERS distinct tickers. Coverage, not
elapsed time -- this tool runs occasionally, so a calendar window says
nothing about how much text the patterns have actually seen.
"""

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import finance.claim_validation as cv
import finance.content_policy as cp

MIN_RUNS = 30
MIN_TICKERS = 10

DEFAULT_LOG = os.environ.get("INTERACTION_LOG_PATH", "logs/interactions.jsonl")


def read_validation_records(path):
    """Every `research_validation` record, oldest first. Malformed lines are
    skipped rather than fatal -- a truncated final line from an interrupted
    run must not make the whole history unreadable."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") == "research_validation":
                records.append(record)
    return records


def aggregate(records):
    """Per-rule totals plus run-level coverage."""
    stats = collections.defaultdict(lambda: {
        "fires": 0, "fatal": 0, "quarantined": 0,
        "symbols": set(), "fields": collections.Counter(),
        "spans": collections.Counter(), "runs_that_passed": 0,
    })
    symbols = set()
    passed_runs = 0
    for record in records:
        symbol = record.get("symbol")
        symbols.add(symbol)
        run_passed = bool(record.get("passed"))
        passed_runs += run_passed
        # One count per RULE per RUN, not per occurrence: a rule firing three
        # times in one report is one piece of evidence about that rule, not
        # three.
        seen_this_run = set()
        for finding in record.get("findings") or []:
            rule_id = finding.get("rule_id")
            if not rule_id:
                continue
            entry = stats[rule_id]
            # WHERE it fired is always recorded, even on a repeat within the
            # same run -- the field distribution is what tells you whether a
            # pattern is a `thesis` problem or a `balanced_assessment` one.
            entry["fields"][finding.get("field_path", "?")] += 1
            span = (finding.get("matched_span") or "").strip()
            if span:
                entry["spans"][span] += 1
            # HOW OFTEN is counted once per run. A rule firing three times in
            # one report is one piece of evidence about that rule, not three;
            # otherwise a single verbose report could make a pattern look
            # essential.
            if rule_id in seen_this_run:
                continue
            seen_this_run.add(rule_id)
            entry["fires"] += 1
            entry["symbols"].add(symbol)
            if finding.get("outcome") == "fatal":
                entry["fatal"] += 1
            else:
                entry["quarantined"] += 1
            entry["runs_that_passed"] += run_passed
    return stats, {"runs": len(records), "symbols": symbols, "passed": passed_runs}


def all_rules():
    return list(cp.RULES) + list(cv.RULES)


def report(stats, coverage):
    rules = all_rules()
    by_id = {r.rule_id: r for r in rules}

    print(f"runs analysed      : {coverage['runs']}")
    print(f"distinct tickers   : {len(coverage['symbols'])}")
    print(f"runs reaching a verdict: {coverage['passed']}")
    print(f"rules defined      : {len(rules)}")
    print()

    fired = [(rid, s) for rid, s in stats.items() if s["fires"]]
    fired.sort(key=lambda pair: (-pair[1]["fires"], pair[0]))

    if fired:
        print("PATTERNS THAT FIRED")
        print(f"  {'rule':8} {'fires':>5} {'fatal':>5} {'tick':>5}  {'label':40} example")
        for rule_id, entry in fired:
            rule = by_id.get(rule_id)
            label = rule.label[:40] if rule else "(unknown rule)"
            span = entry["spans"].most_common(1)
            example = f"{span[0][0]!r}" if span else ""
            print(f"  {rule_id:8} {entry['fires']:>5} {entry['fatal']:>5} "
                  f"{len(entry['symbols']):>5}  {label:40} {example}")
        print()
        print("  fields most often affected:")
        combined = collections.Counter()
        for _rid, entry in fired:
            combined.update(entry["fields"])
        for field, count in combined.most_common(8):
            print(f"    {count:>4}  {field}")
        print()
    else:
        print("PATTERNS THAT FIRED: none\n")

    silent = sorted(r.rule_id for r in rules if not stats.get(r.rule_id, {}).get("fires"))
    print(f"PATTERNS THAT HAVE NEVER FIRED ({len(silent)} of {len(rules)})")
    for rule_id in silent:
        rule = by_id[rule_id]
        print(f"  {rule_id:8} {rule.severity:14} {rule.label}")
    print()

    enough = (coverage["runs"] >= MIN_RUNS
              and len(coverage["symbols"]) >= MIN_TICKERS)
    if enough:
        print(f"COVERAGE MET ({coverage['runs']} runs, {len(coverage['symbols'])} tickers). "
              "The never-fired list above is actionable.")
        print("Overstatement rules that never fired are deletion candidates. "
              "Fabrication rules are NOT -- a rule guarding a sentence the model "
              "has not happened to write is still guarding it.")
    else:
        print(f"COVERAGE NOT MET: need {MIN_RUNS} runs across {MIN_TICKERS} tickers, "
              f"have {coverage['runs']} across {len(coverage['symbols'])}.")
        print("The never-fired list is not yet evidence of anything. Keep using the tool.")
    return 0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    records = read_validation_records(path)
    if not records:
        print(f"No research_validation records in {path}.")
        print("These are written after each research-pipeline run. Run some analyses first.")
        return 0
    stats, coverage = aggregate(records)
    return report(stats, coverage)


if __name__ == "__main__":
    sys.exit(main())
