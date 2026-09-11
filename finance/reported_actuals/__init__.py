"""Reported Actuals Source Integration V2 — making the right documents reachable.

THE PROBLEM THIS SOLVES, AND THE ONE IT DOES NOT

The Actualization V2 resolver already chooses correctly between a newer
complete Q4/FY earnings release and an older Q3 10-Q. It was benchmarked at
100% latest-period accuracy over nineteen historical sequences with every hard
safety counter at zero.

What it could not do was ever SEE the earnings release. Production discovery
reads SEC CompanyFacts, and an earnings-release exhibit attached to an 8-K is
almost never represented there: across eight live issuers, not one had an 8-K
period newer than its newest periodic filing. The resolver was choosing
correctly among candidates that could not include the one document that
mattered.

So this package does exactly one thing: it finds filed earnings-release
exhibits, reads their reported-result tables deterministically, and offers the
result as `ReportedActualCandidate` objects. It decides nothing about which
period is current. That remains the Actualization resolver's job, and adding a
second opinion here is the failure this whole architecture is organised
against.

    discovery.py   which FILING, and which DOCUMENT inside it
    tables.py      the document's tables, with rows and columns intact
    facts.py       one table cell to one identified financial fact
    candidates.py  facts to a ReportedActualCandidate the resolver can rank
    runtime.py     the mode seam: v1 / compare / v2

WHY DETERMINISTIC AND NOT A MODEL

A reported financial statement is a TABLE. Its meaning is carried by
structure -- the row says which line, the column says which period, the
caption says the units and the basis -- and structure is exactly what a
deterministic reader can follow and a prose reader cannot. Guidance needed a
semantic reader because an outlook is a sentence; reported actuals do not,
and introducing one here would put a model in the authority position for
historical financial statements, which section 7 of the brief rules out.

FAIL CLOSED, EVERYWHERE

Every layer refuses rather than guesses. An unknown scale, an unresolvable
period, a row whose values do not line up with its columns, a currency that
cannot be established -- each produces a structured rejection with a code, and
the fact does not exist. A source layer that invents one number destroys the
value of every number it did not invent.
"""

from finance.reported_actuals.candidates import (  # noqa: F401
    ActualFinancialFact,
    ReportedActualsExtraction,
    build_candidates,
    extract_reported_actuals,
)
from finance.reported_actuals.discovery import (  # noqa: F401
    EarningsSourceCandidate,
    SourceDiscoveryCode,
    find_reported_actual_filings,
    select_results_document,
)
from finance.reported_actuals.tables import (  # noqa: F401
    FilingTable,
    PeriodColumn,
    StatementKind,
    parse_filing_tables,
)
