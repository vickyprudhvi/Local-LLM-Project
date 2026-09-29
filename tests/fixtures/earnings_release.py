"""A generic earnings release: historical tables AND a forward outlook.

Spec §11's rule is that guidance is a PROSPECTIVE statement about a named
period. What made that hard to enforce is that a real earnings release is
mostly the opposite: condensed statements, a reconciliation, a share-count
table, all of them historical, and a short outlook section somewhere near the
end. The two sit within a few hundred characters of each other, and the
forward-looking words that qualify the outlook are also scattered through the
prose around the historical tables.

So this fixture is deliberately built as the hard case rather than the easy
one. It contains, in the order a real release uses:

    a historical revenue / operating-income table   (two reported quarters)
    a historical weighted-average share-count table (the EPS denominators)
    a GAAP-to-non-GAAP reconciliation of the quarter just reported
    an OUTLOOK section with three prospective figures

Only the last three numbers are guidance. Everything above them is a report
of what already happened, and no amount of nearby forward-looking vocabulary
makes a reported actual into an outlook.

No issuer is named. The figures are invented and deliberately round.
"""

# The three prospective figures, as ground truth for the tests. Kept beside
# the text so a change to one is visibly a change to the other.
Q3_REVENUE_OUTLOOK = (1_260.0, 1_300.0)          # $ millions
Q3_OPERATING_MARGIN_OUTLOOK = (0.20, 0.22)       # % of projected revenue
Q3_EBITDA_MARGIN_OUTLOOK = (0.28, 0.30)          # % of projected revenue

# Historical figures that must NEVER become guidance.
REPORTED_Q2_REVENUE = 1_180.4
REPORTED_Q2_OPERATING_INCOME = 212.5
REPORTED_WEIGHTED_AVERAGE_DILUTED_SHARES = 139_933
PRIOR_WEIGHTED_AVERAGE_DILUTED_SHARES = 145_758


EARNINGS_RELEASE_HTML = """<html><body>
<p>Example Holdings Inc. Reports Second Quarter Fiscal 2026 Results</p>

<p>The company today announced results for the second quarter of fiscal 2026.
Management commented that execution was strong across the portfolio and that
the team remains focused on delivering durable growth.</p>

<p>Condensed Consolidated Statements of Operations (unaudited)<br/>
Three Months Ended June 30,<br/>
&nbsp;&nbsp;&nbsp;2026&nbsp;&nbsp;&nbsp;2025<br/>
Revenue&nbsp;&nbsp;$1,180.4&nbsp;&nbsp;$1,042.9<br/>
Cost of revenue&nbsp;&nbsp;472.1&nbsp;&nbsp;428.6<br/>
Operating income&nbsp;&nbsp;212.5&nbsp;&nbsp;171.0<br/>
Net income&nbsp;&nbsp;$164.8&nbsp;&nbsp;$131.2<br/></p>

<p>Weighted-average shares used in computing earnings per share (unaudited)<br/>
Three Months Ended June 30,<br/>
&nbsp;&nbsp;&nbsp;2026&nbsp;&nbsp;&nbsp;2025<br/>
Basic&nbsp;&nbsp;137,204&nbsp;&nbsp;143,015<br/>
Diluted&nbsp;&nbsp;139,933&nbsp;&nbsp;145,758<br/></p>

<p>Reconciliation of GAAP to Non-GAAP Results (unaudited)<br/>
Three Months Ended June 30,<br/>
&nbsp;&nbsp;&nbsp;2026&nbsp;&nbsp;&nbsp;2025<br/>
Operating income (GAAP)&nbsp;&nbsp;212.5&nbsp;&nbsp;171.0<br/>
Stock-based compensation&nbsp;&nbsp;44.9&nbsp;&nbsp;39.2<br/>
Non-GAAP operating income&nbsp;&nbsp;257.4&nbsp;&nbsp;210.2<br/>
Operating margin (GAAP)&nbsp;&nbsp;18.0%&nbsp;&nbsp;16.4%<br/></p>

<p>Financial Outlook</p>

<p>For the third quarter of fiscal 2026, the company expects revenue of
$1,260 million to $1,300 million.</p>

<p>For the third quarter of fiscal 2026, the company expects non-GAAP
operating income to be 20% to 22% of projected revenue.</p>

<p>For the third quarter of fiscal 2026, the company expects Adjusted EBITDA
to be 28% to 30% of projected revenue.</p>

<p>This release contains forward-looking statements within the meaning of the
Private Securities Litigation Reform Act of 1995.</p>
</body></html>"""
