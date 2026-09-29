"""An earnings release that guides TWO horizons at once.

This is the ordinary case, not an exotic one: a company reporting Q4 gives
next-quarter guidance and full-year guidance in the same paragraph, and both
remain current until the next release. A resolver that keeps one statement
per metric name can hold only one of them, and the one it drops is invisible
— the report says "guidance: Q1 FY2027" and nothing indicates that a
full-year outlook was published beside it.

Four statements, deliberately overlapping on metric name:

    revenue_growth          Q1 FY2027   27% to 29%
    revenue (absolute)      FY2027      $118.0B to $120.0B
    adjusted EPS            Q1 FY2027   $1.60 to $1.70
    adjusted EPS            FY2027      $7.10 to $7.40

The two EPS statements share a metric name and differ only in target period,
which is exactly the collision `guidance_identity` was written to resolve.

Every figure is invented. No issuer is named.
"""

Q1_REVENUE_GROWTH = (0.27, 0.29)
FY_REVENUE_ABSOLUTE = (118.0, 120.0)          # $ billions
Q1_ADJUSTED_EPS = (1.60, 1.70)
FY_ADJUSTED_EPS = (7.10, 7.40)

# The prior fiscal year's reported revenue, so an annual growth rate can be
# derived from the absolute outlook: 119.0 / 89.0 - 1 = ~33.7%.
PRIOR_FY_REVENUE = 89.0
IMPLIED_FY_GROWTH = (FY_REVENUE_ABSOLUTE[0] + FY_REVENUE_ABSOLUTE[1]) / 2 / PRIOR_FY_REVENUE - 1


MULTI_HORIZON_RELEASE = """
Example Systems Inc. Reports Fourth Quarter and Fiscal Year 2026 Results

Financial Outlook

For the first quarter of fiscal 2027, the company expects revenue growth of
27% to 29% compared with the first quarter of fiscal 2026.

For the first quarter of fiscal 2027, the company expects adjusted earnings
per share of $1.60 to $1.70.

For the full year 2027, the company expects revenue of $118.0 billion to
$120.0 billion.

For the full year 2027, the company expects adjusted earnings per share of
$7.10 to $7.40.

This release contains forward-looking statements within the meaning of the
Private Securities Litigation Reform Act of 1995.
"""
