"""Parts 18-26 — the eight business-model classes, as synthetic issuers.

One company per class, each built so that the class's defining
characteristic is the LOUDEST thing about it and everything else is
deliberately unremarkable. That is what makes a canary a canary: when it
dies, you know which gas.

Every number is invented. Where a canary echoes a live failure it echoes the
SHAPE — an insurer whose operating cash flow is policyholder float, a
broker-dealer holding customer money, an issuer filing a 20-F — never an
issuer's figures.

Scale is deliberately small and round. A canary is read by a person deciding
whether an assertion is right, and $4,000M of revenue against $600M of
operating income is checkable in the head; a real issuer's
$4,127,000,000.00 is not.
"""

from tests.fixtures.canary_company import CanaryCompany, Period

_MILLION = 1_000_000.0


def _quarters(base_end, revenue, op_margin, net_margin, ocf, capex, da, shares,
              fiscal_year, form="10-Q", starts=None):
    """Four consecutive quarters ending at the given dates.

    Each quarter is a full, separately reported period, because that is what
    a TTM is constructed from: `finance/ttm.py` builds four discrete quarters
    or a fiscal-year-plus-year-to-date bridge, and a fixture that supplies
    only annuals silently exercises neither.
    """
    out = []
    for index, (start, end, fp) in enumerate(base_end):
        out.append(Period(
            end=end, start=start, fiscal_year=fiscal_year[index], fiscal_period=fp,
            form=form,
            values={
                "revenue": revenue[index],
                "operating_income": round(revenue[index] * op_margin, 2),
                "net_income": round(revenue[index] * net_margin, 2),
                "operating_cash_flow": ocf[index],
                "capital_expenditure": capex[index],
                "depreciation_and_amortization": da,
                "diluted_shares": shares,
            }))
    return out


# ---------------------------------------------------------------------------
# 1. mature_profitable — the case everything else is measured against
# ---------------------------------------------------------------------------
#
# Slow steady growth, a healthy margin, real free cash flow, modest debt, one
# reconcilable share count. Nothing here should trip any gate, which is
# exactly what makes it useful: a canary that fails HERE means a gate is
# firing on the ordinary case.

MATURE_PROFITABLE = CanaryCompany(
    key="mature_profitable",
    symbol="CNRY1", cik=1000001, name="Canary Industrial Products Inc",
    sic="3564", sic_description="Industrial and Commercial Machinery",
    price=80.0, shares_outstanding=200 * _MILLION,
    notes="Steady industrial manufacturer; the reference case for every gate.",
    annual=[
        Period(end="2025-12-31", fiscal_year=2025, fiscal_period="FY", form="10-K",
               values={"revenue": 4000 * _MILLION, "operating_income": 600 * _MILLION,
                       "net_income": 440 * _MILLION,
                       "income_before_tax": 580 * _MILLION,
                       "income_tax_expense": 140 * _MILLION,
                       "operating_cash_flow": 700 * _MILLION,
                       "capital_expenditure": 160 * _MILLION,
                       "depreciation_and_amortization": 150 * _MILLION,
                       "diluted_shares": 200 * _MILLION,
                       "cash_and_cash_equivalents": 300 * _MILLION,
                       "short_term_investments": 100 * _MILLION,
                       "assets": 5000 * _MILLION, "current_assets": 1800 * _MILLION,
                       "liabilities": 2600 * _MILLION,
                       "current_liabilities": 900 * _MILLION,
                       "stockholders_equity": 2400 * _MILLION,
                       "short_term_debt": 50 * _MILLION,
                       "long_term_debt": 800 * _MILLION}),
        Period(end="2024-12-31", fiscal_year=2024, fiscal_period="FY", form="10-K",
               values={"revenue": 3846 * _MILLION, "operating_income": 570 * _MILLION,
                       "net_income": 415 * _MILLION,
                       "income_before_tax": 550 * _MILLION,
                       "income_tax_expense": 135 * _MILLION,
                       "operating_cash_flow": 660 * _MILLION,
                       "capital_expenditure": 155 * _MILLION,
                       "depreciation_and_amortization": 145 * _MILLION,
                       "diluted_shares": 202 * _MILLION,
                       "cash_and_cash_equivalents": 280 * _MILLION,
                       "short_term_investments": 90 * _MILLION,
                       "assets": 4800 * _MILLION, "current_assets": 1700 * _MILLION,
                       "liabilities": 2550 * _MILLION,
                       "current_liabilities": 880 * _MILLION,
                       "stockholders_equity": 2250 * _MILLION,
                       "short_term_debt": 60 * _MILLION,
                       "long_term_debt": 820 * _MILLION}),
        Period(end="2023-12-31", fiscal_year=2023, fiscal_period="FY", form="10-K",
               values={"revenue": 3700 * _MILLION, "operating_income": 540 * _MILLION,
                       "net_income": 395 * _MILLION,
                       "income_before_tax": 520 * _MILLION,
                       "income_tax_expense": 125 * _MILLION,
                       "operating_cash_flow": 630 * _MILLION,
                       "capital_expenditure": 150 * _MILLION,
                       "depreciation_and_amortization": 140 * _MILLION,
                       "diluted_shares": 205 * _MILLION,
                       "cash_and_cash_equivalents": 260 * _MILLION,
                       "assets": 4600 * _MILLION, "current_assets": 1600 * _MILLION,
                       "liabilities": 2500 * _MILLION,
                       "current_liabilities": 860 * _MILLION,
                       "stockholders_equity": 2100 * _MILLION,
                       "short_term_debt": 55 * _MILLION,
                       "long_term_debt": 840 * _MILLION}),
    ],
    quarterly=_quarters(
        [("2026-04-01", "2026-06-30", "Q2"), ("2026-01-01", "2026-03-31", "Q1"),
         ("2025-10-01", "2025-12-31", "Q4"), ("2025-07-01", "2025-09-30", "Q3")],
        revenue=[1080 * _MILLION, 1040 * _MILLION, 1020 * _MILLION, 1010 * _MILLION],
        op_margin=0.152, net_margin=0.111,
        ocf=[190 * _MILLION, 175 * _MILLION, 180 * _MILLION, 172 * _MILLION],
        capex=[42 * _MILLION, 40 * _MILLION, 41 * _MILLION, 39 * _MILLION],
        da=38 * _MILLION, shares=199 * _MILLION,
        fiscal_year=[2026, 2026, 2025, 2025]),
)
# The current balance sheet, on the newest quarter. Kept beside the flows so
# a reader can see that the equity bridge has a NEWER instant than the
# annual statement -- which is the case the AOS live failure was about.
MATURE_PROFITABLE.quarterly[0].values.update({
    "cash_and_cash_equivalents": 340 * _MILLION,
    "short_term_investments": 110 * _MILLION,
    "assets": 5200 * _MILLION, "current_assets": 1900 * _MILLION,
    "liabilities": 2650 * _MILLION, "current_liabilities": 950 * _MILLION,
    "stockholders_equity": 2550 * _MILLION,
    "short_term_debt": 45 * _MILLION, "long_term_debt": 780 * _MILLION,
})


# ---------------------------------------------------------------------------
# 2. high_growth_profitable — where a model BOUND meets a real growth rate
# ---------------------------------------------------------------------------
#
# 58% current growth against a configured model bound well below it, plus
# consolidated revenue-growth guidance for the current fiscal year. The point
# is Part 20's: the bound must remain the MODEL'S LIMIT and must never be
# rendered as this company's expected growth.

_HIGH_GROWTH_GUIDANCE = {
    "accession": "9999999999-26-000900",
    "filed": "2026-07-28",
    "document": "canary-q2-2026-release.htm",
    "text": (
        "<html><body>"
        "<p>Canary Cloud Platforms, Inc. Reports Second Quarter Fiscal 2026 Results</p>"
        "<p>For the full year 2026, the company expects total revenue growth of "
        "40% to 44%.</p>"
        "<p>For the third quarter of fiscal 2026, the company expects revenue of "
        "$710 million to $730 million.</p>"
        "</body></html>"),
}

HIGH_GROWTH_PROFITABLE = CanaryCompany(
    key="high_growth_profitable",
    symbol="CNRY2", cik=1000002, name="Canary Cloud Platforms Inc",
    sic="7372", sic_description="Services-Prepackaged Software",
    price=210.0, shares_outstanding=300 * _MILLION,
    guidance_release=_HIGH_GROWTH_GUIDANCE,
    notes="Fast-growing profitable software issuer with current-year revenue guidance.",
    annual=[
        Period(end="2025-12-31", fiscal_year=2025, fiscal_period="FY", form="10-K",
               values={"revenue": 1800 * _MILLION, "operating_income": 306 * _MILLION,
                       "net_income": 240 * _MILLION,
                       "income_before_tax": 300 * _MILLION,
                       "income_tax_expense": 60 * _MILLION,
                       "operating_cash_flow": 470 * _MILLION,
                       "capital_expenditure": 90 * _MILLION,
                       "depreciation_and_amortization": 70 * _MILLION,
                       "diluted_shares": 300 * _MILLION,
                       "cash_and_cash_equivalents": 900 * _MILLION,
                       "short_term_investments": 600 * _MILLION,
                       "assets": 3200 * _MILLION, "current_assets": 2100 * _MILLION,
                       "liabilities": 900 * _MILLION,
                       "current_liabilities": 700 * _MILLION,
                       "stockholders_equity": 2300 * _MILLION,
                       "long_term_debt": 150 * _MILLION}),
        Period(end="2024-12-31", fiscal_year=2024, fiscal_period="FY", form="10-K",
               values={"revenue": 1140 * _MILLION, "operating_income": 171 * _MILLION,
                       "net_income": 130 * _MILLION,
                       "income_before_tax": 168 * _MILLION,
                       "income_tax_expense": 38 * _MILLION,
                       "operating_cash_flow": 280 * _MILLION,
                       "capital_expenditure": 60 * _MILLION,
                       "depreciation_and_amortization": 50 * _MILLION,
                       "diluted_shares": 295 * _MILLION,
                       "cash_and_cash_equivalents": 700 * _MILLION,
                       "short_term_investments": 400 * _MILLION,
                       "assets": 2400 * _MILLION, "current_assets": 1500 * _MILLION,
                       "liabilities": 700 * _MILLION,
                       "current_liabilities": 560 * _MILLION,
                       "stockholders_equity": 1700 * _MILLION,
                       "long_term_debt": 140 * _MILLION}),
        Period(end="2023-12-31", fiscal_year=2023, fiscal_period="FY", form="10-K",
               values={"revenue": 730 * _MILLION, "operating_income": 95 * _MILLION,
                       "net_income": 70 * _MILLION,
                       "income_before_tax": 92 * _MILLION,
                       "income_tax_expense": 22 * _MILLION,
                       "operating_cash_flow": 160 * _MILLION,
                       "capital_expenditure": 40 * _MILLION,
                       "depreciation_and_amortization": 35 * _MILLION,
                       "diluted_shares": 290 * _MILLION,
                       "cash_and_cash_equivalents": 500 * _MILLION,
                       "assets": 1700 * _MILLION, "current_assets": 1100 * _MILLION,
                       "liabilities": 520 * _MILLION,
                       "current_liabilities": 420 * _MILLION,
                       "stockholders_equity": 1180 * _MILLION,
                       "long_term_debt": 130 * _MILLION}),
    ],
    quarterly=_quarters(
        [("2026-04-01", "2026-06-30", "Q2"), ("2026-01-01", "2026-03-31", "Q1"),
         ("2025-10-01", "2025-12-31", "Q4"), ("2025-07-01", "2025-09-30", "Q3")],
        revenue=[690 * _MILLION, 610 * _MILLION, 540 * _MILLION, 480 * _MILLION],
        op_margin=0.180, net_margin=0.140,
        ocf=[180 * _MILLION, 160 * _MILLION, 140 * _MILLION, 125 * _MILLION],
        capex=[34 * _MILLION, 30 * _MILLION, 26 * _MILLION, 24 * _MILLION],
        da=22 * _MILLION, shares=300 * _MILLION,
        fiscal_year=[2026, 2026, 2025, 2025]),
)
HIGH_GROWTH_PROFITABLE.quarterly[0].values.update({
    "cash_and_cash_equivalents": 1100 * _MILLION,
    "short_term_investments": 700 * _MILLION,
    "assets": 3800 * _MILLION, "current_assets": 2500 * _MILLION,
    "liabilities": 1000 * _MILLION, "current_liabilities": 800 * _MILLION,
    "stockholders_equity": 2800 * _MILLION,
    "long_term_debt": 150 * _MILLION,
})


# ---------------------------------------------------------------------------
# 3. loss_making_growth — the model working is not the model failing
# ---------------------------------------------------------------------------
#
# Real revenue, real growth, and a persistently negative operating margin and
# free cash flow. Spec §14's forbidden rendering is the target: a DCF that
# correctly declines to grow a negative terminal cash flow into a perpetuity
# must not be reported as an invalid model, and its numbers must not become
# research evidence either way.

LOSS_MAKING_GROWTH = CanaryCompany(
    key="loss_making_growth",
    symbol="CNRY3", cik=1000003, name="Canary Mobility Corp",
    sic="3711", sic_description="Motor Vehicles and Passenger Car Bodies",
    price=14.0, shares_outstanding=900 * _MILLION,
    notes="Growing, cash-burning manufacturer; negative margins in every reported year.",
    annual=[
        Period(end="2025-12-31", fiscal_year=2025, fiscal_period="FY", form="10-K",
               values={"revenue": 2400 * _MILLION,
                       "operating_income": -520 * _MILLION,
                       "net_income": -610 * _MILLION,
                       "income_before_tax": -600 * _MILLION,
                       "income_tax_expense": 10 * _MILLION,
                       "operating_cash_flow": -380 * _MILLION,
                       "capital_expenditure": 420 * _MILLION,
                       "depreciation_and_amortization": 210 * _MILLION,
                       "diluted_shares": 890 * _MILLION,
                       "cash_and_cash_equivalents": 1500 * _MILLION,
                       "short_term_investments": 400 * _MILLION,
                       "assets": 6200 * _MILLION, "current_assets": 2800 * _MILLION,
                       "liabilities": 3400 * _MILLION,
                       "current_liabilities": 1400 * _MILLION,
                       "stockholders_equity": 2800 * _MILLION,
                       "short_term_debt": 120 * _MILLION,
                       "long_term_debt": 1500 * _MILLION}),
        Period(end="2024-12-31", fiscal_year=2024, fiscal_period="FY", form="10-K",
               values={"revenue": 1700 * _MILLION,
                       "operating_income": -480 * _MILLION,
                       "net_income": -540 * _MILLION,
                       "income_before_tax": -535 * _MILLION,
                       "income_tax_expense": 5 * _MILLION,
                       "operating_cash_flow": -340 * _MILLION,
                       "capital_expenditure": 360 * _MILLION,
                       "depreciation_and_amortization": 170 * _MILLION,
                       "diluted_shares": 860 * _MILLION,
                       "cash_and_cash_equivalents": 1800 * _MILLION,
                       "assets": 5600 * _MILLION, "current_assets": 2900 * _MILLION,
                       "liabilities": 3000 * _MILLION,
                       "current_liabilities": 1250 * _MILLION,
                       "stockholders_equity": 2600 * _MILLION,
                       "short_term_debt": 110 * _MILLION,
                       "long_term_debt": 1400 * _MILLION}),
        Period(end="2023-12-31", fiscal_year=2023, fiscal_period="FY", form="10-K",
               values={"revenue": 1100 * _MILLION,
                       "operating_income": -400 * _MILLION,
                       "net_income": -450 * _MILLION,
                       "income_before_tax": -448 * _MILLION,
                       "income_tax_expense": 2 * _MILLION,
                       "operating_cash_flow": -300 * _MILLION,
                       "capital_expenditure": 300 * _MILLION,
                       "depreciation_and_amortization": 130 * _MILLION,
                       "diluted_shares": 820 * _MILLION,
                       "cash_and_cash_equivalents": 2100 * _MILLION,
                       "assets": 5100 * _MILLION, "current_assets": 3000 * _MILLION,
                       "liabilities": 2700 * _MILLION,
                       "current_liabilities": 1100 * _MILLION,
                       "stockholders_equity": 2400 * _MILLION,
                       "long_term_debt": 1300 * _MILLION}),
    ],
    quarterly=_quarters(
        [("2026-04-01", "2026-06-30", "Q2"), ("2026-01-01", "2026-03-31", "Q1"),
         ("2025-10-01", "2025-12-31", "Q4"), ("2025-07-01", "2025-09-30", "Q3")],
        revenue=[760 * _MILLION, 700 * _MILLION, 660 * _MILLION, 620 * _MILLION],
        op_margin=-0.205, net_margin=-0.245,
        ocf=[-100 * _MILLION, -95 * _MILLION, -92 * _MILLION, -90 * _MILLION],
        capex=[115 * _MILLION, 108 * _MILLION, 105 * _MILLION, 100 * _MILLION],
        da=58 * _MILLION, shares=900 * _MILLION,
        fiscal_year=[2026, 2026, 2025, 2025]),
)
LOSS_MAKING_GROWTH.quarterly[0].values.update({
    "cash_and_cash_equivalents": 1200 * _MILLION,
    "short_term_investments": 300 * _MILLION,
    "assets": 6400 * _MILLION, "current_assets": 2600 * _MILLION,
    "liabilities": 3600 * _MILLION, "current_liabilities": 1500 * _MILLION,
    "stockholders_equity": 2800 * _MILLION,
    "short_term_debt": 130 * _MILLION, "long_term_debt": 1600 * _MILLION,
})


# ---------------------------------------------------------------------------
# 4. insurer — OCF minus CapEx is not owner cash
# ---------------------------------------------------------------------------
#
# Large positive operating cash flow against modest operating income, because
# premiums arrive before claims are paid. Spec §12's live example: "strong
# free cash flow demonstrates cash generation" for an issuer whose cash is
# policyholder float. The current ratio is likewise low-information here.

INSURER = CanaryCompany(
    key="insurer",
    symbol="CNRY4", cik=1000004, name="Canary Mutual Casualty Group Inc",
    sic="6311", sic_description="Life Insurance",
    price=55.0, shares_outstanding=400 * _MILLION,
    corroborating_concepts=(
        "PolicyholderBenefitsAndClaimsIncurredNet",
        "LiabilityForFuturePolicyBenefits",
        "PremiumsEarnedNet",
        "DeferredPolicyAcquisitionCosts",
    ),
    notes="Insurance carrier; operating cash flow is policyholder float, not owner cash.",
    annual=[
        Period(end="2025-12-31", fiscal_year=2025, fiscal_period="FY", form="10-K",
               values={"revenue": 9000 * _MILLION, "operating_income": 720 * _MILLION,
                       "net_income": 560 * _MILLION,
                       "income_before_tax": 700 * _MILLION,
                       "income_tax_expense": 140 * _MILLION,
                       # Nine times operating income: this is the number the
                       # architecture must refuse to call owner free cash flow.
                       "operating_cash_flow": 6400 * _MILLION,
                       "capital_expenditure": 90 * _MILLION,
                       "depreciation_and_amortization": 110 * _MILLION,
                       "diluted_shares": 400 * _MILLION,
                       "cash_and_cash_equivalents": 2200 * _MILLION,
                       "short_term_investments": 8000 * _MILLION,
                       "assets": 62000 * _MILLION, "current_assets": 9000 * _MILLION,
                       "liabilities": 54000 * _MILLION,
                       "current_liabilities": 11000 * _MILLION,
                       "stockholders_equity": 8000 * _MILLION,
                       "long_term_debt": 2400 * _MILLION}),
        Period(end="2024-12-31", fiscal_year=2024, fiscal_period="FY", form="10-K",
               values={"revenue": 8600 * _MILLION, "operating_income": 690 * _MILLION,
                       "net_income": 530 * _MILLION,
                       "income_before_tax": 670 * _MILLION,
                       "income_tax_expense": 140 * _MILLION,
                       "operating_cash_flow": 6100 * _MILLION,
                       "capital_expenditure": 85 * _MILLION,
                       "depreciation_and_amortization": 105 * _MILLION,
                       "diluted_shares": 405 * _MILLION,
                       "cash_and_cash_equivalents": 2000 * _MILLION,
                       "short_term_investments": 7600 * _MILLION,
                       "assets": 59000 * _MILLION, "current_assets": 8600 * _MILLION,
                       "liabilities": 51500 * _MILLION,
                       "current_liabilities": 10600 * _MILLION,
                       "stockholders_equity": 7500 * _MILLION,
                       "long_term_debt": 2300 * _MILLION}),
        Period(end="2023-12-31", fiscal_year=2023, fiscal_period="FY", form="10-K",
               values={"revenue": 8200 * _MILLION, "operating_income": 650 * _MILLION,
                       "net_income": 500 * _MILLION,
                       "income_before_tax": 640 * _MILLION,
                       "income_tax_expense": 140 * _MILLION,
                       "operating_cash_flow": 5800 * _MILLION,
                       "capital_expenditure": 80 * _MILLION,
                       "depreciation_and_amortization": 100 * _MILLION,
                       "diluted_shares": 410 * _MILLION,
                       "cash_and_cash_equivalents": 1900 * _MILLION,
                       "assets": 56000 * _MILLION, "current_assets": 8200 * _MILLION,
                       "liabilities": 49000 * _MILLION,
                       "current_liabilities": 10200 * _MILLION,
                       "stockholders_equity": 7000 * _MILLION,
                       "long_term_debt": 2200 * _MILLION}),
    ],
    quarterly=_quarters(
        [("2026-04-01", "2026-06-30", "Q2"), ("2026-01-01", "2026-03-31", "Q1"),
         ("2025-10-01", "2025-12-31", "Q4"), ("2025-07-01", "2025-09-30", "Q3")],
        revenue=[2350 * _MILLION, 2300 * _MILLION, 2280 * _MILLION, 2250 * _MILLION],
        op_margin=0.081, net_margin=0.063,
        ocf=[1650 * _MILLION, 1620 * _MILLION, 1600 * _MILLION, 1580 * _MILLION],
        capex=[24 * _MILLION, 23 * _MILLION, 22 * _MILLION, 22 * _MILLION],
        da=28 * _MILLION, shares=400 * _MILLION,
        fiscal_year=[2026, 2026, 2025, 2025]),
)
INSURER.quarterly[0].values.update({
    "cash_and_cash_equivalents": 2400 * _MILLION,
    "short_term_investments": 8400 * _MILLION,
    "assets": 64000 * _MILLION, "current_assets": 9400 * _MILLION,
    "liabilities": 55600 * _MILLION,
    # Below 1.0. For an insurer this is a structural feature of the balance
    # sheet, not a liquidity warning -- spec §12's second forbidden claim.
    "current_liabilities": 11500 * _MILLION,
    "stockholders_equity": 8400 * _MILLION,
    "long_term_debt": 2400 * _MILLION,
})


# ---------------------------------------------------------------------------
# 5. broker_dealer — the cash belongs to the customers
# ---------------------------------------------------------------------------
#
# The live case that produced `finance/business_model.py`: $566M of operating
# cash flow against $51M of operating income and $25M of net income, and a
# vendor sector field reading "Technology / Software". Here the SIC code says
# 6211 and the filings carry customer payables and segregated cash, so
# classification has both signals and needs neither a name nor a sector.

BROKER_DEALER = CanaryCompany(
    key="broker_dealer",
    symbol="CNRY5", cik=1000005, name="Canary Securities Holdings Inc",
    sic="6211", sic_description="Security Brokers, Dealers and Flotation Companies",
    price=42.0, shares_outstanding=800 * _MILLION,
    corroborating_concepts=(
        "PayablesToCustomers",
        "ReceivablesFromCustomers",
        "CashAndSecuritiesSegregatedUnderFederalAndOtherRegulations",
        "SecuritiesBorrowed",
    ),
    notes="Retail broker-dealer; most operating cash flow is customer money in transit.",
    annual=[
        Period(end="2025-12-31", fiscal_year=2025, fiscal_period="FY", form="10-K",
               values={"revenue": 1900 * _MILLION, "operating_income": 210 * _MILLION,
                       "net_income": 150 * _MILLION,
                       "income_before_tax": 200 * _MILLION,
                       "income_tax_expense": 50 * _MILLION,
                       "operating_cash_flow": 4200 * _MILLION,
                       "capital_expenditure": 60 * _MILLION,
                       "depreciation_and_amortization": 55 * _MILLION,
                       "diluted_shares": 800 * _MILLION,
                       "cash_and_cash_equivalents": 3000 * _MILLION,
                       "restricted_cash": 5200 * _MILLION,
                       "short_term_investments": 1200 * _MILLION,
                       "assets": 27000 * _MILLION, "current_assets": 24000 * _MILLION,
                       "liabilities": 23000 * _MILLION,
                       "current_liabilities": 22000 * _MILLION,
                       "stockholders_equity": 4000 * _MILLION,
                       "long_term_debt": 700 * _MILLION}),
        Period(end="2024-12-31", fiscal_year=2024, fiscal_period="FY", form="10-K",
               values={"revenue": 1650 * _MILLION, "operating_income": 175 * _MILLION,
                       "net_income": 125 * _MILLION,
                       "income_before_tax": 168 * _MILLION,
                       "income_tax_expense": 43 * _MILLION,
                       "operating_cash_flow": 3600 * _MILLION,
                       "capital_expenditure": 52 * _MILLION,
                       "depreciation_and_amortization": 48 * _MILLION,
                       "diluted_shares": 795 * _MILLION,
                       "cash_and_cash_equivalents": 2700 * _MILLION,
                       "restricted_cash": 4800 * _MILLION,
                       "assets": 24000 * _MILLION, "current_assets": 21500 * _MILLION,
                       "liabilities": 20500 * _MILLION,
                       "current_liabilities": 19700 * _MILLION,
                       "stockholders_equity": 3500 * _MILLION,
                       "long_term_debt": 650 * _MILLION}),
        Period(end="2023-12-31", fiscal_year=2023, fiscal_period="FY", form="10-K",
               values={"revenue": 1400 * _MILLION, "operating_income": 140 * _MILLION,
                       "net_income": 100 * _MILLION,
                       "income_before_tax": 135 * _MILLION,
                       "income_tax_expense": 35 * _MILLION,
                       "operating_cash_flow": 3100 * _MILLION,
                       "capital_expenditure": 45 * _MILLION,
                       "depreciation_and_amortization": 42 * _MILLION,
                       "diluted_shares": 790 * _MILLION,
                       "cash_and_cash_equivalents": 2400 * _MILLION,
                       "assets": 21000 * _MILLION, "current_assets": 19000 * _MILLION,
                       "liabilities": 18000 * _MILLION,
                       "current_liabilities": 17400 * _MILLION,
                       "stockholders_equity": 3000 * _MILLION,
                       "long_term_debt": 600 * _MILLION}),
    ],
    quarterly=_quarters(
        [("2026-04-01", "2026-06-30", "Q2"), ("2026-01-01", "2026-03-31", "Q1"),
         ("2025-10-01", "2025-12-31", "Q4"), ("2025-07-01", "2025-09-30", "Q3")],
        revenue=[540 * _MILLION, 510 * _MILLION, 490 * _MILLION, 470 * _MILLION],
        op_margin=0.115, net_margin=0.082,
        ocf=[1150 * _MILLION, 1080 * _MILLION, 1040 * _MILLION, 1000 * _MILLION],
        capex=[16 * _MILLION, 15 * _MILLION, 15 * _MILLION, 14 * _MILLION],
        da=14 * _MILLION, shares=800 * _MILLION,
        fiscal_year=[2026, 2026, 2025, 2025]),
)
BROKER_DEALER.quarterly[0].values.update({
    "cash_and_cash_equivalents": 3200 * _MILLION,
    "restricted_cash": 5600 * _MILLION,
    "short_term_investments": 1300 * _MILLION,
    "assets": 29000 * _MILLION, "current_assets": 26000 * _MILLION,
    "liabilities": 24600 * _MILLION, "current_liabilities": 23600 * _MILLION,
    "stockholders_equity": 4400 * _MILLION,
    "long_term_debt": 720 * _MILLION,
})


# ---------------------------------------------------------------------------
# 6. foreign_private_issuer — a 20-F is an annual report
# ---------------------------------------------------------------------------
#
# The failure this class exists for: `xbrl_mapping` filtered annual facts to
# 10-K forms only, so a foreign private issuer -- which files a 20-F and
# never a 10-K -- lost every annual fact, and the analysis reported no
# revenue for a company that had filed complete accounts. The interim 6-K is
# here for the same reason on the other side.

FOREIGN_PRIVATE_ISSUER = CanaryCompany(
    key="foreign_private_issuer",
    symbol="CNRY6", cik=1000006, name="Canary International Holdings plc",
    sic="2836", sic_description="Biological Products",
    price=64.0, shares_outstanding=500 * _MILLION,
    notes="Foreign private issuer: annual 20-F, interim 6-K, US-dollar reporting.",
    annual=[
        Period(end="2025-12-31", fiscal_year=2025, fiscal_period="FY", form="20-F",
               values={"revenue": 5200 * _MILLION, "operating_income": 910 * _MILLION,
                       "net_income": 700 * _MILLION,
                       "income_before_tax": 890 * _MILLION,
                       "income_tax_expense": 190 * _MILLION,
                       "operating_cash_flow": 1150 * _MILLION,
                       "capital_expenditure": 260 * _MILLION,
                       "depreciation_and_amortization": 240 * _MILLION,
                       "diluted_shares": 500 * _MILLION,
                       "cash_and_cash_equivalents": 1400 * _MILLION,
                       "short_term_investments": 300 * _MILLION,
                       "assets": 11000 * _MILLION, "current_assets": 4200 * _MILLION,
                       "liabilities": 5600 * _MILLION,
                       "current_liabilities": 2100 * _MILLION,
                       "stockholders_equity": 5400 * _MILLION,
                       "short_term_debt": 200 * _MILLION,
                       "long_term_debt": 2100 * _MILLION}),
        Period(end="2024-12-31", fiscal_year=2024, fiscal_period="FY", form="20-F",
               values={"revenue": 4900 * _MILLION, "operating_income": 850 * _MILLION,
                       "net_income": 650 * _MILLION,
                       "income_before_tax": 830 * _MILLION,
                       "income_tax_expense": 180 * _MILLION,
                       "operating_cash_flow": 1080 * _MILLION,
                       "capital_expenditure": 240 * _MILLION,
                       "depreciation_and_amortization": 225 * _MILLION,
                       "diluted_shares": 505 * _MILLION,
                       "cash_and_cash_equivalents": 1250 * _MILLION,
                       "assets": 10400 * _MILLION, "current_assets": 3900 * _MILLION,
                       "liabilities": 5400 * _MILLION,
                       "current_liabilities": 2000 * _MILLION,
                       "stockholders_equity": 5000 * _MILLION,
                       "short_term_debt": 190 * _MILLION,
                       "long_term_debt": 2050 * _MILLION}),
        Period(end="2023-12-31", fiscal_year=2023, fiscal_period="FY", form="20-F",
               values={"revenue": 4600 * _MILLION, "operating_income": 780 * _MILLION,
                       "net_income": 600 * _MILLION,
                       "income_before_tax": 765 * _MILLION,
                       "income_tax_expense": 165 * _MILLION,
                       "operating_cash_flow": 1000 * _MILLION,
                       "capital_expenditure": 220 * _MILLION,
                       "depreciation_and_amortization": 210 * _MILLION,
                       "diluted_shares": 510 * _MILLION,
                       "cash_and_cash_equivalents": 1100 * _MILLION,
                       "assets": 9800 * _MILLION, "current_assets": 3700 * _MILLION,
                       "liabilities": 5200 * _MILLION,
                       "current_liabilities": 1950 * _MILLION,
                       "stockholders_equity": 4600 * _MILLION,
                       "long_term_debt": 2000 * _MILLION}),
    ],
    # Interim reporting on 6-K. Only two interim periods, because a foreign
    # private issuer commonly reports half-yearly rather than quarterly --
    # which is itself part of what this canary tests: the latest-period
    # selection must cope with an interim cadence that is not four-a-year.
    quarterly=_quarters(
        [("2026-04-01", "2026-06-30", "Q2"), ("2026-01-01", "2026-03-31", "Q1")],
        revenue=[1380 * _MILLION, 1330 * _MILLION],
        op_margin=0.178, net_margin=0.136,
        ocf=[300 * _MILLION, 290 * _MILLION],
        capex=[68 * _MILLION, 65 * _MILLION],
        da=62 * _MILLION, shares=498 * _MILLION,
        fiscal_year=[2026, 2026], form="6-K"),
)
FOREIGN_PRIVATE_ISSUER.quarterly[0].values.update({
    "cash_and_cash_equivalents": 1500 * _MILLION,
    "short_term_investments": 350 * _MILLION,
    "assets": 11400 * _MILLION, "current_assets": 4400 * _MILLION,
    "liabilities": 5700 * _MILLION, "current_liabilities": 2150 * _MILLION,
    "stockholders_equity": 5700 * _MILLION,
    "short_term_debt": 210 * _MILLION, "long_term_debt": 2050 * _MILLION,
})


# ---------------------------------------------------------------------------
# 7. unusual_item_company — three different profitabilities, all true
# ---------------------------------------------------------------------------
#
# A large impairment and a restructuring charge in the latest year, both
# TAGGED WITH AMOUNTS (spec §12's rule: an unusual item without a figure is a
# disclosure, not an adjustment). Reported GAAP margin, a company-adjusted
# margin and a system-normalized margin are three different numbers here, and
# which one a forecast uses changes the valuation.

UNUSUAL_ITEM_COMPANY = CanaryCompany(
    key="unusual_item_company",
    symbol="CNRY7", cik=1000007, name="Canary Therapeutics Inc",
    sic="2834", sic_description="Pharmaceutical Preparations",
    price=96.0, shares_outstanding=250 * _MILLION,
    notes="A large impairment and restructuring charge distort the latest reported year.",
    annual=[
        Period(end="2025-12-31", fiscal_year=2025, fiscal_period="FY", form="10-K",
               values={"revenue": 7000 * _MILLION,
                       # 2.1% reported, against roughly 20% before the
                       # one-off charges below.
                       "operating_income": 150 * _MILLION,
                       "net_income": 60 * _MILLION,
                       "income_before_tax": 120 * _MILLION,
                       "income_tax_expense": 60 * _MILLION,
                       "impairment_charges": 900 * _MILLION,
                       "restructuring_charges": 350 * _MILLION,
                       "operating_cash_flow": 1500 * _MILLION,
                       "capital_expenditure": 380 * _MILLION,
                       "depreciation_and_amortization": 620 * _MILLION,
                       "diluted_shares": 250 * _MILLION,
                       "cash_and_cash_equivalents": 2100 * _MILLION,
                       "short_term_investments": 900 * _MILLION,
                       "assets": 19000 * _MILLION, "current_assets": 6400 * _MILLION,
                       "liabilities": 10500 * _MILLION,
                       "current_liabilities": 3600 * _MILLION,
                       "stockholders_equity": 8500 * _MILLION,
                       "short_term_debt": 400 * _MILLION,
                       "long_term_debt": 4600 * _MILLION}),
        Period(end="2024-12-31", fiscal_year=2024, fiscal_period="FY", form="10-K",
               values={"revenue": 6800 * _MILLION, "operating_income": 1360 * _MILLION,
                       "net_income": 1000 * _MILLION,
                       "income_before_tax": 1320 * _MILLION,
                       "income_tax_expense": 320 * _MILLION,
                       "operating_cash_flow": 1620 * _MILLION,
                       "capital_expenditure": 360 * _MILLION,
                       "depreciation_and_amortization": 600 * _MILLION,
                       "diluted_shares": 254 * _MILLION,
                       "cash_and_cash_equivalents": 1900 * _MILLION,
                       "assets": 19500 * _MILLION, "current_assets": 6200 * _MILLION,
                       "liabilities": 10300 * _MILLION,
                       "current_liabilities": 3500 * _MILLION,
                       "stockholders_equity": 9200 * _MILLION,
                       "short_term_debt": 380 * _MILLION,
                       "long_term_debt": 4500 * _MILLION}),
        Period(end="2023-12-31", fiscal_year=2023, fiscal_period="FY", form="10-K",
               values={"revenue": 6500 * _MILLION, "operating_income": 1280 * _MILLION,
                       "net_income": 950 * _MILLION,
                       "income_before_tax": 1250 * _MILLION,
                       "income_tax_expense": 300 * _MILLION,
                       "operating_cash_flow": 1550 * _MILLION,
                       "capital_expenditure": 340 * _MILLION,
                       "depreciation_and_amortization": 580 * _MILLION,
                       "diluted_shares": 258 * _MILLION,
                       "cash_and_cash_equivalents": 1750 * _MILLION,
                       "assets": 19000 * _MILLION, "current_assets": 6000 * _MILLION,
                       "liabilities": 10100 * _MILLION,
                       "current_liabilities": 3400 * _MILLION,
                       "stockholders_equity": 8900 * _MILLION,
                       "long_term_debt": 4400 * _MILLION}),
    ],
    quarterly=_quarters(
        [("2026-04-01", "2026-06-30", "Q2"), ("2026-01-01", "2026-03-31", "Q1"),
         ("2025-10-01", "2025-12-31", "Q4"), ("2025-07-01", "2025-09-30", "Q3")],
        revenue=[1800 * _MILLION, 1760 * _MILLION, 1740 * _MILLION, 1750 * _MILLION],
        op_margin=0.195, net_margin=0.140,
        ocf=[400 * _MILLION, 390 * _MILLION, 385 * _MILLION, 380 * _MILLION],
        capex=[96 * _MILLION, 94 * _MILLION, 93 * _MILLION, 92 * _MILLION],
        da=155 * _MILLION, shares=250 * _MILLION,
        fiscal_year=[2026, 2026, 2025, 2025]),
)
# The large charge sits in ONE quarter, which is what makes a trailing-twelve-
# month margin and a full-year margin disagree. The other quarters carry the
# small recurring restructuring an ongoing programme actually produces --
# without them the trailing-twelve-month window cannot be CONSTRUCTED for
# these concepts at all (finance/ttm.py needs four discrete quarters), and an
# unusual item that cannot be measured over the profitability window is not
# available as an adjustment. That is also what lets `classify_recurrence`
# separate the one-off from the baseline instead of assuming.
UNUSUAL_ITEM_COMPANY.quarterly[2].values.update({
    "impairment_charges": 900 * _MILLION,
    "restructuring_charges": 350 * _MILLION,
    "operating_income": -1000 * _MILLION,
    "net_income": -800 * _MILLION,
})
for _index, (_impairment, _restructuring) in enumerate(
        [(6 * _MILLION, 14 * _MILLION), (5 * _MILLION, 12 * _MILLION),
         (None, None), (7 * _MILLION, 15 * _MILLION)]):
    if _impairment is None:
        continue
    UNUSUAL_ITEM_COMPANY.quarterly[_index].values.update({
        "impairment_charges": _impairment, "restructuring_charges": _restructuring})
# The same concepts on the annual series, so recurrence is judged against
# reported history rather than against a single window.
UNUSUAL_ITEM_COMPANY.annual[1].values.update({
    "impairment_charges": 40 * _MILLION, "restructuring_charges": 60 * _MILLION})
UNUSUAL_ITEM_COMPANY.annual[2].values.update({
    "impairment_charges": 35 * _MILLION, "restructuring_charges": 55 * _MILLION})
UNUSUAL_ITEM_COMPANY.quarterly[0].values.update({
    "cash_and_cash_equivalents": 2300 * _MILLION,
    "short_term_investments": 950 * _MILLION,
    "assets": 19400 * _MILLION, "current_assets": 6600 * _MILLION,
    "liabilities": 10600 * _MILLION, "current_liabilities": 3700 * _MILLION,
    "stockholders_equity": 8800 * _MILLION,
    "short_term_debt": 420 * _MILLION, "long_term_debt": 4550 * _MILLION,
})


# ---------------------------------------------------------------------------
# 8. complex_capital_structure — current debt is not total debt
# ---------------------------------------------------------------------------
#
# Short-term borrowings, the current portion of long-term debt, and long-term
# debt, all separately reported; preferred equity and a non-controlling
# interest in the bridge; and an issuer-reported combined total that
# DISAGREES with the component sum. Spec §10 says that resolves by documented
# precedence -- the issuer's own consolidated total wins -- and says so on
# the metric, rather than the sum silently standing.

COMPLEX_CAPITAL_STRUCTURE = CanaryCompany(
    key="complex_capital_structure",
    symbol="CNRY8", cik=1000008, name="Canary Infrastructure Partners Inc",
    sic="4911", sic_description="Electric Services",
    price=38.0, shares_outstanding=1200 * _MILLION,
    notes=("Multi-tranche debt, preferred equity and a non-controlling interest; the "
           "issuer's reported combined debt total disagrees with the component sum."),
    annual=[
        Period(end="2025-12-31", fiscal_year=2025, fiscal_period="FY", form="10-K",
               values={"revenue": 14000 * _MILLION, "operating_income": 2800 * _MILLION,
                       "net_income": 1400 * _MILLION,
                       "income_before_tax": 1800 * _MILLION,
                       "income_tax_expense": 400 * _MILLION,
                       "operating_cash_flow": 4200 * _MILLION,
                       "capital_expenditure": 3100 * _MILLION,
                       "depreciation_and_amortization": 1900 * _MILLION,
                       "diluted_shares": 1200 * _MILLION,
                       "cash_and_cash_equivalents": 800 * _MILLION,
                       "assets": 78000 * _MILLION, "current_assets": 5200 * _MILLION,
                       "liabilities": 54000 * _MILLION,
                       "current_liabilities": 7800 * _MILLION,
                       "stockholders_equity": 22000 * _MILLION,
                       "preferred_equity": 1500 * _MILLION,
                       "minority_interest": 2600 * _MILLION,
                       "short_term_debt": 1800 * _MILLION,
                       "current_portion_of_long_term_debt": 2200 * _MILLION,
                       "long_term_debt": 27000 * _MILLION,
                       # 31,000 by components; the issuer reports 33,400,
                       # a 7.2% difference. Spec §10: the issuer's own
                       # consolidated total outranks a sum assembled here.
                       "total_debt_combined": 33400 * _MILLION}),
        Period(end="2024-12-31", fiscal_year=2024, fiscal_period="FY", form="10-K",
               values={"revenue": 13400 * _MILLION, "operating_income": 2650 * _MILLION,
                       "net_income": 1320 * _MILLION,
                       "income_before_tax": 1700 * _MILLION,
                       "income_tax_expense": 380 * _MILLION,
                       "operating_cash_flow": 4000 * _MILLION,
                       "capital_expenditure": 2950 * _MILLION,
                       "depreciation_and_amortization": 1820 * _MILLION,
                       "diluted_shares": 1180 * _MILLION,
                       "cash_and_cash_equivalents": 750 * _MILLION,
                       "assets": 74000 * _MILLION, "current_assets": 5000 * _MILLION,
                       "liabilities": 51500 * _MILLION,
                       "current_liabilities": 7500 * _MILLION,
                       "stockholders_equity": 21000 * _MILLION,
                       "preferred_equity": 1500 * _MILLION,
                       "minority_interest": 2500 * _MILLION,
                       "short_term_debt": 1700 * _MILLION,
                       "current_portion_of_long_term_debt": 2100 * _MILLION,
                       "long_term_debt": 26000 * _MILLION}),
        Period(end="2023-12-31", fiscal_year=2023, fiscal_period="FY", form="10-K",
               values={"revenue": 12800 * _MILLION, "operating_income": 2500 * _MILLION,
                       "net_income": 1250 * _MILLION,
                       "income_before_tax": 1600 * _MILLION,
                       "income_tax_expense": 350 * _MILLION,
                       "operating_cash_flow": 3800 * _MILLION,
                       "capital_expenditure": 2800 * _MILLION,
                       "depreciation_and_amortization": 1740 * _MILLION,
                       "diluted_shares": 1160 * _MILLION,
                       "cash_and_cash_equivalents": 700 * _MILLION,
                       "assets": 70000 * _MILLION, "current_assets": 4800 * _MILLION,
                       "liabilities": 49000 * _MILLION,
                       "current_liabilities": 7200 * _MILLION,
                       "stockholders_equity": 20000 * _MILLION,
                       "preferred_equity": 1500 * _MILLION,
                       "minority_interest": 2400 * _MILLION,
                       "short_term_debt": 1600 * _MILLION,
                       "current_portion_of_long_term_debt": 2000 * _MILLION,
                       "long_term_debt": 25000 * _MILLION}),
    ],
    quarterly=_quarters(
        [("2026-04-01", "2026-06-30", "Q2"), ("2026-01-01", "2026-03-31", "Q1"),
         ("2025-10-01", "2025-12-31", "Q4"), ("2025-07-01", "2025-09-30", "Q3")],
        revenue=[3700 * _MILLION, 3600 * _MILLION, 3500 * _MILLION, 3450 * _MILLION],
        op_margin=0.200, net_margin=0.100,
        ocf=[1100 * _MILLION, 1060 * _MILLION, 1040 * _MILLION, 1020 * _MILLION],
        capex=[800 * _MILLION, 780 * _MILLION, 770 * _MILLION, 760 * _MILLION],
        da=480 * _MILLION, shares=1210 * _MILLION,
        fiscal_year=[2026, 2026, 2025, 2025]),
)
COMPLEX_CAPITAL_STRUCTURE.quarterly[0].values.update({
    "cash_and_cash_equivalents": 900 * _MILLION,
    "assets": 80000 * _MILLION, "current_assets": 5400 * _MILLION,
    "liabilities": 55000 * _MILLION, "current_liabilities": 8000 * _MILLION,
    "stockholders_equity": 22500 * _MILLION,
    "preferred_equity": 1500 * _MILLION,
    "minority_interest": 2700 * _MILLION,
    "short_term_debt": 1900 * _MILLION,
    "current_portion_of_long_term_debt": 2300 * _MILLION,
    "long_term_debt": 27500 * _MILLION,
    "total_debt_combined": 34100 * _MILLION,
})


# ---------------------------------------------------------------------------
# A ninth issuer, used by the negative runtime tests (Parts 7-8)
# ---------------------------------------------------------------------------
#
# Not a business-model class -- a DEFECT class. Identical in every respect to
# the complex-capital-structure canary except that its current balance sheet
# reports no debt component at all while its liabilities plainly carry debt.
# Total debt therefore cannot be established, and before this phase the
# equity bridge silently valued it as a debt-free company.

import copy  # noqa: E402  -- placed here so the deep copy reads beside its source

UNESTABLISHED_DEBT = copy.deepcopy(COMPLEX_CAPITAL_STRUCTURE)
UNESTABLISHED_DEBT.key = "unestablished_debt"
UNESTABLISHED_DEBT.symbol = "CNRY9"
UNESTABLISHED_DEBT.cik = 1000009
UNESTABLISHED_DEBT.name = "Canary Opaque Leverage Inc"
UNESTABLISHED_DEBT.notes = (
    "Reports liabilities that plainly include borrowings but tags no debt component on "
    "its current balance sheet, so total debt cannot be established.")
for _period in [UNESTABLISHED_DEBT.quarterly[0]]:
    for _field in ("short_term_debt", "current_portion_of_long_term_debt",
                   "long_term_debt", "total_debt_combined"):
        _period.values.pop(_field, None)


# ---------------------------------------------------------------------------
# A tenth issuer, for Part 8: guidance that is NOT revenue growth
# ---------------------------------------------------------------------------
#
# Also a defect class rather than a business-model class. Identical to the
# high-growth canary except that its earnings release guides FREE CASH FLOW
# growth and nothing else. Spec 11's hard invariant: a free-cash-flow growth
# figure may never populate a DCF revenue-growth assumption, however
# precisely it is stated and however obviously it is "growth".
#
# This is the AT&T failure in miniature. There, a growth figure from the
# wrong metric reached the revenue assumption, produced a nonsense rate, and
# the rate was CLAMPED into range -- which made the error look like a model
# bound doing its job. A clamp must never repair a semantic error.

_FCF_GROWTH_GUIDANCE = {
    "accession": "9999999999-26-000901",
    "filed": "2026-07-28",
    "document": "canary-fcf-release.htm",
    "text": (
        "<html><body>"
        "<p>Canary Cash Generation Corp Reports Second Quarter Fiscal 2026 Results</p>"
        "<p>For the full year 2026, the company expects free cash flow growth of "
        "40% to 44%.</p>"
        "</body></html>"),
}

FCF_GROWTH_GUIDANCE_ONLY = copy.deepcopy(HIGH_GROWTH_PROFITABLE)
FCF_GROWTH_GUIDANCE_ONLY.key = "fcf_growth_guidance_only"
FCF_GROWTH_GUIDANCE_ONLY.symbol = "CNRY10"
FCF_GROWTH_GUIDANCE_ONLY.cik = 1000010
FCF_GROWTH_GUIDANCE_ONLY.name = "Canary Cash Generation Corp"
FCF_GROWTH_GUIDANCE_ONLY.guidance_release = _FCF_GROWTH_GUIDANCE
FCF_GROWTH_GUIDANCE_ONLY.notes = (
    "Guides free-cash-flow growth and nothing else; no revenue guidance exists.")


ALL_CANARIES = (
    MATURE_PROFITABLE,
    HIGH_GROWTH_PROFITABLE,
    LOSS_MAKING_GROWTH,
    INSURER,
    BROKER_DEALER,
    FOREIGN_PRIVATE_ISSUER,
    UNUSUAL_ITEM_COMPANY,
    COMPLEX_CAPITAL_STRUCTURE,
)

DEFECT_CANARIES = (UNESTABLISHED_DEBT, FCF_GROWTH_GUIDANCE_ONLY)

BY_KEY = {company.key: company for company in ALL_CANARIES + DEFECT_CANARIES}
