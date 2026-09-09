"""
Microsoft DCF Valuation Engine
Author: Matthew Walker

Public portfolio version of the valuation engine supporting:
- 10-year explicit forecast
- Unlevered free cash flow
- Exact date-based mid-year discounting
- Gordon Growth terminal value
- WACC / terminal growth sensitivity
- Reverse DCF

Raw proprietary financial datasets are intentionally excluded.

All figures are US$ millions unless otherwise stated.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List


# ============================================================
# MODEL INPUTS
# ============================================================

@dataclass
class ModelInputs:
    valuation_date: date = date(2026, 9, 8)

    # FY2026 public financial anchor
    starting_revenue: float = 331_839.0

    # Valuation assumptions
    wacc: float = 0.0933888110663
    terminal_growth: float = 0.025

    # Equity bridge
    cash_and_investments: float = 76_843.0
    total_debt: float = 40_294.0
    diluted_shares_m: float = 7_453.0

    # Reference market price
    market_price: float = 493.95


INPUTS = ModelInputs()


# ============================================================
# BASE FORECAST ASSUMPTIONS
# ============================================================

YEARS = list(range(2027, 2037))

BASE_ASSUMPTIONS: Dict[str, List[float]] = {
    "revenue_growth": [
        0.155, 0.140, 0.125, 0.110, 0.095,
        0.081, 0.067, 0.053, 0.039, 0.025
    ],

    "ebitda_margin": [
        0.580, 0.587, 0.595, 0.603, 0.610,
        0.614, 0.615, 0.613, 0.607, 0.600
    ],

    "ppe_depreciation_pct_revenue": [
        0.125, 0.132, 0.138, 0.143, 0.147,
        0.151, 0.154, 0.156, 0.158, 0.159
    ],

    "intangible_amortization": [
        3_097, 2_141, 1_944, 1_477, 1_128,
        950, 850, 780, 720, 670
    ],

    "capex_pct_revenue": [
        0.340, 0.320, 0.290, 0.260, 0.240,
        0.230, 0.220, 0.210, 0.200, 0.186144912401586
    ],

    "tax_rate": [
        0.200, 0.195, 0.190, 0.190, 0.190,
        0.193, 0.196, 0.199, 0.202, 0.205
    ],

    "wc_cash_impact_pct_revenue": [
        0.0100, 0.0100, 0.0080, 0.0080, 0.0050,
        0.0057, 0.0048, 0.0038, 0.0029, 0.0019
    ],
}


# ============================================================
# FORECAST ENGINE
# ============================================================

def build_forecast(
    starting_revenue: float = INPUTS.starting_revenue,
    assumptions: Dict[str, List[float]] = BASE_ASSUMPTIONS,
):
    """
    Build the FY2027E-FY2036E unlevered cash flow forecast.

    UFCF =
        NOPAT
        + Total D&A
        - Capital Expenditure
        + Working Capital Cash Impact
    """

    revenue = starting_revenue
    forecast = []

    for i, year in enumerate(YEARS):

        revenue *= 1 + assumptions["revenue_growth"][i]

        ebitda = revenue * assumptions["ebitda_margin"][i]

        ppe_depreciation = (
            revenue
            * assumptions["ppe_depreciation_pct_revenue"][i]
        )

        intangible_amortization = (
            assumptions["intangible_amortization"][i]
        )

        total_da = (
            ppe_depreciation
            + intangible_amortization
        )

        ebit = ebitda - total_da

        tax_rate = assumptions["tax_rate"][i]

        nopat = ebit * (1 - tax_rate)

        capex = (
            revenue
            * assumptions["capex_pct_revenue"][i]
        )

        working_capital_cash_impact = (
            revenue
            * assumptions["wc_cash_impact_pct_revenue"][i]
        )

        ufcf = (
            nopat
            + total_da
            - capex
            + working_capital_cash_impact
        )

        forecast.append(
            {
                "year": year,
                "revenue": revenue,
                "ebitda": ebitda,
                "ebit": ebit,
                "nopat": nopat,
                "ppe_depreciation": ppe_depreciation,
                "intangible_amortization": intangible_amortization,
                "total_da": total_da,
                "capex": capex,
                "wc_cash_impact": working_capital_cash_impact,
                "ufcf": ufcf,
            }
        )

    return forecast


# ============================================================
# DISCOUNTING
# ============================================================

def year_fraction(start: date, end: date) -> float:
    """
    Actual-day-count convention divided by 365.25.
    """
    return (end - start).days / 365.25


def midpoint_date(fiscal_year: int) -> date:
    """
    Normalized mid-year date used by the portfolio model.

    Microsoft's fiscal year ends 30 June.
    """
    fiscal_year_end = date(fiscal_year, 6, 30)

    return fiscal_year_end - timedelta(days=183)


# ============================================================
# DCF VALUATION
# ============================================================

def value_dcf(
    forecast,
    wacc: float = INPUTS.wacc,
    terminal_growth: float = INPUTS.terminal_growth,
):
    """
    Calculate enterprise value, equity value and value per share.
    """

    if terminal_growth >= wacc:
        raise ValueError(
            "Terminal growth must be lower than WACC."
        )

    pv_explicit = 0.0

    for row in forecast:

        discount_period = year_fraction(
            INPUTS.valuation_date,
            midpoint_date(row["year"]),
        )

        present_value = (
            row["ufcf"]
            / ((1 + wacc) ** discount_period)
        )

        pv_explicit += present_value

    final_ufcf = forecast[-1]["ufcf"]

    terminal_value = (
        final_ufcf
        * (1 + terminal_growth)
        / (wacc - terminal_growth)
    )

    terminal_date = date(2036, 6, 30)

    terminal_period = year_fraction(
        INPUTS.valuation_date,
        terminal_date,
    )

    pv_terminal = (
        terminal_value
        / ((1 + wacc) ** terminal_period)
    )

    enterprise_value = (
        pv_explicit
        + pv_terminal
    )

    net_cash = (
        INPUTS.cash_and_investments
        - INPUTS.total_debt
    )

    equity_value = (
        enterprise_value
        + net_cash
    )

    value_per_share = (
        equity_value
        / INPUTS.diluted_shares_m
    )

    terminal_value_share = (
        pv_terminal
        / enterprise_value
    )

    return {
        "pv_explicit": pv_explicit,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": value_per_share,
        "terminal_value_share": terminal_value_share,
    }


# ============================================================
# REVERSE DCF
# ============================================================

def bisect_root(
    function,
    lower: float,
    upper: float,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
):
    """
    Simple bisection root solver using only Python's standard library.
    """

    f_lower = function(lower)
    f_upper = function(upper)

    if f_lower * f_upper > 0:
        raise ValueError(
            "Root is not bracketed by the supplied range."
        )

    for _ in range(max_iterations):

        midpoint = (lower + upper) / 2
        f_midpoint = function(midpoint)

        if abs(f_midpoint) < tolerance:
            return midpoint

        if f_lower * f_midpoint <= 0:
            upper = midpoint
            f_upper = f_midpoint
        else:
            lower = midpoint
            f_lower = f_midpoint

    return (lower + upper) / 2


def implied_terminal_growth(forecast) -> float:

    def objective(growth):
        valuation = value_dcf(
            forecast,
            wacc=INPUTS.wacc,
            terminal_growth=growth,
        )

        return (
            valuation["value_per_share"]
            - INPUTS.market_price
        )

    return bisect_root(
        objective,
        lower=0.00,
        upper=INPUTS.wacc - 0.001,
    )


def implied_wacc(forecast) -> float:

    def objective(wacc):
        valuation = value_dcf(
            forecast,
            wacc=wacc,
            terminal_growth=INPUTS.terminal_growth,
        )

        return (
            valuation["value_per_share"]
            - INPUTS.market_price
        )

    return bisect_root(
        objective,
        lower=0.04,
        upper=0.15,
    )


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_analysis(forecast):

    wacc_values = [
        0.0800,
        0.0850,
        0.0900,
        INPUTS.wacc,
        0.0950,
        0.1000,
        0.1050,
    ]

    growth_values = [
        0.020,
        0.025,
        0.030,
        0.035,
    ]

    print("\nWACC / TERMINAL GROWTH SENSITIVITY")
    print("-" * 62)

    header = "WACC     " + "".join(
        f"{g:>12.1%}"
        for g in growth_values
    )

    print(header)

    for wacc in wacc_values:

        values = []

        for growth in growth_values:

            valuation = value_dcf(
                forecast,
                wacc=wacc,
                terminal_growth=growth,
            )

            values.append(
                valuation["value_per_share"]
            )

        row = f"{wacc:>6.2%}  " + "".join(
            f"${value:>11.2f}"
            for value in values
        )

        print(row)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    forecast = build_forecast()

    base = value_dcf(forecast)

    market_implied_growth = (
        implied_terminal_growth(forecast)
    )

    market_implied_wacc = (
        implied_wacc(forecast)
    )

    print("\nMICROSOFT DCF VALUATION")
    print("=" * 45)

    print(
        f"PV of explicit forecast: "
        f"${base['pv_explicit'] / 1_000:,.1f}bn"
    )

    print(
        f"PV of terminal value:    "
        f"${base['pv_terminal'] / 1_000:,.1f}bn"
    )

    print(
        f"Enterprise value:        "
        f"${base['enterprise_value'] / 1_000:,.1f}bn"
    )

    print(
        f"Equity value:            "
        f"${base['equity_value'] / 1_000:,.1f}bn"
    )

    print(
        f"Base value per share:    "
        f"${base['value_per_share']:,.2f}"
    )

    print(
        f"Reference market price:  "
        f"${INPUTS.market_price:,.2f}"
    )

    print(
        f"Terminal value / EV:     "
        f"{base['terminal_value_share']:.1%}"
    )

    print("\nREVERSE DCF")
    print("-" * 45)

    print(
        f"Market-implied terminal growth: "
        f"{market_implied_growth:.2%}"
    )

    print(
        f"Market-implied WACC:            "
        f"{market_implied_wacc:.2%}"
    )

    sensitivity_analysis(forecast)
