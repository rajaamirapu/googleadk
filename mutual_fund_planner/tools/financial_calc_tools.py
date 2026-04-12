"""
financial_calc_tools.py
───────────────────────
Pure financial calculation tools — no external API calls.

Covers:
  • SIP (Systematic Investment Plan) returns
  • Lump-sum investment returns
  • Goal-based SIP planning (how much to invest to reach a target)
  • CAGR calculation
  • Risk profiling
  • Asset allocation recommendations
  • Inflation-adjusted return calculations
"""

from __future__ import annotations
import math


# ─────────────────────────────────────────────────────────────────────────────
# SIP & Lump-sum calculators
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sip_returns(
    monthly_investment: float,
    annual_return_rate: float,
    years: int,
) -> dict:
    """
    Calculate the future value of a monthly SIP investment.

    Args:
        monthly_investment: Monthly SIP amount in ₹
        annual_return_rate: Expected annual return rate in % (e.g. 12.0 for 12%)
        years:              Investment duration in years

    Returns:
        Dict with invested amount, estimated returns, total corpus, and XIRR.
    """
    if monthly_investment <= 0 or annual_return_rate <= 0 or years <= 0:
        return {"error": "All inputs must be positive numbers"}

    r = annual_return_rate / 100 / 12   # monthly rate
    n = years * 12                       # total months

    # FV of SIP = P × [(1+r)^n - 1] / r × (1+r)
    fv = monthly_investment * (((1 + r) ** n - 1) / r) * (1 + r)

    invested = monthly_investment * n
    returns  = fv - invested

    return {
        "monthly_sip":          f"₹{monthly_investment:,.0f}",
        "annual_return_rate":   f"{annual_return_rate}%",
        "duration_years":       years,
        "total_invested":       f"₹{invested:,.0f}",
        "estimated_returns":    f"₹{returns:,.0f}",
        "total_corpus":         f"₹{fv:,.0f}",
        "wealth_gain_multiple": round(fv / invested, 2),
    }


def calculate_lumpsum_returns(
    principal: float,
    annual_return_rate: float,
    years: int,
) -> dict:
    """
    Calculate the future value of a one-time lump-sum investment.

    Args:
        principal:          Lump-sum amount in ₹
        annual_return_rate: Expected annual return rate in % (e.g. 12.0)
        years:              Investment duration in years

    Returns:
        Dict with principal, maturity amount, and absolute returns.
    """
    if principal <= 0 or annual_return_rate <= 0 or years <= 0:
        return {"error": "All inputs must be positive numbers"}

    fv      = principal * (1 + annual_return_rate / 100) ** years
    returns = fv - principal

    return {
        "principal":          f"₹{principal:,.0f}",
        "annual_return_rate": f"{annual_return_rate}%",
        "duration_years":     years,
        "maturity_amount":    f"₹{fv:,.0f}",
        "absolute_returns":   f"₹{returns:,.0f}",
        "return_multiple":    round(fv / principal, 2),
    }


def calculate_goal_based_sip(
    target_amount: float,
    annual_return_rate: float,
    years: int,
) -> dict:
    """
    Calculate the monthly SIP needed to reach a financial goal.

    Args:
        target_amount:      Target corpus in ₹ (e.g. 1_00_00_000 for 1 crore)
        annual_return_rate: Expected annual return rate in %
        years:              Time horizon in years

    Returns:
        Dict with required monthly SIP, total investment, and projected corpus.
    """
    if target_amount <= 0 or annual_return_rate <= 0 or years <= 0:
        return {"error": "All inputs must be positive numbers"}

    r = annual_return_rate / 100 / 12
    n = years * 12

    # Reverse SIP formula: P = FV × r / [(1+r)^n - 1] / (1+r)
    monthly_sip = target_amount * r / (((1 + r) ** n - 1) * (1 + r))

    invested = monthly_sip * n

    return {
        "target_corpus":        f"₹{target_amount:,.0f}",
        "annual_return_rate":   f"{annual_return_rate}%",
        "duration_years":       years,
        "required_monthly_sip": f"₹{monthly_sip:,.0f}",
        "total_investment":     f"₹{invested:,.0f}",
        "returns_earned":       f"₹{target_amount - invested:,.0f}",
    }


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> dict:
    """
    Calculate the Compound Annual Growth Rate (CAGR) between two values.

    Args:
        initial_value: Starting NAV or investment value
        final_value:   Ending NAV or investment value
        years:         Number of years between the two values

    Returns:
        Dict with CAGR percentage and absolute return.
    """
    if initial_value <= 0 or final_value <= 0 or years <= 0:
        return {"error": "All values must be positive numbers"}

    cagr = (final_value / initial_value) ** (1 / years) - 1
    absolute_return = (final_value - initial_value) / initial_value * 100

    return {
        "initial_value":     f"₹{initial_value:,.2f}",
        "final_value":       f"₹{final_value:,.2f}",
        "years":             years,
        "cagr":              f"{cagr * 100:.2f}%",
        "absolute_return":   f"{absolute_return:.2f}%",
    }


def calculate_inflation_adjusted_return(
    nominal_return_pct: float,
    inflation_rate_pct: float,
) -> dict:
    """
    Calculate the real (inflation-adjusted) return on an investment.

    Args:
        nominal_return_pct: Nominal annual return in %
        inflation_rate_pct: Annual inflation rate in % (India avg ~6%)

    Returns:
        Dict with real return and purchasing power erosion analysis.
    """
    real_return = ((1 + nominal_return_pct / 100) / (1 + inflation_rate_pct / 100) - 1) * 100

    return {
        "nominal_return":    f"{nominal_return_pct}%",
        "inflation_rate":    f"{inflation_rate_pct}%",
        "real_return":       f"{real_return:.2f}%",
        "note":              "Real return = purchasing power after inflation adjustment",
        "verdict":           "Beats inflation ✅" if real_return > 0 else "Loses to inflation ❌",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Risk & Portfolio tools
# ─────────────────────────────────────────────────────────────────────────────

def assess_risk_profile(
    age: int,
    monthly_income: float,
    investment_horizon_years: int,
    existing_loans: bool,
    dependents: int,
) -> dict:
    """
    Assess investor risk profile based on personal financial parameters.

    Args:
        age:                       Investor age
        monthly_income:            Monthly income in ₹
        investment_horizon_years:  How long they plan to stay invested
        existing_loans:            Whether they have existing loans
        dependents:                Number of financial dependents

    Returns:
        Dict with risk profile (Conservative / Moderate / Aggressive) and rationale.
    """
    score = 0

    # Age scoring (younger = more risk capacity)
    if age < 30:      score += 3
    elif age < 40:    score += 2
    elif age < 50:    score += 1

    # Horizon scoring
    if investment_horizon_years >= 10:  score += 3
    elif investment_horizon_years >= 5: score += 2
    else:                               score += 1

    # Loans penalty
    if existing_loans:
        score -= 1

    # Dependents penalty
    score -= min(dependents, 2)

    # Classify
    if score >= 6:
        profile = "Aggressive"
        allocation = {"equity": "80-90%", "debt": "5-10%", "gold": "5-10%"}
        fund_types = ["Small Cap", "Mid Cap", "Flexi Cap", "Sector Funds"]
    elif score >= 3:
        profile = "Moderate"
        allocation = {"equity": "60-70%", "debt": "20-25%", "gold": "10-15%"}
        fund_types = ["Large Cap", "Large & Mid Cap", "Hybrid Equity", "Balanced Advantage"]
    else:
        profile = "Conservative"
        allocation = {"equity": "20-30%", "debt": "55-65%", "gold": "10-15%"}
        fund_types = ["Liquid Funds", "Short Duration Debt", "Conservative Hybrid", "ELSS for tax saving"]

    return {
        "risk_profile":             profile,
        "score":                    score,
        "recommended_allocation":   allocation,
        "suitable_fund_types":      fund_types,
        "tax_saving_note":          "Consider ELSS funds for Section 80C tax benefit (up to ₹1.5L/year)",
    }


def calculate_portfolio_allocation(
    total_investment: float,
    risk_profile: str,
) -> dict:
    """
    Calculate recommended fund-wise portfolio allocation.

    Args:
        total_investment: Total amount to invest in ₹
        risk_profile:     "conservative", "moderate", or "aggressive"

    Returns:
        Dict with amount to allocate to each fund category.
    """
    allocations = {
        "conservative": {
            "Liquid / Overnight Fund":    0.20,
            "Short Duration Debt":        0.35,
            "Conservative Hybrid":        0.25,
            "Large Cap Equity":           0.15,
            "Gold ETF / Sovereign Gold":  0.05,
        },
        "moderate": {
            "Large Cap Fund":             0.30,
            "Large & Mid Cap Fund":       0.20,
            "Balanced Advantage Fund":    0.15,
            "Short Duration Debt":        0.20,
            "Gold ETF":                   0.10,
            "International Fund":         0.05,
        },
        "aggressive": {
            "Flexi Cap / Multi Cap":      0.25,
            "Mid Cap Fund":               0.25,
            "Small Cap Fund":             0.20,
            "Sector / Thematic Fund":     0.15,
            "International Equity Fund":  0.10,
            "Gold ETF":                   0.05,
        },
    }

    profile_key = risk_profile.lower()
    if profile_key not in allocations:
        return {"error": f"Invalid profile. Choose: conservative, moderate, or aggressive"}

    alloc = allocations[profile_key]
    result = {
        "total_investment": f"₹{total_investment:,.0f}",
        "risk_profile":     risk_profile.capitalize(),
        "allocation": {}
    }
    for fund_type, pct in alloc.items():
        amount = total_investment * pct
        result["allocation"][fund_type] = {
            "percentage": f"{pct * 100:.0f}%",
            "amount":     f"₹{amount:,.0f}",
        }
    return result


def calculate_emergency_fund(monthly_expenses: float, months: int = 6) -> dict:
    """
    Calculate the recommended emergency fund corpus.

    Args:
        monthly_expenses: Monthly household expenses in ₹
        months:           Months of expenses to keep as emergency fund (default 6)

    Returns:
        Dict with emergency fund target and recommended fund types.
    """
    target = monthly_expenses * months
    return {
        "monthly_expenses":        f"₹{monthly_expenses:,.0f}",
        "recommended_months":      months,
        "emergency_fund_target":   f"₹{target:,.0f}",
        "where_to_keep": [
            "Liquid Mutual Fund (e.g. HDFC Liquid Fund) — instant redemption",
            "Overnight Fund — lowest risk",
            "Savings Bank Account — for immediate access",
        ],
        "note": "Keep 3-6 months expenses in liquid assets before investing in equities",
    }
