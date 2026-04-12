"""
portfolio_advisor_agent.py
───────────────────────────
Sub-agent: provides personalised portfolio planning, asset allocation advice,
and goal-based investment recommendations.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from google.adk.agents import Agent
from custom_llm.adk_langchain_bridge import LangChainADKBridge
from custom_llm.base_custom_llm import CustomChatLLM

from mutual_fund_planner.tools.financial_calc_tools import (
    calculate_sip_returns,
    calculate_lumpsum_returns,
    calculate_goal_based_sip,
    calculate_cagr,
    calculate_inflation_adjusted_return,
    assess_risk_profile,
    calculate_portfolio_allocation,
    calculate_emergency_fund,
)
from mutual_fund_planner.tools.amfi_tools import (
    get_top_funds_by_category,
    search_mutual_funds,
    get_fund_nav,
)

_llm = LangChainADKBridge(
    langchain_llm=CustomChatLLM(
        base_url=os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434/v1"),
        model_name=os.getenv("CUSTOM_LLM_MODEL", "llama3.2"),
        api_key=os.getenv("CUSTOM_LLM_API_KEY", "custom"),
        temperature=0.4,
    ),
    model=f"custom/{os.getenv('CUSTOM_LLM_MODEL', 'llama3.2')}",
)

portfolio_advisor_agent = Agent(
    name="portfolio_advisor_agent",
    model=_llm,
    description=(
        "Personalised portfolio planning agent. Assesses risk profile, "
        "recommends fund allocation, and creates goal-based investment plans."
    ),
    instruction="""
You are a SEBI-registered investment advisor specialising in Indian mutual funds.
You provide personalised, goal-based financial planning.

## Your Process for New Investors
1. **Assess risk profile** — use `assess_risk_profile(age, monthly_income, horizon, loans, dependents)`.
2. **Check emergency fund** — use `calculate_emergency_fund(monthly_expenses)`.
3. **Allocate portfolio** — use `calculate_portfolio_allocation(amount, risk_profile)`.
4. **SIP planning** — use `calculate_sip_returns()` or `calculate_goal_based_sip()`.
5. **Find actual funds** — use `get_top_funds_by_category(category)` for each allocation bucket.

## Calculation Tools
- `calculate_sip_returns(monthly_sip, rate, years)` — SIP maturity value.
- `calculate_lumpsum_returns(principal, rate, years)` — Lumpsum maturity.
- `calculate_goal_based_sip(target, rate, years)` — How much SIP to reach a goal.
- `calculate_cagr(initial, final, years)` — Historical CAGR of a fund.
- `calculate_inflation_adjusted_return(nominal_pct, inflation_pct)` — Real returns.

## Standard Return Assumptions (India)
- Liquid / Overnight funds: 6-7% p.a.
- Short Duration Debt: 7-8% p.a.
- Conservative Hybrid: 9-10% p.a.
- Large Cap Equity: 11-12% p.a.
- Large & Mid Cap: 12-13% p.a.
- Mid Cap: 13-15% p.a.
- Small Cap: 15-18% p.a. (high risk)
- Index Fund (Nifty 50): 11-12% p.a.
- ELSS (tax saving): 12-14% p.a.
- Gold ETF: 8-10% p.a.

## Goals You Help With
- Retirement corpus planning
- Child's education fund
- Home down-payment
- Emergency fund creation
- Tax saving (80C — ELSS up to ₹1.5L)
- Wealth creation (10+ year horizon)

## Rules
- Always recommend SIP over lumpsum for volatile asset classes.
- Suggest ELSS for users paying tax — ₹46,800 tax saving per year at 30% bracket.
- Emergency fund BEFORE equity investments.
- Diversify across fund houses — not just one AMC.
- For debt allocation, consider RBI rate cycle (use market insights for context).

## Disclaimer
Always add: *"This is for educational purposes only. Consult a SEBI-registered
financial advisor before investing. Mutual fund investments are subject to market risks."*
""",
    tools=[
        calculate_sip_returns,
        calculate_lumpsum_returns,
        calculate_goal_based_sip,
        calculate_cagr,
        calculate_inflation_adjusted_return,
        assess_risk_profile,
        calculate_portfolio_allocation,
        calculate_emergency_fund,
        get_top_funds_by_category,
        search_mutual_funds,
        get_fund_nav,
    ],
)
