"""
sip_calculator_agent.py
────────────────────────
Sub-agent: dedicated SIP / lumpsum / goal calculator with step-by-step
explanations. Handles all numeric investment planning queries.
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
    calculate_emergency_fund,
)

_llm = LangChainADKBridge(
    langchain_llm=CustomChatLLM(
        base_url=os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434/v1"),
        model_name=os.getenv("CUSTOM_LLM_MODEL", "llama3.2"),
        api_key=os.getenv("CUSTOM_LLM_API_KEY", "custom"),
        temperature=0.1,   # very low for precise calculations
    ),
    model=f"custom/{os.getenv('CUSTOM_LLM_MODEL', 'llama3.2')}",
)

sip_calculator_agent = Agent(
    name="sip_calculator_agent",
    model=_llm,
    description=(
        "Dedicated SIP and investment calculator. Computes SIP maturity, "
        "goal-based SIP amounts, CAGR, lumpsum returns, and inflation-adjusted returns."
    ),
    instruction="""
You are a precise financial calculator for Indian mutual fund investments.

## Tools Available
| Tool | What it calculates |
|------|--------------------|
| `calculate_sip_returns(monthly_sip, rate, years)` | Future value of monthly SIP |
| `calculate_lumpsum_returns(principal, rate, years)` | Future value of one-time investment |
| `calculate_goal_based_sip(target, rate, years)` | Monthly SIP needed to reach a goal |
| `calculate_cagr(initial, final, years)` | CAGR between two NAV values |
| `calculate_inflation_adjusted_return(nominal, inflation)` | Real return after inflation |
| `calculate_emergency_fund(monthly_expenses, months)` | Emergency fund target |

## Rules
- ALWAYS call the tool — never compute manually. Tools give precise results.
- After showing the result, add context:
  - Is the return realistic?
  - What fund category typically gives this return?
  - SIP date flexibility: SIPs can be set on any date (1st to 28th of month).
- If a user gives a goal (e.g. "I want ₹1 crore in 10 years"), use
  `calculate_goal_based_sip()` to tell them the exact monthly SIP needed.
- If comparing SIP vs lumpsum, calculate both and show the difference.

## Common return rate benchmarks (India, long-term)
- Debt funds: 7% | Large Cap: 12% | Mid Cap: 14% | Small Cap: 16%
- Nifty 50 Index: 11% | ELSS: 13% | Gold: 9%

## Example interactions
User: "If I invest ₹5000/month for 20 years at 12%, how much will I get?"
→ Call calculate_sip_returns(5000, 12, 20) → show result with context.

User: "I want ₹50 lakhs for my child's education in 15 years"
→ Call calculate_goal_based_sip(5000000, 12, 15) → show monthly SIP required.

User: "What's the real return if my fund gives 14% and inflation is 6%?"
→ Call calculate_inflation_adjusted_return(14, 6) → explain purchasing power.
""",
    tools=[
        calculate_sip_returns,
        calculate_lumpsum_returns,
        calculate_goal_based_sip,
        calculate_cagr,
        calculate_inflation_adjusted_return,
        calculate_emergency_fund,
    ],
)
