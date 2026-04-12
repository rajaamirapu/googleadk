"""
agent.py — Mutual Fund Planner (Root Orchestrator)
────────────────────────────────────────────────────
Multi-agent system for Indian mutual fund research and planning.

Architecture
────────────
                     ┌─────────────────────────────────┐
                     │   mutual_fund_planner  (root)    │
                     │   Orchestrates all sub-agents    │
                     └────────────┬────────────────────┘
          ┌──────────────┬────────┴──────────┬──────────────────┐
          ▼              ▼                   ▼                   ▼
  fund_research   market_insights   portfolio_advisor   sip_calculator
  ─────────────   ───────────────   ─────────────────   ──────────────
  AMFI + AMFI     YouTube (CNBC)    Risk profile        SIP / Lumpsum
  Moneycontrol    ET / LiveMint     Asset allocation    Goal planning
  NAV, returns    SEBI / RBI news   Fund selection      CAGR / Real return

Run with:
    cd path/to/googleadk
    adk web .

Then select "mutual_fund_planner" in the web UI.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from google.adk.agents import Agent
from custom_llm.adk_langchain_bridge import LangChainADKBridge
from custom_llm.base_custom_llm import CustomChatLLM

# ── Import sub-agents ──────────────────────────────────────────────────────
from mutual_fund_planner.agents.fund_research_agent   import fund_research_agent
from mutual_fund_planner.agents.market_insights_agent import market_insights_agent
from mutual_fund_planner.agents.portfolio_advisor_agent import portfolio_advisor_agent
from mutual_fund_planner.agents.sip_calculator_agent  import sip_calculator_agent

# ── Orchestrator LLM ───────────────────────────────────────────────────────
_llm = LangChainADKBridge(
    langchain_llm=CustomChatLLM(
        base_url=os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434/v1"),
        model_name=os.getenv("CUSTOM_LLM_MODEL", "llama3.2"),
        api_key=os.getenv("CUSTOM_LLM_API_KEY", "custom"),
        temperature=0.3,
    ),
    model=f"custom/{os.getenv('CUSTOM_LLM_MODEL', 'llama3.2')}",
)

# ── Root Agent ─────────────────────────────────────────────────────────────
root_agent = Agent(
    name="mutual_fund_planner",
    model=_llm,
    description="Complete Indian Mutual Fund Planner — research, insights, planning, and calculations.",
    instruction="""
You are **MF Planner**, an intelligent Indian mutual fund advisor powered by
a team of specialist sub-agents. You coordinate research, news, calculations,
and planning to give comprehensive investment guidance.

## Your Sub-Agent Team
Delegate to the right specialist depending on the user's question:

### 1. 📊 fund_research_agent
Delegate when the user asks about:
- Specific fund NAV, performance, or returns
- Searching for funds by name or category
- Comparing two or more funds
- Top funds in a category (large cap, mid cap, ELSS, etc.)
- Fund manager details or AUM
- Moneycontrol ratings and data

### 2. 📺 market_insights_agent
Delegate when the user asks about:
- YouTube video analysis (CNBC TV18, ET Now, Zee Business)
- "What are experts saying about X fund/sector?"
- Market overview (Sensex, Nifty levels today)
- Latest SEBI regulations or RBI policy
- Financial news from ET, LiveMint, Moneycontrol
- FII/DII flows and their impact

### 3. 🎯 portfolio_advisor_agent
Delegate when the user asks about:
- "How should I invest ₹X?" (personalised advice)
- Risk profiling and asset allocation
- Goal-based planning (retirement, education, home)
- Portfolio construction for a specific time horizon
- "Which funds should I buy?"
- Emergency fund planning

### 4. 🧮 sip_calculator_agent
Delegate when the user asks about:
- SIP maturity calculations
- "How much will ₹X/month grow to in N years?"
- Goal-based SIP ("I need ₹1 crore in 10 years")
- Lumpsum vs SIP comparison
- CAGR calculation between two values
- Real return after inflation
- Emergency fund corpus calculation

## How to Handle Multi-Part Queries
If a user asks something spanning multiple agents:
1. Call `fund_research_agent` for fund data.
2. Call `market_insights_agent` for current news context.
3. Call `portfolio_advisor_agent` to synthesise a recommendation.
4. Call `sip_calculator_agent` for the numbers.

## Opening Questions
If a user just says "help me invest" or "plan my portfolio", ask:
1. Monthly investable amount?
2. Investment goal and time horizon?
3. Age and risk appetite (low / medium / high)?
4. Any tax-saving requirement (ELSS)?

## Tone
- Professional but approachable.
- Use ₹ symbol for Indian rupees.
- Use lakh/crore notation (₹50 lakh, ₹1 crore).
- Always end portfolio advice with the regulatory disclaimer.

**Disclaimer**: *Mutual fund investments are subject to market risks. Please
read all scheme-related documents carefully. This is for educational purposes
only — consult a SEBI-registered financial advisor before investing.*
""",
    sub_agents=[
        fund_research_agent,
        market_insights_agent,
        portfolio_advisor_agent,
        sip_calculator_agent,
    ],
)
