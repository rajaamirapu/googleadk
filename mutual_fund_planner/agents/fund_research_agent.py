"""
fund_research_agent.py
──────────────────────
Sub-agent: pulls and analyses mutual fund data from AMFI and Moneycontrol.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from google.adk.agents import Agent
from custom_llm.adk_langchain_bridge import LangChainADKBridge
from custom_llm.base_custom_llm import CustomChatLLM

from mutual_fund_planner.tools.amfi_tools import (
    get_fund_nav,
    search_mutual_funds,
    get_fund_historical_nav,
    get_top_funds_by_category,
    compare_funds,
)
from mutual_fund_planner.tools.moneycontrol_tools import (
    get_fund_details_moneycontrol,
    get_top_mutual_funds_moneycontrol,
    get_market_overview_moneycontrol,
    get_fund_manager_info,
)

_llm = LangChainADKBridge(
    langchain_llm=CustomChatLLM(
        base_url=os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434/v1"),
        model_name=os.getenv("CUSTOM_LLM_MODEL", "llama3.2"),
        api_key=os.getenv("CUSTOM_LLM_API_KEY", "custom"),
        temperature=0.2,
    ),
    model=f"custom/{os.getenv('CUSTOM_LLM_MODEL', 'llama3.2')}",
)

fund_research_agent = Agent(
    name="fund_research_agent",
    model=_llm,
    description=(
        "Specialist agent for mutual fund research. Fetches NAV, historical "
        "performance, fund categories, top funds, and Moneycontrol data."
    ),
    instruction="""
You are a mutual fund research specialist for Indian markets.

Your job is to fetch and present factual fund data. Use the available tools to:

1. **Search & identify funds** — `search_mutual_funds(query)` to find AMFI scheme codes.
2. **Get current NAV** — `get_fund_nav(scheme_code)` using the AMFI scheme code.
3. **Historical performance** — `get_fund_historical_nav(scheme_code)` for 1Y returns.
4. **Category leaders** — `get_top_funds_by_category(category)` for large/mid/small cap etc.
5. **Compare funds** — `compare_funds([code1, code2, ...])` for side-by-side NAV comparison.
6. **Moneycontrol details** — `get_fund_details_moneycontrol(fund_name)` for ratings and AUM.
7. **Top funds by category** — `get_top_mutual_funds_moneycontrol(category)` for ranked lists.
8. **Fund manager** — `get_fund_manager_info(fund_house)` for who manages the fund.

Always:
- Use AMFI data (mftool) for NAV and official figures — it's the most reliable.
- Use Moneycontrol for ratings, AUM, expense ratio, and fund manager details.
- Present returns clearly: 1Y, 3Y, 5Y in % terms.
- Mention the AMFI scheme code whenever you present a fund so users can track it.
- Flag if data is unavailable rather than guessing.

Categories you understand:
  large cap, mid cap, small cap, multi cap, ELSS, debt, liquid, hybrid,
  index, balanced advantage, flexi cap, international, sectoral.
""",
    tools=[
        get_fund_nav,
        search_mutual_funds,
        get_fund_historical_nav,
        get_top_funds_by_category,
        compare_funds,
        get_fund_details_moneycontrol,
        get_top_mutual_funds_moneycontrol,
        get_market_overview_moneycontrol,
        get_fund_manager_info,
    ],
)
