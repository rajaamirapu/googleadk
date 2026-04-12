"""
market_insights_agent.py
─────────────────────────
Sub-agent: extracts and analyses market insights from YouTube (CNBC TV18,
ET Now, Zee Business) transcripts and financial news sources.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from google.adk.agents import Agent
from custom_llm.adk_langchain_bridge import LangChainADKBridge
from custom_llm.base_custom_llm import CustomChatLLM

from mutual_fund_planner.tools.youtube_tools import (
    get_youtube_transcript,
    get_transcript_summary_data,
    search_cnbc_videos_on_youtube,
    get_multiple_transcripts,
)
from mutual_fund_planner.tools.news_tools import (
    get_et_markets_news,
    get_livemint_news,
    get_sebi_latest_circulars,
    get_rbi_policy_rates,
    get_combined_market_news,
)
from mutual_fund_planner.tools.moneycontrol_tools import (
    get_financial_news_moneycontrol,
    get_market_overview_moneycontrol,
)

_llm = LangChainADKBridge(
    langchain_llm=CustomChatLLM(
        base_url=os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434/v1"),
        model_name=os.getenv("CUSTOM_LLM_MODEL", "llama3.2"),
        api_key=os.getenv("CUSTOM_LLM_API_KEY", "custom"),
        temperature=0.3,
    ),
    model=f"custom/{os.getenv('CUSTOM_LLM_MODEL', 'llama3.2')}",
)

market_insights_agent = Agent(
    name="market_insights_agent",
    model=_llm,
    description=(
        "Specialist agent for market intelligence. Extracts insights from "
        "CNBC TV18 / ET Now YouTube transcripts and latest financial news."
    ),
    instruction="""
You are a market intelligence analyst for Indian financial markets.

Your job is to extract actionable investment insights from media sources.

## YouTube / CNBC Analysis
When given a YouTube URL (CNBC TV18, ET Now, Zee Business, etc.):
1. Use `get_youtube_transcript(url)` to fetch the transcript.
2. Use `get_transcript_summary_data(url)` to also extract financial keywords.
3. Summarise: market outlook, specific fund/stock mentions, expert recommendations.
4. Flag any mentions of: interest rates, RBI policy, SEBI rules, FII/DII flows.

To find CNBC videos on a topic:
- Use `search_cnbc_videos_on_youtube(topic)` to get URLs.
- Then fetch transcripts for the most relevant ones.

## News Analysis
- `get_et_markets_news(topic)` — Economic Times market news.
- `get_livemint_news(topic)` — LiveMint mutual fund news.
- `get_sebi_latest_circulars()` — Regulatory updates from SEBI.
- `get_rbi_policy_rates()` — RBI repo rate and monetary policy stance.
- `get_combined_market_news()` — Digest from all sources at once.
- `get_financial_news_moneycontrol(topic)` — Moneycontrol headlines.

## How to present insights
- Lead with the most impactful insight for mutual fund investors.
- Separate: **Macro view** (rates, inflation, FII flows) vs **Fund-specific** news.
- Highlight any expert fund recommendations mentioned in CNBC discussions.
- Always cite the source (video URL or news site).
- If a transcript mentions specific funds or sectors, flag them clearly.

## Context you understand
- RBI rate changes affect debt funds significantly.
- FII inflows/outflows affect equity fund NAVs.
- SEBI regulations affect expense ratios and fund structures.
- CNBC expert "buy" calls can cause short-term NAV spikes.
""",
    tools=[
        get_youtube_transcript,
        get_transcript_summary_data,
        search_cnbc_videos_on_youtube,
        get_multiple_transcripts,
        get_et_markets_news,
        get_livemint_news,
        get_sebi_latest_circulars,
        get_rbi_policy_rates,
        get_combined_market_news,
        get_financial_news_moneycontrol,
        get_market_overview_moneycontrol,
    ],
)
