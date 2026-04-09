"""
Weather & Sunrise/Sunset Agent — powered by Google ADK + Custom LangChain LLM.

How to use
──────────
# Interactive web UI
    adk web weather_sunrise_agent

# Single-turn CLI
    adk run weather_sunrise_agent

# Programmatic
    python demo.py

Swapping the LLM
────────────────
Edit the "LLM CONFIGURATION" section below to point at YOUR LangChain LLM.

Three options are pre-wired (uncomment the one you want):

  Option A — CustomChatLLM  (BaseChatModel, OpenAI-compatible endpoint)
  Option B — CustomTextLLM  (plain LLM, text completion endpoint)
  Option C — Drop in YOUR OWN LangChain LLM class directly
"""

import os
from google.adk.agents import Agent

# ─────────────────────────────────────────────────────────────
# LangChain → ADK bridge
# ─────────────────────────────────────────────────────────────
from custom_llm.adk_langchain_bridge import LangChainADKBridge
from custom_llm.base_custom_llm import CustomChatLLM, CustomTextLLM

# ─────────────────────────────────────────────────────────────
# Agent tools
# ─────────────────────────────────────────────────────────────
from weather_sunrise_agent.tools import get_weather, get_sunrise_sunset, get_full_report


# ═════════════════════════════════════════════════════════════
# LLM CONFIGURATION  — edit this section
# ═════════════════════════════════════════════════════════════

# ── Option A: CustomChatLLM (BaseChatModel, recommended) ─────
#    Uses any OpenAI-compatible chat endpoint (Ollama, vLLM, LM Studio, etc.)
_base_url  = os.getenv("CUSTOM_LLM_BASE_URL",  "http://localhost:11434/v1")
_model     = os.getenv("CUSTOM_LLM_MODEL",     "llama3.2")
_api_key   = os.getenv("CUSTOM_LLM_API_KEY",   "custom")

langchain_llm = CustomChatLLM(
    base_url=_base_url,
    model_name=_model,
    api_key=_api_key,
    temperature=0.3,    # lower = more deterministic tool calls
    max_tokens=2048,
)

# ── Option B: CustomTextLLM (plain LLM) ──────────────────────
#    Uncomment and comment out Option A above to switch.
# langchain_llm = CustomTextLLM(
#     base_url=os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434"),
#     model_name=os.getenv("CUSTOM_LLM_MODEL", "llama3.2"),
# )

# ── Option C: YOUR OWN LangChain LLM ─────────────────────────
#    Just replace `langchain_llm` with your own instance:
#
#    from my_module import MyCustomLLM
#    langchain_llm = MyCustomLLM(...)


# ─────────────────────────────────────────────────────────────
# Wrap the LangChain LLM for Google ADK
# ─────────────────────────────────────────────────────────────
adk_llm = LangChainADKBridge(
    langchain_llm=langchain_llm,
    model=f"custom/{_model}",   # display name shown in ADK web UI
)


# ─────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────
root_agent = Agent(
    name="weather_sunrise_agent",
    model=adk_llm,
    description=(
        "A helpful assistant that provides real-time weather conditions, "
        "sunrise, and sunset times for any location given latitude and longitude."
    ),
    instruction="""
You are a friendly and informative weather assistant.

When the user provides a latitude and longitude (or asks about a location you can resolve to lat/lng):

1. **Always call the appropriate tool(s)**:
   - `get_full_report`      → weather + sun times in one call (preferred)
   - `get_weather`          → current weather only
   - `get_sunrise_sunset`   → sunrise/sunset times only

2. **Present the data clearly**, including:
   - 🌡️  Temperature (°C) and feels-like
   - 💧 Humidity and precipitation
   - 🌬️ Wind speed and direction
   - ☁️  Cloud cover and weather description
   - 🌅 Sunrise and sunset times (UTC)
   - ⏳ Day length
   - 🌄 Civil / nautical twilight times when relevant

3. **Convert UTC times to local time** if you know the timezone from the weather data.

4. If the user gives a city name instead of coordinates, politely explain that you need
   numeric latitude and longitude values and give an example of how to find them.

5. Keep responses concise but complete — use emojis to make the output easy to scan.

Example user inputs:
  - "What's the weather at 40.7128, -74.0060?"
  - "Show me sunrise/sunset for lat 51.5, lon -0.12"
  - "Full report for 35.6895, 139.6917"
""",
    tools=[get_weather, get_sunrise_sunset, get_full_report],
)
