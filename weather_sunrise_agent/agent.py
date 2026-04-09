"""
Weather & Sunrise/Sunset Agent — powered by Google ADK + Ollama (via LiteLLM).

Usage:
    # Interactive web UI
    adk web weather_sunrise_agent

    # Single-turn CLI
    adk run weather_sunrise_agent

Make sure Ollama is running locally (default: http://localhost:11434)
and the desired model is pulled, e.g.:
    ollama pull llama3.2
"""

import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from weather_sunrise_agent.tools import get_weather, get_sunrise_sunset, get_full_report

# ─────────────────────────────────────────────────────────────
# Ollama model configuration
#
# Change OLLAMA_MODEL to any model you have pulled locally, e.g.:
#   llama3.2  |  llama3.1  |  gemma3  |  mistral  |  phi4
#
# IMPORTANT: Use the "ollama_chat/" prefix (not plain "ollama/")
#            to avoid tool-call loop issues.
# ─────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

ollama_llm = LiteLlm(
    model=f"ollama_chat/{OLLAMA_MODEL}",
    api_base=OLLAMA_BASE_URL,
)

# ─────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────
root_agent = Agent(
    name="weather_sunrise_agent",
    model=ollama_llm,
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
