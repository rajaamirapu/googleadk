"""
Quick demo — run the agent programmatically (no web UI needed).

Usage:
    python demo.py

Make sure:
  1. Ollama is running:  `ollama serve`
  2. Model is pulled:    `ollama pull llama3.2`
  3. Dependencies installed: `pip install -r requirements.txt`
"""

import asyncio
import os

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from weather_sunrise_agent import root_agent

SESSION_ID = "demo-session-001"
USER_ID = "demo-user"
APP_NAME = "weather_sunrise_demo"


async def run_query(runner: Runner, session_id: str, query: str) -> str:
    """Send a single query to the agent and return the final text response."""
    content = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )
    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
    return final_response


async def main():
    print("=" * 60)
    print("  Weather & Sunrise/Sunset Agent  |  Powered by Ollama")
    print("=" * 60)

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ── Demo queries ──────────────────────────────────────────
    queries = [
        # New York City
        "Give me a full weather and sunrise/sunset report for latitude 40.7128, longitude -74.0060 (New York City).",
        # Tokyo
        "What's the current weather at lat 35.6895, lon 139.6917?",
        # London
        "When is sunrise and sunset today at 51.5074, -0.1278 (London)?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 60)
        response = await run_query(runner, SESSION_ID, query)
        print(response)
        print()


if __name__ == "__main__":
    asyncio.run(main())
