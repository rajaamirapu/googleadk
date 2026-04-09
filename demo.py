"""
demo.py — run the Weather & Sunrise/Sunset Agent programmatically.

Usage:
    python demo.py

Prerequisites:
    1. Your custom LLM server is running (e.g. `ollama serve`)
    2. pip install -r requirements.txt
    3. cp .env.example .env  (and fill in CUSTOM_LLM_BASE_URL, etc.)
"""

import asyncio

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
USER_ID    = "demo-user"
APP_NAME   = "weather_sunrise_demo"


async def ask(runner: Runner, session_id: str, query: str) -> str:
    content = types.Content(role="user", parts=[types.Part(text=query)])
    reply = ""
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            reply = event.content.parts[0].text
    return reply


async def main():
    print("=" * 62)
    print("  Weather & Sunrise/Sunset Agent  |  Custom LangChain LLM")
    print("=" * 62)

    session_svc = InMemorySessionService()
    await session_svc.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_svc,
    )

    queries = [
        "Give me a full weather and sunrise/sunset report for latitude 40.7128, longitude -74.0060 (New York City).",
        "What's the current weather at lat 35.6895, lon 139.6917 (Tokyo)?",
        "When does the sun rise and set in London today? Coordinates: 51.5074, -0.1278",
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n[Query {i}] {q}")
        print("-" * 62)
        print(await ask(runner, SESSION_ID, q))

if __name__ == "__main__":
    asyncio.run(main())
