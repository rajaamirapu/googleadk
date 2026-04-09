"""
debug_connection.py
───────────────────
Standalone diagnostic script — tests your custom LLM DIRECTLY,
completely bypassing ADK and the bridge.

Run this first to confirm your LLM works before testing the full agent.

Usage:
    python debug_connection.py

    # Override settings on the command line:
    CUSTOM_LLM_BASE_URL=http://localhost:8000/v1 \\
    CUSTOM_LLM_MODEL=my-model \\
    python debug_connection.py
"""

import asyncio
import json
import os
import sys
import traceback

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Read config from env (or edit defaults here) ──────────────────────────────
BASE_URL = os.getenv("CUSTOM_LLM_BASE_URL", "http://localhost:11434/v1")
MODEL    = os.getenv("CUSTOM_LLM_MODEL",    "llama3.2")
API_KEY  = os.getenv("CUSTOM_LLM_API_KEY",  "custom")

SEPARATOR = "─" * 62


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ── Test 1: raw HTTP connectivity ─────────────────────────────────────────────
def test_raw_http():
    section("Test 1 — Raw HTTP connection to LLM server")
    import requests

    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
        "max_tokens": 10,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    print(f"  URL   : {url}")
    print(f"  Model : {MODEL}")
    print(f"  Key   : {API_KEY[:4]}{'*' * (len(API_KEY) - 4) if len(API_KEY) > 4 else '***'}")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"\n  HTTP status : {resp.status_code}")
        data = resp.json()
        print(f"  Response    : {json.dumps(data, indent=2)[:500]}")
        if resp.status_code == 200:
            text = data["choices"][0]["message"]["content"]
            print(f"\n  ✅ Raw HTTP OK — model replied: {text!r}")
            return True
        else:
            print(f"\n  ❌ HTTP {resp.status_code}: {resp.text[:300]}")
            return False
    except Exception:
        print(f"\n  ❌ Connection failed:\n{traceback.format_exc()}")
        return False


# ── Test 2: LangChain CustomChatLLM ───────────────────────────────────────────
def test_langchain_chat_llm():
    section("Test 2 — LangChain CustomChatLLM (no tools)")
    from custom_llm.base_custom_llm import CustomChatLLM
    from langchain_core.messages import HumanMessage

    llm = CustomChatLLM(base_url=BASE_URL, model_name=MODEL, api_key=API_KEY)
    try:
        resp = llm.invoke([HumanMessage(content="Say 'hello' in one word.")])
        print(f"  ✅ LangChain ChatLLM OK")
        print(f"     content     : {resp.content!r}")
        print(f"     tool_calls  : {resp.tool_calls}")
        return True
    except Exception:
        print(f"  ❌ Failed:\n{traceback.format_exc()}")
        return False


# ── Test 3: LangChain CustomChatLLM with tools via kwargs ─────────────────────
def test_langchain_chat_llm_with_tools():
    section("Test 3 — LangChain CustomChatLLM with tools kwarg")
    from custom_llm.base_custom_llm import CustomChatLLM
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = CustomChatLLM(base_url=BASE_URL, model_name=MODEL, api_key=API_KEY)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude":  {"type": "number", "description": "Latitude"},
                        "longitude": {"type": "number", "description": "Longitude"},
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }
    ]

    messages = [
        SystemMessage(content="You are a weather assistant."),
        HumanMessage(content="What is the weather at lat 40.7, lon -74.0?"),
    ]

    try:
        resp = llm.invoke(messages, tools=tools)
        print(f"  ✅ Tool-kwargs call OK")
        print(f"     content    : {resp.content!r}")
        print(f"     tool_calls : {resp.tool_calls}")
        return True
    except NotImplementedError:
        print(f"  ⚠️  Model does not support tools kwarg (NotImplementedError)")
        print(f"     → Bridge will automatically fall back to prompt injection.")
        return True   # expected for many custom LLMs
    except TypeError as e:
        print(f"  ⚠️  TypeError on tools kwarg: {e}")
        print(f"     → Bridge will automatically fall back to prompt injection.")
        return True
    except Exception:
        print(f"  ❌ Unexpected error:\n{traceback.format_exc()}")
        return False


# ── Test 4: LangChain CustomChatLLM — prompt-injected tools ──────────────────
def test_langchain_prompt_injection():
    section("Test 4 — LangChain CustomChatLLM with prompt-injected tools")
    from custom_llm.base_custom_llm import CustomChatLLM
    from custom_llm.adk_langchain_bridge import _build_tool_system_suffix
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = CustomChatLLM(base_url=BASE_URL, model_name=MODEL, api_key=API_KEY)

    decls = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude":  {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        }
    ]

    system_text = "You are a weather assistant." + _build_tool_system_suffix(decls)
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content="What is the weather at lat 40.7, lon -74.0?"),
    ]

    try:
        resp = llm.invoke(messages)
        print(f"  ✅ Prompt-injection call OK")
        print(f"     content    : {resp.content!r}")
        # Check if model emitted a tool call JSON
        from custom_llm.adk_langchain_bridge import _parse_tool_call_from_text
        parsed = _parse_tool_call_from_text(resp.content or "")
        if parsed:
            print(f"     🔧 Tool call parsed: name={parsed[0]!r}, args={parsed[1]}")
        return True
    except Exception:
        print(f"  ❌ Failed:\n{traceback.format_exc()}")
        return False


# ── Test 5: YOUR custom LLM class ─────────────────────────────────────────────
def test_your_custom_llm():
    section("Test 5 — YOUR Custom LangChain LLM (edit path below)")
    print("  ℹ️  Edit this test to import YOUR LLM class and set its parameters.")
    print("  Example:")
    print("    from my_module import MyCustomLLM")
    print("    llm = MyCustomLLM(api_key=..., model=...)")
    print()

    # ── EDIT HERE ───────────────────────────────────────────────────────────
    # Uncomment and replace with your own LLM:
    #
    # from my_module import MyCustomLLM
    # from langchain_core.messages import HumanMessage
    # llm = MyCustomLLM(...)
    # try:
    #     resp = llm.invoke([HumanMessage(content="Say 'hello' in one word.")])
    #     print(f"  ✅ Your LLM replied: {resp!r}")
    # except Exception:
    #     print(f"  ❌ Failed:\n{traceback.format_exc()}")
    # ────────────────────────────────────────────────────────────────────────

    print("  (skipped — uncomment the block above to enable)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 62)
    print("  LangChain → ADK Bridge  |  Connection Diagnostics")
    print("═" * 62)
    print(f"\n  BASE_URL : {BASE_URL}")
    print(f"  MODEL    : {MODEL}")
    print(f"  API_KEY  : {API_KEY[:4]}{'*' * max(0, len(API_KEY) - 4)}")

    results = {}

    results["raw_http"]          = test_raw_http()
    results["langchain_no_tools"] = test_langchain_chat_llm()
    results["langchain_tools_kw"] = test_langchain_chat_llm_with_tools()
    results["prompt_injection"]  = test_langchain_prompt_injection()
    test_your_custom_llm()

    section("Summary")
    all_ok = True
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All tests passed! Your LLM setup looks good.")
    else:
        print("  Some tests failed. Fix the issues above, then re-run.")
        print()
        print("  Common fixes:")
        print("    • Wrong BASE_URL       → check CUSTOM_LLM_BASE_URL in .env")
        print("    • Wrong model name     → run `ollama list` or check your server")
        print("    • Auth error (401)     → set CUSTOM_LLM_API_KEY correctly")
        print("    • SSL error            → set ssl_verify=False in CustomChatLLM")
        print("    • Connection refused   → make sure your LLM server is running")
    print()


if __name__ == "__main__":
    main()
