"""
base_custom_llm.py
──────────────────
Example custom LangChain LLMs that you can replace with your own implementation.

Two patterns are shown:

  1. CustomChatLLM  (extends BaseChatModel) — RECOMMENDED
     ✅ Native tool/function-calling support via bind_tools
     ✅ Returns AIMessage with tool_calls
     ✅ Works best with LangChainADKBridge

  2. CustomTextLLM  (extends LLM) — simpler but limited
     ⚠️ Text completion only — tool calls injected via system prompt
     ⚠️ Requires the model to output JSON function-call blocks
     ✅ Works with LangChainADKBridge via system-prompt injection

Replace the _call / _generate bodies with your own LLM backend logic.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, List, Optional

import requests

# ── LangChain base classes ──────────────────────────────────────────────────
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from pydantic import Field


# ═══════════════════════════════════════════════════════════════════════════
# Pattern 1 — BaseChatModel (RECOMMENDED for tool calling)
# ═══════════════════════════════════════════════════════════════════════════
class CustomChatLLM(BaseChatModel):
    """
    A custom chat LLM that calls an OpenAI-compatible HTTP endpoint.

    Swap out _generate with your own backend (local model, proprietary API, etc.).

    Example (Ollama):
        llm = CustomChatLLM(
            base_url="http://localhost:11434/v1",
            model_name="llama3.2",
            api_key="ollama",
        )

    Example (any OpenAI-compatible server):
        llm = CustomChatLLM(
            base_url="http://my-server:8000/v1",
            model_name="my-model",
            api_key="secret",
        )
    """

    base_url: str = Field(default="http://localhost:11434/v1")
    model_name: str = Field(default="llama3.2")
    api_key: str = Field(default="custom")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    timeout: int = Field(default=120)

    @property
    def _llm_type(self) -> str:
        return "custom-chat-llm"

    # ── Message serialiser ─────────────────────────────────────────────────
    @staticmethod
    def _msg_to_openai(msg: BaseMessage) -> dict:
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        if isinstance(msg, HumanMessage):
            return {"role": "user", "content": msg.content}
        if isinstance(msg, AIMessage):
            d: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.get("id", tc["name"]),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args", {})),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return d
        # ToolMessage / generic
        return {"role": "user", "content": str(msg.content)}

    # ── Core generation (sync) ─────────────────────────────────────────────
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """
        ─────────────────────────────────────────────────────────────
        REPLACE THIS BODY with your own model invocation.
        ─────────────────────────────────────────────────────────────
        """
        payload: dict = {
            "model": self.model_name,
            "messages": [self._msg_to_openai(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Forward tool schemas if bound (passed via kwargs by bind_tools)
        if tools := kwargs.get("tools"):
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]
        text = choice.get("content") or ""

        # Parse tool calls (OpenAI format)
        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {
                    "name": fn.get("name", ""),
                    "args": args,
                    "id": tc.get("id", fn.get("name", "")),
                    "type": "tool_call",
                }
            )

        ai_msg = AIMessage(content=text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "base_url": self.base_url}


# ═══════════════════════════════════════════════════════════════════════════
# Pattern 2 — Plain LLM (text completion style)
# ═══════════════════════════════════════════════════════════════════════════
class CustomTextLLM(LLM):
    """
    A custom text-completion LLM.

    Tool calling is handled via system-prompt injection by LangChainADKBridge.
    The model must output JSON blocks as instructed (see bridge.py).

    Example (Ollama generate endpoint):
        llm = CustomTextLLM(
            base_url="http://localhost:11434",
            model_name="llama3.2",
        )
    """

    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    timeout: int = Field(default=120)

    @property
    def _llm_type(self) -> str:
        return "custom-text-llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        """
        ─────────────────────────────────────────────────────────────
        REPLACE THIS BODY with your own text generation logic.
        ─────────────────────────────────────────────────────────────
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "base_url": self.base_url}
