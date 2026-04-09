"""
base_custom_llm.py
──────────────────
Example custom LangChain LLMs. Replace the _generate / _call body with
your own backend. The shell (config, error handling, SSL, kwargs routing)
is already wired up.

Two patterns
────────────
1. CustomChatLLM  (BaseChatModel) — RECOMMENDED
   • invoke() returns AIMessage
   • Tools are received via _generate(messages, **kwargs) → kwargs["tools"]
   • Works natively with LangChainADKBridge (native kwargs path)

2. CustomTextLLM  (LLM) — plain text completion
   • invoke() returns str
   • Tools are injected into the system prompt by the bridge (prompt path)
   • Simpler but the model must emit JSON tool-call blocks

IMPORTANT: To plug in YOUR OWN LLM, edit agent.py — not this file.
           This file is just a working reference implementation.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import requests

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


# ═══════════════════════════════════════════════════════════════════════════
# Pattern 1 — BaseChatModel  (RECOMMENDED)
# ═══════════════════════════════════════════════════════════════════════════
class CustomChatLLM(BaseChatModel):
    """
    Custom chat LLM calling any OpenAI-compatible /v1/chat/completions endpoint.

    Quick setup
    ───────────
        llm = CustomChatLLM(
            base_url="http://localhost:11434/v1",   # Ollama, vLLM, LM Studio …
            model_name="llama3.2",
            api_key="custom",                       # any string for keyless servers
        )

    Tool calling
    ────────────
    When the ADK bridge calls llm.invoke(messages, tools=[...]), the
    ``tools`` kwarg lands in _generate(**kwargs). It is forwarded to the
    OpenAI-format request automatically. If your server does not support
    function calling, the bridge will catch the error and fall back to
    prompt injection — no action needed on your part.
    """

    base_url: str    = Field(default="http://localhost:11434/v1")
    model_name: str  = Field(default="llama3.2")
    api_key: str     = Field(default="custom")
    temperature: float = Field(default=0.3)
    max_tokens: int  = Field(default=2048)
    timeout: int     = Field(default=120)
    ssl_verify: bool = Field(
        default=True,
        description="Set to False to skip TLS verification (self-signed certs).",
    )

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
        if isinstance(msg, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }
        if isinstance(msg, AIMessage):
            d: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id":       tc.get("id", tc["name"]),
                        "type":     "function",
                        "function": {
                            "name":      tc["name"],
                            "arguments": json.dumps(tc.get("args", {})),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return d
        return {"role": "user", "content": str(msg.content)}

    # ── Core generation ────────────────────────────────────────────────────
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """
        ─────────────────────────────────────────────────────────────
        REPLACE THIS BODY with your own model invocation if needed.
        Most OpenAI-compatible servers will work without any changes.
        ─────────────────────────────────────────────────────────────
        """
        payload: dict[str, Any] = {
            "model":       self.model_name,
            "messages":    [self._msg_to_openai(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }
        if stop:
            payload["stop"] = stop

        # ── Forward tool schemas when provided (OpenAI format) ────
        # The bridge passes these as: llm.invoke(messages, tools=[...])
        # which flows into _generate via **kwargs
        tools = kwargs.get("tools")
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.ssl_verify,
            )
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                f"SSL certificate error connecting to {self.base_url}. "
                f"Set ssl_verify=False in CustomChatLLM to skip verification.\n"
                f"Original error: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to {self.base_url}. "
                f"Is your LLM server running?\nOriginal error: {exc}"
            ) from exc
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Request to {self.base_url} timed out after {self.timeout}s. "
                f"Increase the timeout= parameter."
            )

        if not resp.ok:
            raise RuntimeError(
                f"LLM server returned HTTP {resp.status_code}.\n"
                f"URL    : {self.base_url}/chat/completions\n"
                f"Body   : {resp.text[:600]}"
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse JSON from LLM response: {resp.text[:400]}"
            ) from exc

        choice = data.get("choices", [{}])[0].get("message", {})
        text   = choice.get("content") or ""

        # ── Parse tool_calls (OpenAI format) ───────────────────────
        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "name": fn.get("name", ""),
                "args": args,
                "id":   tc.get("id", fn.get("name", "")),
                "type": "tool_call",
            })

        ai_msg = AIMessage(content=text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "base_url": self.base_url}


# ═══════════════════════════════════════════════════════════════════════════
# Pattern 2 — Plain LLM  (text completion)
# ═══════════════════════════════════════════════════════════════════════════
class CustomTextLLM(LLM):
    """
    Custom text-completion LLM (e.g. Ollama /api/generate endpoint).

    Tool calling is handled via system-prompt injection by LangChainADKBridge.
    The model must output JSON blocks as described in the injected instructions.
    """

    base_url: str   = Field(default="http://localhost:11434")
    model_name: str = Field(default="llama3.2")
    temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=2048)
    timeout: int    = Field(default=120)
    ssl_verify: bool = Field(default=True)

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
        payload: dict[str, Any] = {
            "model":  self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                verify=self.ssl_verify,
            )
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                f"SSL error. Set ssl_verify=False in CustomTextLLM.\n{exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to {self.base_url}. Is the server running?\n{exc}"
            ) from exc

        if not resp.ok:
            raise RuntimeError(
                f"HTTP {resp.status_code} from {self.base_url}: {resp.text[:400]}"
            )

        return resp.json().get("response", "")

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "base_url": self.base_url}
