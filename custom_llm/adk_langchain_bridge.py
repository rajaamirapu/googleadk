"""
adk_langchain_bridge.py
───────────────────────
Bridges ANY LangChain LLM (LLM or BaseChatModel) into Google ADK's
BaseLlm interface so it can power an ADK Agent directly.

Supports:
  ✅ Plain LangChain LLM  (text completion — tool calls via system-prompt injection)
  ✅ LangChain BaseChatModel  (native bind_tools / tool_calls)
  ✅ Async (ainvoke) and sync (invoke) LLMs
  ✅ Streaming (yields a single final LlmResponse)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Optional

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ConfigDict

# LangChain message types
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helper: Google genai Schema → plain JSON-Schema dict
# ─────────────────────────────────────────────────────────────
_TYPE_MAP = {
    "TYPE_UNSPECIFIED": "object",
    "STRING": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
    "OBJECT": "object",
}


def _schema_to_dict(schema: Any) -> dict:
    """Convert a google.genai Schema (or dict) to a JSON Schema dict."""
    if schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(schema, dict):
        return schema

    result: dict = {}

    # Type
    raw_type = getattr(schema, "type", None)
    if raw_type is not None:
        type_str = raw_type.name if hasattr(raw_type, "name") else str(raw_type)
        result["type"] = _TYPE_MAP.get(type_str, "object").lower()
    else:
        result["type"] = "object"

    if desc := getattr(schema, "description", None):
        result["description"] = desc

    if props := getattr(schema, "properties", None):
        result["properties"] = {k: _schema_to_dict(v) for k, v in props.items()}

    if req := getattr(schema, "required", None):
        result["required"] = list(req)

    if items := getattr(schema, "items", None):
        result["items"] = _schema_to_dict(items)

    if enum := getattr(schema, "enum", None):
        result["enum"] = list(enum)

    return result


# ─────────────────────────────────────────────────────────────
# Helper: extract FunctionDeclaration list from LlmRequest
# ─────────────────────────────────────────────────────────────
def _extract_function_declarations(llm_request: LlmRequest) -> list[dict]:
    """Return a list of {name, description, parameters} dicts from the request."""
    decls: list[dict] = []
    if not (llm_request.config and llm_request.config.tools):
        return decls

    for tool in llm_request.config.tools:
        if isinstance(tool, types.Tool) and tool.function_declarations:
            for fd in tool.function_declarations:
                decls.append(
                    {
                        "name": fd.name,
                        "description": fd.description or "",
                        "parameters": _schema_to_dict(fd.parameters),
                    }
                )
    return decls


# ─────────────────────────────────────────────────────────────
# Helper: build tool-call injection text for plain LLMs
# ─────────────────────────────────────────────────────────────
def _build_tool_system_suffix(decls: list[dict]) -> str:
    """
    When the LLM has no native tool-calling support, describe the tools
    in the system prompt and ask the model to emit JSON function calls.
    """
    if not decls:
        return ""

    tools_json = json.dumps(decls, indent=2)
    return f"""

## Available Tools
You have access to the following tools. When you want to call a tool, respond
with a JSON block (and nothing else in that turn) using this exact format:

```json
{{
  "function_call": {{
    "name": "<tool_name>",
    "arguments": {{ "<param>": <value>, ... }}
  }}
}}
```

### Tool definitions
{tools_json}

After receiving the tool result, continue your answer normally.
"""


# ─────────────────────────────────────────────────────────────
# Helper: parse JSON tool call from plain-LLM text output
# ─────────────────────────────────────────────────────────────
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"function_call\"[^{}]*\})", re.DOTALL)


def _parse_tool_call_from_text(text: str) -> Optional[tuple[str, dict]]:
    """
    Try to extract a function_call JSON from plain text.
    Returns (name, args) or None.
    """
    # Try fenced code block first
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        m = _BARE_JSON_RE.search(text)
    if not m:
        return None

    try:
        obj = json.loads(m.group(1))
        fc = obj.get("function_call", {})
        name = fc.get("name")
        args = fc.get("arguments", fc.get("args", {}))
        if name:
            return name, args
    except json.JSONDecodeError:
        pass

    return None


# ─────────────────────────────────────────────────────────────
# Message converters: ADK ↔ LangChain
# ─────────────────────────────────────────────────────────────
def _adk_contents_to_langchain(
    contents: list[types.Content],
    system_instruction: str = "",
    tool_suffix: str = "",
) -> list[BaseMessage]:
    """Convert ADK Content list to a list of LangChain BaseMessages."""
    messages: list[BaseMessage] = []

    # System message (instruction + optional tool descriptions)
    full_system = (system_instruction or "").strip()
    if tool_suffix:
        full_system = full_system + tool_suffix
    if full_system:
        messages.append(SystemMessage(content=full_system))

    for content in contents:
        role = content.role
        parts = content.parts or []

        fn_responses = [p for p in parts if getattr(p, "function_response", None)]
        fn_calls = [p for p in parts if getattr(p, "function_call", None)]
        text_parts = [p for p in parts if getattr(p, "text", None)]

        if role == "user":
            if fn_responses:
                # Tool results → ToolMessage
                for p in fn_responses:
                    fr = p.function_response
                    raw = fr.response
                    content_str = (
                        json.dumps(raw) if isinstance(raw, dict) else str(raw)
                    )
                    messages.append(
                        ToolMessage(
                            content=content_str,
                            tool_call_id=fr.name,  # ADK uses tool name as ID
                            name=fr.name,
                        )
                    )
            else:
                text = "\n".join(p.text for p in text_parts if p.text)
                if text:
                    messages.append(HumanMessage(content=text))

        elif role == "model":
            text = "\n".join(p.text for p in text_parts if p.text)
            tool_calls = []
            for p in fn_calls:
                fc = p.function_call
                tool_calls.append(
                    {
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                        "id": fc.name,
                        "type": "tool_call",
                    }
                )
            messages.append(AIMessage(content=text, tool_calls=tool_calls))

    return messages


def _langchain_to_adk_response(lc_output: Any) -> LlmResponse:
    """
    Convert a LangChain response (AIMessage, str, or other) to LlmResponse.
    Handles both native tool_calls (BaseChatModel) and plain text (LLM).
    """
    parts: list[types.Part] = []

    # ── Case 1: AIMessage (from BaseChatModel) ─────────────────
    if isinstance(lc_output, AIMessage):
        # Text content
        raw_content = lc_output.content
        if isinstance(raw_content, str) and raw_content:
            # Check if the text itself encodes a tool call (plain LLM fallback)
            maybe_tc = _parse_tool_call_from_text(raw_content)
            if maybe_tc:
                name, args = maybe_tc
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=name, args=args)
                    )
                )
            else:
                parts.append(types.Part(text=raw_content))
        elif isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, str) and item:
                    parts.append(types.Part(text=item))
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(types.Part(text=item.get("text", "")))

        # Native tool_calls (BaseChatModel with bind_tools)
        if lc_output.tool_calls:
            for tc in lc_output.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tc["name"],
                            args=tc.get("args", {}),
                        )
                    )
                )

        # Legacy additional_kwargs function_call
        if not lc_output.tool_calls:
            fc_kw = (lc_output.additional_kwargs or {}).get("function_call")
            if fc_kw:
                try:
                    args = json.loads(fc_kw.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=fc_kw.get("name", ""), args=args
                        )
                    )
                )

    # ── Case 2: plain string (from LLM.invoke / LLM.ainvoke) ──
    elif isinstance(lc_output, str):
        maybe_tc = _parse_tool_call_from_text(lc_output)
        if maybe_tc:
            name, args = maybe_tc
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(name=name, args=args)
                )
            )
        else:
            parts.append(types.Part(text=lc_output))

    # ── Fallback ───────────────────────────────────────────────
    else:
        parts.append(types.Part(text=str(lc_output)))

    if not parts:
        parts = [types.Part(text="")]

    return LlmResponse(content=types.Content(role="model", parts=parts))


# ─────────────────────────────────────────────────────────────
# Main bridge class
# ─────────────────────────────────────────────────────────────
class LangChainADKBridge(BaseLlm):
    """
    Wraps any LangChain LLM or BaseChatModel as a Google ADK BaseLlm.

    Usage
    -----
    from custom_llm.adk_langchain_bridge import LangChainADKBridge
    from my_module import MyCustomLLM          # your LangChain LLM

    bridge = LangChainADKBridge(
        langchain_llm=MyCustomLLM(...),
        model="my-custom-llm",                # display name only
    )

    root_agent = Agent(model=bridge, ...)

    Notes
    -----
    - If your LLM is a BaseChatModel with bind_tools support, tool calls are
      handled natively.
    - If your LLM is a plain LLM (text completion), tool call instructions are
      injected into the system prompt and the output is parsed for JSON blocks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required by BaseLlm (Pydantic field)
    model: str = "langchain-custom-llm"

    # Your LangChain LLM instance — injected at construction time
    langchain_llm: Any

    # ── Optional knobs ──────────────────────────────────────────
    prefer_native_tools: bool = True
    """
    If True (default) and the LLM supports bind_tools, use native tool calling.
    Set to False to force system-prompt-based tool injection for all LLMs.
    """

    # ── Internal helpers ────────────────────────────────────────
    def _get_system_instruction(self, llm_request: LlmRequest) -> str:
        si = (llm_request.config or types.GenerateContentConfig()).system_instruction
        if isinstance(si, str):
            return si
        if isinstance(si, types.Content):
            return " ".join(p.text for p in (si.parts or []) if p.text)
        return ""

    def _is_chat_model(self) -> bool:
        """Return True if the wrapped LLM behaves like a BaseChatModel."""
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            return isinstance(self.langchain_llm, BaseChatModel)
        except ImportError:
            return False

    def _supports_bind_tools(self) -> bool:
        return self.prefer_native_tools and hasattr(self.langchain_llm, "bind_tools")

    async def _invoke_llm(self, messages: list[BaseMessage], lc_tools: list[dict]) -> Any:
        """Invoke the LangChain LLM, using bind_tools when available."""
        llm = self.langchain_llm
        if lc_tools and self._supports_bind_tools():
            llm = llm.bind_tools(lc_tools)

        if hasattr(llm, "ainvoke"):
            return await llm.ainvoke(messages)
        else:
            # Sync LLM — run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, llm.invoke, messages)

    # ── Main ADK entry point ────────────────────────────────────
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        """Translate between ADK and LangChain, then yield one LlmResponse."""

        # 1. Gather system instruction
        system_inst = self._get_system_instruction(llm_request)

        # 2. Extract tool declarations
        decls = _extract_function_declarations(llm_request)

        # 3. Decide tool strategy
        use_native = self._supports_bind_tools()
        tool_suffix = "" if use_native else _build_tool_system_suffix(decls)
        lc_tools = decls if use_native else []

        # 4. Convert contents → LangChain messages
        messages = _adk_contents_to_langchain(
            llm_request.contents, system_inst, tool_suffix
        )

        logger.debug(
            "[LangChainADKBridge] model=%s | msgs=%d | tools=%d | native=%s",
            self.model, len(messages), len(decls), use_native,
        )

        # 5. Invoke LLM
        try:
            lc_response = await self._invoke_llm(messages, lc_tools)
        except Exception as exc:
            logger.error("[LangChainADKBridge] LLM invocation failed: %s", exc)
            yield LlmResponse(
                error_code="LLM_INVOCATION_ERROR",
                error_message=str(exc),
            )
            return

        # 6. Convert response → ADK LlmResponse
        adk_response = _langchain_to_adk_response(lc_response)
        yield adk_response
