"""
adk_langchain_bridge.py
───────────────────────
Bridges ANY LangChain LLM (LLM or BaseChatModel) into Google ADK's
BaseLlm interface so it can power an ADK Agent directly.

Root-cause notes
────────────────
• BaseChatModel.bind_tools() raises NotImplementedError unless the subclass
  overrides it (ChatOpenAI does, a plain custom class usually does NOT).
• This bridge NEVER calls bind_tools() blindly. Instead it:
    1. Tries passing tools as kwargs to ainvoke/invoke   (native path)
    2. Falls back to system-prompt injection + JSON parsing  (prompt path)

Supported LLM styles
─────────────────────
  ✅ BaseChatModel with tools via kwargs  (native tool_calls in AIMessage)
  ✅ BaseChatModel without tool support   (falls back to prompt injection)
  ✅ Plain LLM (text completion)          (always uses prompt injection)
  ✅ Sync and Async LLMs
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import traceback as tb_module
from typing import Any, AsyncGenerator, Optional

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ConfigDict

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
    """Convert a google.genai Schema (or dict) to a plain JSON Schema dict."""
    if schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(schema, dict):
        return schema

    result: dict = {}

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
    """Return [{name, description, parameters}, ...] from the request tools."""
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
# Helper: build OpenAI-format tool list for kwargs passing
# ─────────────────────────────────────────────────────────────
def _to_openai_tools(decls: list[dict]) -> list[dict]:
    """Wrap function declarations in OpenAI tool schema format."""
    return [{"type": "function", "function": d} for d in decls]


# ─────────────────────────────────────────────────────────────
# Helper: build tool-call injection text for plain LLMs
# ─────────────────────────────────────────────────────────────
def _build_tool_system_suffix(decls: list[dict]) -> str:
    """
    Describe tools in the system prompt and instruct the model to emit
    a JSON function-call block when it wants to call a tool.
    """
    if not decls:
        return ""

    tools_json = json.dumps(decls, indent=2)
    return f"""

## Available Tools
You have access to the following tools. When you want to call a tool, respond
ONLY with a JSON block in the format below (nothing else in that turn):

```json
{{
  "function_call": {{
    "name": "<tool_name>",
    "arguments": {{ "<param>": <value> }}
  }}
}}
```

Tool definitions:
{tools_json}

After receiving the tool result in a follow-up message, continue your answer.
"""


# ─────────────────────────────────────────────────────────────
# Helper: parse JSON tool call from plain-LLM text output
# ─────────────────────────────────────────────────────────────
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_balanced_json(text: str, start: int) -> Optional[str]:
    """
    Extract a balanced JSON object starting at `start` index in `text`.
    Handles arbitrary nesting — regex alone can't do this correctly.
    """
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_tool_call_from_text(text: str) -> Optional[tuple[str, dict]]:
    """Extract (name, args) from a JSON function-call block in plain text."""
    # 1. Fenced code block  ```json { ... } ```
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidate = m.group(1)
    else:
        # 2. Bare JSON — find first { that contains "function_call"
        idx = text.find('{"function_call"')
        if idx == -1:
            idx = text.find('{ "function_call"')
        if idx == -1:
            return None
        candidate = _extract_balanced_json(text, idx)
        if not candidate:
            return None

    try:
        obj  = json.loads(candidate)
        fc   = obj.get("function_call", {})
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
    """Convert an ADK Content list + system instruction to LangChain messages."""
    messages: list[BaseMessage] = []

    full_system = (system_instruction or "").strip()
    if tool_suffix:
        full_system = full_system + tool_suffix
    if full_system:
        messages.append(SystemMessage(content=full_system))

    for content in contents:
        role  = content.role
        parts = content.parts or []

        fn_responses = [p for p in parts if getattr(p, "function_response", None)]
        fn_calls     = [p for p in parts if getattr(p, "function_call",     None)]
        text_parts   = [p for p in parts if getattr(p, "text",              None)]

        if role == "user":
            if fn_responses:
                for p in fn_responses:
                    fr  = p.function_response
                    raw = fr.response
                    content_str = json.dumps(raw) if isinstance(raw, dict) else str(raw)
                    messages.append(
                        ToolMessage(content=content_str, tool_call_id=fr.name, name=fr.name)
                    )
            else:
                text = "\n".join(p.text for p in text_parts if p.text)
                if text:
                    messages.append(HumanMessage(content=text))

        elif role == "model":
            text       = "\n".join(p.text for p in text_parts if p.text)
            tool_calls = [
                {
                    "name": p.function_call.name,
                    "args": dict(p.function_call.args) if p.function_call.args else {},
                    "id":   p.function_call.name,
                    "type": "tool_call",
                }
                for p in fn_calls
            ]
            messages.append(AIMessage(content=text, tool_calls=tool_calls))

    return messages


def _langchain_to_adk_response(lc_output: Any) -> LlmResponse:
    """Convert a LangChain response (AIMessage / str / other) to LlmResponse."""
    parts: list[types.Part] = []

    if isinstance(lc_output, AIMessage):
        raw = lc_output.content

        # ── text ────────────────────────────────────────────────
        if isinstance(raw, str) and raw:
            maybe_tc = _parse_tool_call_from_text(raw)
            if maybe_tc:
                name, args = maybe_tc
                parts.append(
                    types.Part(function_call=types.FunctionCall(name=name, args=args))
                )
            else:
                parts.append(types.Part(text=raw))
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item:
                    parts.append(types.Part(text=item))
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(types.Part(text=item.get("text", "")))

        # ── native tool_calls ────────────────────────────────────
        for tc in (lc_output.tool_calls or []):
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=tc["name"], args=tc.get("args", {})
                    )
                )
            )

        # ── legacy additional_kwargs function_call ───────────────
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

    elif isinstance(lc_output, str):
        maybe_tc = _parse_tool_call_from_text(lc_output)
        if maybe_tc:
            name, args = maybe_tc
            parts.append(types.Part(function_call=types.FunctionCall(name=name, args=args)))
        else:
            parts.append(types.Part(text=lc_output))

    else:
        parts.append(types.Part(text=str(lc_output)))

    if not parts:
        parts = [types.Part(text="")]

    return LlmResponse(content=types.Content(role="model", parts=parts))


# ─────────────────────────────────────────────────────────────
# Low-level async/sync LLM caller (no bind_tools)
# ─────────────────────────────────────────────────────────────
async def _call_llm(llm: Any, messages: list[BaseMessage], **invoke_kwargs) -> Any:
    """
    Call llm.ainvoke or llm.invoke (sync wrapped in executor).
    Extra kwargs (e.g. tools=[...]) are forwarded to the LLM.
    """
    if hasattr(llm, "ainvoke"):
        return await llm.ainvoke(messages, **invoke_kwargs)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: llm.invoke(messages, **invoke_kwargs)
        )


# ─────────────────────────────────────────────────────────────
# Main bridge class
# ─────────────────────────────────────────────────────────────
class LangChainADKBridge(BaseLlm):
    """
    Wraps any LangChain LLM or BaseChatModel as a Google ADK BaseLlm.

    Key design decisions
    ────────────────────
    • We NEVER call bind_tools() — it raises NotImplementedError on most custom LLMs.
    • Tool calling strategy (tried in order):
        1. Pass tools as ``tools=[...]`` kwarg directly to ainvoke/invoke.
           Your ``_generate`` method receives them in **kwargs.
           If this raises an exception, we fall back automatically.
        2. Inject tool descriptions into the system prompt and parse JSON
           function-call blocks from the text output.
    • Full Python tracebacks are logged (and included in LlmResponse.error_message)
      so you can see exactly what went wrong.

    Quick usage
    ───────────
        from custom_llm.adk_langchain_bridge import LangChainADKBridge
        from my_module import MyCustomLLM

        bridge = LangChainADKBridge(
            langchain_llm=MyCustomLLM(...),
            model="my-custom-llm",
        )
        root_agent = Agent(model=bridge, ...)

    Configuration
    ─────────────
    prefer_native_tools : bool (default True)
        Try passing tools via kwargs before falling back to prompt injection.
        Set to False to always use prompt injection (useful if your LLM does
        not read extra kwargs or always errors on the tools param).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "langchain-custom-llm"
    langchain_llm: Any

    prefer_native_tools: bool = True
    """
    Try native kwargs-based tool passing first.
    Flip to False if your LLM ignores or errors on extra kwargs.
    """

    # ── Helpers ─────────────────────────────────────────────────
    def _get_system_instruction(self, llm_request: LlmRequest) -> str:
        si = (llm_request.config or types.GenerateContentConfig()).system_instruction
        if isinstance(si, str):
            return si
        if isinstance(si, types.Content):
            return " ".join(p.text for p in (si.parts or []) if p.text)
        return ""

    # ── Main ADK entry point ─────────────────────────────────────
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:

        system_inst = self._get_system_instruction(llm_request)
        decls       = _extract_function_declarations(llm_request)

        logger.debug(
            "[LangChainADKBridge] model=%s | content_turns=%d | tools=%d",
            self.model, len(llm_request.contents), len(decls),
        )

        # ── Strategy 1: native kwargs tool passing ───────────────
        if decls and self.prefer_native_tools:
            messages    = _adk_contents_to_langchain(llm_request.contents, system_inst)
            openai_tools = _to_openai_tools(decls)
            try:
                lc_resp = await _call_llm(self.langchain_llm, messages, tools=openai_tools)
                yield _langchain_to_adk_response(lc_resp)
                return
            except (NotImplementedError, TypeError) as exc:
                logger.warning(
                    "[LangChainADKBridge] Native tools kwarg not supported (%s). "
                    "Falling back to prompt injection.", exc
                )
            except Exception as exc:
                logger.warning(
                    "[LangChainADKBridge] Native tool calling raised %s: %s. "
                    "Falling back to prompt injection.\n%s",
                    type(exc).__name__, exc, tb_module.format_exc(),
                )

        # ── Strategy 2: system-prompt tool injection ─────────────
        tool_suffix = _build_tool_system_suffix(decls) if decls else ""
        messages    = _adk_contents_to_langchain(llm_request.contents, system_inst, tool_suffix)
        try:
            lc_resp = await _call_llm(self.langchain_llm, messages)
            yield _langchain_to_adk_response(lc_resp)
        except Exception as exc:
            full_tb = tb_module.format_exc()
            logger.error(
                "[LangChainADKBridge] LLM invocation failed:\n%s", full_tb
            )
            # Surface the real error to the ADK caller
            yield LlmResponse(
                error_code="LLM_INVOCATION_ERROR",
                error_message=(
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"--- Full traceback ---\n{full_tb}"
                ),
            )
