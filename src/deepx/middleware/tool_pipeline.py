"""Tool wrapping pipeline: large-result eviction, optional logging, HITL — applies to all FunctionTools."""

from __future__ import annotations

import dataclasses
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from agents.tool import FunctionTool, Tool

from deepx.backends.filesystem import (
    OUTPUTS_LARGE_TOOL_RESULTS_PREFIX,
    TOO_LARGE_TOOL_MSG,
    TOOL_RESULT_TOKEN_LIMIT,
    TOOLS_EXCLUDED_FROM_EVICTION,
    create_large_tool_result_preview,
    tool_result_char_budget,
)
from deepx.backends.protocol import BackendProtocol
from deepx.middleware.hitl import wrap_tools_for_hitl


def _tool_call_id(ctx: Any) -> str:
    tid = getattr(ctx, "tool_call_id", None)
    if isinstance(tid, str) and tid.strip():
        return tid.strip()
    return "unknown_tool_call"


def _slug_hint(s: str, max_len: int = 24) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip())[:max_len].strip("_")
    return slug or ""


def _hints_from_args_json(args_json: str) -> str:
    raw = (args_json or "").strip()
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return _slug_hint(raw[:96])
    if not isinstance(obj, dict):
        return ""
    parts: list[str] = []
    cmd = obj.get("command")
    if isinstance(cmd, str) and cmd.strip():
        parts.append(_slug_hint(cmd.strip(), 28))
    for key in ("url", "urls", "query", "pattern", "path", "db_name", "queries"):
        v = obj.get(key)
        if v is None:
            continue
        if key == "urls" and isinstance(v, list) and v:
            v = v[0]
        if key == "queries" and isinstance(v, list) and v:
            v = v[0]
        if not isinstance(v, str) or not v.strip():
            continue
        s = v.strip()
        if key in ("url", "urls") and "://" in s:
            host = (urlparse(s).netloc or s).replace("www.", "")
            parts.append(_slug_hint(host, 20))
        else:
            parts.append(_slug_hint(s))
        if len(parts) >= 2:
            break
    joined = "_".join(parts)
    return joined[:32] if joined else ""


def _readable_large_tool_agent_path(
    tool_name: str, _tool_call_id: str, args_json: str = ""
) -> str:
    base = (
        re.sub(r"[^a-zA-Z0-9_-]+", "_", (tool_name or "tool").strip()).strip("_")[:40]
        or "tool"
    )
    hint = _hints_from_args_json(args_json)
    hint_part = f"_{hint}" if hint else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{OUTPUTS_LARGE_TOOL_RESULTS_PREFIX}/{base}{hint_part}_{stamp}_{suffix}.txt"


def _make_large_tool_results_invoke(
    original_invoke: Any,
    backend: BackendProtocol,
    *,
    tool_name: str,
    token_limit: int | None = TOOL_RESULT_TOKEN_LIMIT,
) -> Any:
    """Evict oversized tool returns under ``/_outputs/large_tool_results/``."""

    budget = tool_result_char_budget(token_limit=token_limit)

    async def invoke(ctx: Any, args_json: str) -> Any:
        result = await original_invoke(ctx, args_json)
        if budget is None or tool_name in TOOLS_EXCLUDED_FROM_EVICTION:
            return result
        text = str(result)
        if len(text) <= budget:
            return result

        call_id = _tool_call_id(ctx)
        agent_path = _readable_large_tool_agent_path(tool_name, call_id, args_json)
        session_id = ctx.context.session_id
        wr = backend.write(session_id, agent_path, text)
        if wr.error:
            return result

        preview = create_large_tool_result_preview(text)
        return TOO_LARGE_TOOL_MSG.format(
            tool_call_id=call_id,
            file_path=agent_path,
            content_sample=preview,
        )

    return invoke


def wrap_tools_for_large_tool_results(
    tools: list[Tool],
    backend: BackendProtocol,
    *,
    token_limit: int | None = TOOL_RESULT_TOKEN_LIMIT,
) -> list[Tool]:
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            inv = _make_large_tool_results_invoke(
                tool.on_invoke_tool,
                backend,
                tool_name=tool.name,
                token_limit=token_limit,
            )
            out.append(dataclasses.replace(tool, on_invoke_tool=inv))
        else:
            out.append(tool)
    return out


def apply_tool_pipeline(
    tools: list[Tool],
    backend: BackendProtocol,
    *,
    interrupt_on: frozenset[str] | None = None,
) -> list[Tool]:
    """Large-result eviction + ``interrupt_on`` HITL for static ``agent.tools``.

    Session JSON logs for **all** tools (including MCP) come from :class:`SessionToolLogHooks` when
    ``debug=True`` on the runner.
    """
    wrapped = wrap_tools_for_large_tool_results(tools, backend)
    wrapped = wrap_tools_for_hitl(wrapped, interrupt_on or frozenset())
    return wrapped


__all__ = [
    "apply_tool_pipeline",
    "wrap_tools_for_large_tool_results",
]
