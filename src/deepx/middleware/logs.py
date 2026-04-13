from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from typing import Any

from agents.tool import FunctionTool, Tool

from deepx.backends.protocol import BackendProtocol


def _make_logged_invoke(
    original_invoke: Any,
    tool_name: str,
    agent_name: str,
    backend: BackendProtocol,
) -> Any:
    async def logged_invoke(ctx: Any, args_json: str) -> Any:
        result = await original_invoke(ctx, args_json)
        session_id = ctx.context.session_id
        backend.save_tool_log(
            session_id,
            {
                "tool_name": tool_name,
                "agent_name": agent_name,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": json.loads(args_json) if args_json else {},
                "output": str(result),
                "output_chars": len(str(result)),
            },
        )
        return result

    return logged_invoke


def wrap_tools_for_logging(
    tools: list[Tool],
    backend: BackendProtocol,
    agent_name: str,
) -> list[Tool]:
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            logged = _make_logged_invoke(
                tool.on_invoke_tool, tool.name, agent_name, backend
            )
            out.append(dataclasses.replace(tool, on_invoke_tool=logged))
        else:
            out.append(tool)
    return out
