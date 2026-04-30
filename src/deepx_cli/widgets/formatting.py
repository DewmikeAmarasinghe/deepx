from __future__ import annotations

import json
from typing import Any

from agents.items import ToolCallItem


def agent_label(agent: Any) -> str:
    if agent is None:
        return "agent"
    name = getattr(agent, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return type(agent).__name__


def tool_name_and_arguments(item: ToolCallItem) -> tuple[str, Any]:
    ri = item.raw_item
    if isinstance(ri, dict):
        tname = str(ri.get("name") or ri.get("call_id") or "?")
        raw_args: Any = ri.get("arguments", "{}")
    elif hasattr(ri, "model_dump"):
        d = ri.model_dump(mode="python", exclude_unset=True)
        tname = str(d.get("name", getattr(ri, "name", "?")))
        raw_args = d.get("arguments", getattr(ri, "arguments", "{}"))
    else:
        tname = str(getattr(ri, "name", None) or "?")
        raw_args = getattr(ri, "arguments", "{}")

    if isinstance(raw_args, str):
        s = raw_args.strip()
        if not s:
            return tname, {}
        try:
            return tname, json.loads(raw_args)
        except json.JSONDecodeError:
            return tname, {"_raw_arguments": raw_args}
    if raw_args is None:
        return tname, {}
    if isinstance(raw_args, dict):
        return tname, raw_args
    return tname, {"_value": raw_args}
