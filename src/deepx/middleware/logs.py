from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import RunContextWrapper
from agents.tool import Tool
from agents.tool_context import ToolContext

from deepx.backends.protocol import BackendProtocol
from deepx.backends.utils import MAX_READ_FILE_LINES, data_root_as_agent_path
from deepx.context import AgentContext


def _safe_agent_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name) or "agent"


def _logs_dir_rel(session_id: str) -> str:
    return f"sessions/{session_id}/logs"


def run_log_save_plan(
    backend: BackendProtocol, session_id: str, agent_name: str, plan_json: str
) -> None:
    rel = f"{_logs_dir_rel(session_id)}/plans/{_safe_agent_name(agent_name)}.json"
    path = data_root_as_agent_path(rel)
    wr = backend.write(session_id, path, plan_json)
    if wr.error:
        raise OSError(wr.error)


def run_log_load_plan(
    backend: BackendProtocol, session_id: str, agent_name: str
) -> str | None:
    rel = f"{_logs_dir_rel(session_id)}/plans/{_safe_agent_name(agent_name)}.json"
    path = data_root_as_agent_path(rel)
    rr = backend.read(session_id, path, 0, MAX_READ_FILE_LINES)
    if rr.error:
        return None
    return rr.content


def run_log_append_plan_event(
    backend: BackendProtocol, session_id: str, entry_json: str
) -> None:
    rel = f"{_logs_dir_rel(session_id)}/plans/events.json"
    path = data_root_as_agent_path(rel)
    rr = backend.read(session_id, path, 0, MAX_READ_FILE_LINES)
    if rr.error or not rr.content:
        arr: list[Any] = []
    else:
        try:
            arr = json.loads(rr.content)
            if not isinstance(arr, list):
                arr = []
        except json.JSONDecodeError:
            arr = []
    arr.append(json.loads(entry_json))
    wr = backend.write(session_id, path, json.dumps(arr, indent=2))
    if wr.error:
        raise OSError(wr.error)


def run_log_write_tool(
    backend: BackendProtocol, session_id: str, log_data: dict
) -> None:
    tool_name = str(log_data["tool_name"])
    out = log_data.get("output", "")
    oc = log_data.get("output_chars", len(str(out)))
    entry = {
        "tool_name": tool_name,
        "agent_name": log_data.get("agent_name", ""),
        "session_id": session_id,
        "timestamp": log_data.get("timestamp", ""),
        "input": log_data.get("input", {}),
        "output_chars": oc,
        "output": out,
    }
    dir_rel = f"{_logs_dir_rel(session_id)}/tools/{tool_name}"
    dir_path = data_root_as_agent_path(dir_rel)
    gr = backend.glob(session_id, "*.json", path=dir_path)
    stems: list[int] = []
    if not gr.error:
        for fi in gr.files:
            name = fi.path.rstrip("/").rsplit("/", 1)[-1]
            if name.endswith(".json"):
                stem = name[:-5]
                if stem.isdigit():
                    stems.append(int(stem))
    next_id = max(stems, default=0) + 1
    rel_path = f"{dir_rel}/{next_id}.json"
    file_path = data_root_as_agent_path(rel_path)
    disk_entry = {**entry, "call_id": str(next_id)}
    wr = backend.write(
        session_id, file_path, json.dumps(disk_entry, indent=2, default=str)
    )
    if wr.error:
        raise OSError(wr.error)


def _tool_call_input_for_log(
    context: RunContextWrapper[AgentContext],
) -> dict[str, Any]:
    """Best-effort structured tool args for JSON logs (function tools use :class:`ToolContext`)."""
    if isinstance(context, ToolContext):
        raw = (context.tool_arguments or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw[:8000]}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    ti = getattr(context, "tool_input", None)
    if isinstance(ti, dict):
        return ti
    if ti is not None:
        return {"_repr": repr(ti)[:8000]}
    return {}


class SessionToolLogHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    """Append one JSON file per tool call under ``data_root/sessions/<id>/logs/tools/<tool>/``."""

    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent[AgentContext],
        tool: Tool,
        result: str,
    ) -> None:
        ac = context.context
        if not isinstance(ac, AgentContext):
            return
        name = getattr(tool, "name", None) or "unknown_tool"
        inp = _tool_call_input_for_log(context)
        run_log_write_tool(
            self._backend,
            ac.session_id,
            {
                "tool_name": name,
                "agent_name": agent.name,
                "session_id": ac.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": inp,
                "output_chars": len(str(result)),
                "output": str(result),
            },
        )


__all__ = [
    "SessionToolLogHooks",
    "run_log_append_plan_event",
    "run_log_load_plan",
    "run_log_save_plan",
    "run_log_write_tool",
]
