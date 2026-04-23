from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import RunContextWrapper
from agents.tool import Tool
from agents.tool_context import ToolContext

from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext

# In-memory fallback when no FilesystemBackend data_root (e.g. InMemoryBackend tests).
_ephemeral_plans: dict[tuple[str, str], str] = {}
_ephemeral_plan_events: dict[str, list[dict]] = {}
_ephemeral_tool_rows: dict[str, list[dict]] = {}


def _safe_agent_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name) or "agent"


def resolve_data_root(backend: BackendProtocol) -> Path | None:
    """Return `.deepx` data root for on-disk run logs, or None if not file-backed."""
    from deepx.backends.filesystem import FilesystemBackend

    if isinstance(backend, FilesystemBackend):
        return backend.data_root
    return None


def _logs_dir(data_root: Path, session_id: str) -> Path:
    return data_root / "sessions" / session_id / "logs"


def run_log_save_plan(
    backend: BackendProtocol, session_id: str, agent_name: str, plan_json: str
) -> None:
    dr = resolve_data_root(backend)
    if dr is None:
        _ephemeral_plans[(session_id, agent_name)] = plan_json
        return
    p = _logs_dir(dr, session_id) / "plans" / f"{_safe_agent_name(agent_name)}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(plan_json, encoding="utf-8")


def run_log_load_plan(
    backend: BackendProtocol, session_id: str, agent_name: str
) -> str | None:
    dr = resolve_data_root(backend)
    if dr is None:
        return _ephemeral_plans.get((session_id, agent_name))
    p = _logs_dir(dr, session_id) / "plans" / f"{_safe_agent_name(agent_name)}.json"
    return p.read_text(encoding="utf-8") if p.is_file() else None


def run_log_append_plan_event(
    backend: BackendProtocol, session_id: str, entry_json: str
) -> None:
    dr = resolve_data_root(backend)
    if dr is None:
        obj = json.loads(entry_json)
        _ephemeral_plan_events.setdefault(session_id, []).append(obj)
        return
    p = _logs_dir(dr, session_id) / "plans" / "events.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        arr = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(arr, list):
            arr = []
    else:
        arr = []
    arr.append(json.loads(entry_json))
    p.write_text(json.dumps(arr, indent=2), encoding="utf-8")


def run_log_write_tool(
    backend: BackendProtocol, session_id: str, log_data: dict
) -> None:
    dr = resolve_data_root(backend)
    tool_name = str(log_data["tool_name"])
    inp = log_data.get("input", {})
    out = log_data.get("output", "")
    oc = log_data.get("output_chars", len(str(out)))
    entry = {
        "tool_name": tool_name,
        "agent_name": log_data.get("agent_name", ""),
        "session_id": session_id,
        "timestamp": log_data.get("timestamp", ""),
        "input": inp,
        "output_chars": oc,
        "output": out,
    }
    if dr is None:
        logs = _ephemeral_tool_rows.setdefault(session_id, [])
        n = 1 + sum(1 for e in logs if str(e.get("tool_name")) == tool_name)
        logs.append({**entry, "call_id": str(n)})
        return
    dir_path = _logs_dir(dr, session_id) / "tools" / tool_name
    dir_path.mkdir(parents=True, exist_ok=True)
    existing = [int(x.stem) for x in dir_path.glob("*.json") if x.stem.isdigit()]
    next_id = max(existing, default=0) + 1
    disk_entry = {**entry, "call_id": str(next_id)}
    (dir_path / f"{next_id}.json").write_text(
        json.dumps(disk_entry, indent=2), encoding="utf-8"
    )


class SessionToolLogHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    """Append one JSON file per tool call under ``.deepx/sessions/<id>/logs/tools/<tool>/``.

    Covers static tools and MCP-backed tools (the latter are merged at runtime and never pass
    through per-tool logging wrappers).
    """

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
        inp: dict[str, Any] = {}
        if isinstance(context, ToolContext):
            raw = (context.tool_arguments or "").strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                    inp = parsed if isinstance(parsed, dict) else {"_value": parsed}
                except json.JSONDecodeError:
                    inp = {"_raw": raw[:8000]}
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
    "resolve_data_root",
    "run_log_append_plan_event",
    "run_log_load_plan",
    "run_log_save_plan",
    "run_log_write_tool",
]
