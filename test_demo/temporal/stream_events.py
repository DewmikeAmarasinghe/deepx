from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)


def _trunc(s: str, n: int = 8000) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "…"


def append_ndjson(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def serialize_stream_event(ev: Any) -> dict[str, Any]:
    if isinstance(ev, RunItemStreamEvent):
        out: dict[str, Any] = {"kind": "run_item", "name": ev.name}
        item = ev.item
        ri = getattr(item, "raw_item", None)
        if ri is not None:
            name = getattr(ri, "name", None)
            if name is None and isinstance(ri, dict):
                name = ri.get("name")
            if name:
                out["tool_name"] = name
            args = getattr(ri, "arguments", None)
            if args is None and isinstance(ri, dict):
                args = ri.get("arguments")
            if isinstance(args, str) and args:
                out["tool_args"] = _trunc(args)
        out["item_type"] = getattr(item, "type", type(item).__name__)
        if ev.name == "tool_output":
            outv = getattr(item, "output", None)
            if outv is not None:
                out["output_preview"] = _trunc(str(outv), 4000)
        return out
    if isinstance(ev, RawResponsesStreamEvent):
        data = ev.data
        typ = getattr(data, "type", None) or (
            data.get("type") if isinstance(data, dict) else ""
        )
        row: dict[str, Any] = {"kind": "raw", "type": typ}
        delta = getattr(data, "delta", None)
        if isinstance(delta, str) and delta:
            row["delta"] = _trunc(delta, 2000)
        return row
    if isinstance(ev, AgentUpdatedStreamEvent):
        nm = getattr(ev.new_agent, "name", "")
        return {"kind": "agent", "name": nm}
    return {"kind": "other", "repr": _trunc(repr(ev), 500)}


async def tail_ndjson_events(
    path: Path,
    *,
    poll_interval: float = 0.2,
    wait_for_exists_s: float = 90.0,
    result_task: asyncio.Task | None = None,
) -> AsyncIterator[dict[str, Any]]:
    deadline = time.monotonic() + wait_for_exists_s
    while not path.exists():
        if result_task is not None and result_task.done():
            return
        if time.monotonic() > deadline:
            return
        await asyncio.sleep(poll_interval)
    consumed = 0
    while True:
        if result_task is not None and result_task.done():
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return
            lines = raw.splitlines()
            while consumed < len(lines):
                line = lines[consumed]
                consumed += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
            return
        await asyncio.sleep(poll_interval)
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = raw.splitlines()
        while consumed < len(lines):
            line = lines[consumed]
            consumed += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
                if obj.get("kind") == "done":
                    return


async def poll_workflow_stream_events(
    handle: Any,
    *,
    workflow_cls: type,
    poll_interval: float = 0.2,
    result_task: asyncio.Task | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Tail stream rows via a workflow query (``get_stream_events`` on ``workflow_cls``)."""
    seen = 0
    q = workflow_cls.get_stream_events

    while True:
        rows: list[Any] = await handle.query(q)
        while seen < len(rows):
            rec = rows[seen]
            seen += 1
            if isinstance(rec, dict):
                yield rec
                if rec.get("kind") == "done":
                    if result_task is not None:
                        await result_task
                    return
        if result_task is not None and result_task.done():
            rows = await handle.query(q)
            while seen < len(rows):
                rec = rows[seen]
                seen += 1
                if isinstance(rec, dict):
                    yield rec
            return
        await asyncio.sleep(poll_interval)
