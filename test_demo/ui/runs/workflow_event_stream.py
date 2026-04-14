"""Chainlit helpers: map SDK stream events and poll Temporal workflow hook logs."""

from __future__ import annotations

import asyncio
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


async def poll_workflow_stream_events(
    handle: Any,
    *,
    workflow_cls: type,
    poll_interval: float = 0.2,
    result_task: asyncio.Task | None = None,
):
    """Yield rows from ``workflow_cls.get_stream_events`` until a ``kind=done`` record."""
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
