"""Temporal vs local agent runs and stream rendering for Chainlit."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import chainlit as cl
from chainlit.context import context
from chainlit.step import Step

from test_demo.orchestrator import DEMO_BACKEND
from test_demo.temporal.workflows import DeepxOrchestratorWorkflow
from test_demo.ui.runs.session_prefs import default_use_temporal
from test_demo.ui.runs.workflow_event_stream import (
    poll_workflow_stream_events,
    serialize_stream_event,
)


def _deepx_session_id() -> str:
    return str(context.session.thread_id)


def _tool_logs_by_agent(session_id: str) -> dict[str, list[dict]]:
    base = DEMO_BACKEND.data_root / "sessions" / session_id / "logs" / "tools"
    by_agent: dict[str, list[dict]] = {}
    if not base.is_dir():
        return by_agent
    for tool_dir in sorted(base.iterdir()):
        if not tool_dir.is_dir():
            continue
        for jf in sorted(tool_dir.glob("*.json"), key=lambda p: p.name):
            try:
                obj = json.loads(jf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            an = str(obj.get("agent_name", "")) or "orchestrator"
            by_agent.setdefault(an, []).append(obj)
    for rows in by_agent.values():
        rows.sort(key=lambda r: str(r.get("timestamp", "")))
    return by_agent


async def _emit_render_files_ui(session_id: str, tool_args_json: str) -> None:
    try:
        args = json.loads(tool_args_json) if tool_args_json.strip() else {}
    except json.JSONDecodeError:
        return
    paths = args.get("paths") if isinstance(args, dict) else None
    if not isinstance(paths, list):
        return
    chunks: list[str] = []
    for path in paths:
        p = str(path or "").strip()
        if not p:
            continue
        rr = DEMO_BACKEND.read(session_id, p, 0, 800)
        if rr.error:
            chunks.append(f"## `{p}`\n\n_{rr.error}_")
        else:
            body = (rr.content or "")[:24_000]
            chunks.append(f"## `{p}`\n\n{body}")
    if chunks:
        await cl.Message(
            content="\n\n---\n\n".join(chunks),
            author="render_files",
        ).send()


async def _apply_stream_record(
    rec: dict[str, Any],
    *,
    session_id: str,
    assistant: cl.Message,
    pending_tools: list[Step],
    pending_names: list[str],
    think_body: list[str],
    think_step: Step | None,
) -> tuple[list[str], Step | None]:
    k = rec.get("kind")
    if k == "raw" and rec.get("type") == "response.output_text.delta":
        d = rec.get("delta") or ""
        if isinstance(d, str) and d:
            await assistant.stream_token(d)
    elif k == "raw" and rec.get("type") == "response.reasoning_text.delta":
        d = rec.get("delta") or ""
        if isinstance(d, str) and d:
            think_body.append(d)
            if think_step is None:
                think_step = Step(name="Thinking", default_open=False)
                think_step.output = ""
                await think_step.send()
            think_step.output = "".join(think_body)
            await think_step.update()
    elif k == "run_item":
        nm = rec.get("name")
        if nm == "tool_called":
            tn = rec.get("tool_name") or "tool"
            st = Step(name=tn, type="tool", show_input="json")
            raw_args = rec.get("tool_args") or "{}"
            if isinstance(raw_args, str) and raw_args.strip():
                try:
                    st.input = json.dumps(
                        json.loads(raw_args), indent=2, ensure_ascii=False
                    )
                except json.JSONDecodeError:
                    st.input = raw_args
            else:
                st.input = "{}"
            await st.send()
            pending_tools.append(st)
            pending_names.append(tn)
        elif nm == "tool_output":
            out = rec.get("output_preview") or ""
            if pending_tools:
                st = pending_tools.pop(0)
                tnm = pending_names.pop(0) if pending_names else ""
                st.output = str(out)[:12_000]
                await st.update()
                if tnm == "render_files":
                    await _emit_render_files_ui(session_id, st.input or "{}")
    elif k == "agent":
        an = rec.get("name", "agent")
        async with Step(name="agent", type="undefined") as st4:
            st4.output = str(an)
    elif k == "llm":
        preview = str(rec.get("output_preview") or "")[:6000]
        if preview:
            async with Step(name="LLM", default_open=False) as st_llm:
                st_llm.output = preview
    elif k == "done":
        prev = rec.get("output_preview") or ""
        if isinstance(prev, str) and prev.strip() and not (assistant.content or "").strip():
            assistant.content = prev[:12_000]
    return think_body, think_step


async def _run_temporal_stream(message: str, sid: str) -> None:
    from test_demo.temporal.client import start_orchestrator_workflow

    handle, _wf_id, _ = await start_orchestrator_workflow(message, sid)
    assistant = cl.Message(content="", author="orchestrator")
    await assistant.send()
    pending_tools: list[Step] = []
    pending_names: list[str] = []
    think_body: list[str] = []
    think_step: Step | None = None
    result_task = asyncio.create_task(handle.result())
    buffer: list[dict] = []

    async for rec in poll_workflow_stream_events(
        handle,
        workflow_cls=DeepxOrchestratorWorkflow,
        result_task=result_task,
    ):
        buffer.append(rec)
        think_body, think_step = await _apply_stream_record(
            rec,
            session_id=sid,
            assistant=assistant,
            pending_tools=pending_tools,
            pending_names=pending_names,
            think_body=think_body,
            think_step=think_step,
        )

    out = await result_task
    if not (assistant.content or "").strip():
        assistant.content = str(out) if out else "(no output)"
    await assistant.update()

    log_dir = DEMO_BACKEND.data_root / "sessions" / sid / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "events.ndjson"
    try:
        with log_path.open("w", encoding="utf-8") as f:
            for row in buffer:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


async def _run_local_stream(message: str, sid: str) -> None:
    from test_demo import orchestrator as orch

    runner = orch.build_orchestrator_runner(hitl_approval_fn=lambda *_a, **_k: True)
    assistant = cl.Message(content="", author="orchestrator")
    await assistant.send()
    pending_tools: list[Step] = []
    pending_names: list[str] = []
    think_body: list[str] = []
    think_step: Step | None = None
    buffer: list[dict] = []

    async for ev in runner.run_stream(message, session_id=sid, resume=False):
        if isinstance(ev, dict) and ev.get("kind") == "done":
            buffer.append(ev)
            think_body, think_step = await _apply_stream_record(
                ev,
                session_id=sid,
                assistant=assistant,
                pending_tools=pending_tools,
                pending_names=pending_names,
                think_body=think_body,
                think_step=think_step,
            )
            continue
        rec = serialize_stream_event(ev)
        buffer.append(rec)
        think_body, think_step = await _apply_stream_record(
            rec,
            session_id=sid,
            assistant=assistant,
            pending_tools=pending_tools,
            pending_names=pending_names,
            think_body=think_body,
            think_step=think_step,
        )

    if not (assistant.content or "").strip():
        assistant.content = "(no streamed text in final message)"
    await assistant.update()

    log_dir = DEMO_BACKEND.data_root / "sessions" / sid / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "events.ndjson"
    try:
        with log_path.open("w", encoding="utf-8") as f:
            for row in buffer:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = (message.content or "").strip()
    low = text.lower()
    if low in ("/temporal", "/temporal on"):
        cl.user_session.set("use_temporal", True)
        await cl.Message(content="Run mode: **Temporal** (discrete tool events).").send()
        return
    if low in ("/local", "/temporal off"):
        cl.user_session.set("use_temporal", False)
        await cl.Message(
            content="Run mode: **Local** (token streaming in this Chainlit process)."
        ).send()
        return

    if cl.user_session.get("use_temporal") is None:
        cl.user_session.set("use_temporal", default_use_temporal())

    sid = _deepx_session_id()
    cl.user_session.set("session_id", sid)
    turns = int(cl.user_session.get("turns") or 0)

    use_temporal = bool(cl.user_session.get("use_temporal"))
    if use_temporal:
        try:
            await _run_temporal_stream(message.content, sid)
        except Exception as e:
            await cl.ErrorMessage(
                content=(
                    f"Temporal run failed ({e!s}). Start `temporal server start-dev` and "
                    "`uv run --extra demo python -m test_demo.temporal.worker`, or send "
                    "`/local` to run without Temporal."
                )
            ).send()
            return
    else:
        try:
            await _run_local_stream(message.content, sid)
        except Exception as e:
            await cl.ErrorMessage(
                content=f"Local run failed ({e!s}). Check API keys and logs."
            ).send()
            return

    cl.user_session.set("turns", turns + 1)
    by_agent = _tool_logs_by_agent(sid)
    pref = cl.user_session.get("preferred_log_agent")
    if by_agent:
        lines = ["### Tool logs by agent"]
        items = sorted(by_agent.items())
        if isinstance(pref, str) and pref in by_agent:
            items = [(k, v) for k, v in items if k == pref] + [
                (k, v) for k, v in items if k != pref
            ]
        for agent_name, rows in items:
            tail = rows[-8:]
            lines.append(f"\n**{agent_name}** ({len(rows)} calls)")
            for row in tail:
                lines.append(
                    f"- `{row.get('tool_name', '')}` @ {row.get('timestamp', '')} "
                    f"({row.get('output_chars', '')} chars)"
                )
        await cl.Message(content="\n".join(lines)).send()
