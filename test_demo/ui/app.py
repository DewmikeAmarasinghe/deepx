"""Chainlit UI for the orchestrator (Temporal + OpenAI Agents streaming).

Run from **repository root**::

    uv sync --extra demo
    uv run chainlit run test_demo/ui/app.py

Temporal (required for this UI): ``temporal server start-dev`` and
``uv run --extra demo python -m test_demo.temporal.worker``.

Stream events are mirrored from the workflow query ``get_stream_events`` and, after each run,
appended to ``.deepx/sessions/<session_id>/logs/events.ndjson`` for local inspection.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import chainlit as cl
from chainlit.action import Action
from chainlit.step import Step

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from test_demo.orchestrator import DEMO_BACKEND  # noqa: E402
from test_demo.temporal.workflows import DeepxOrchestratorWorkflow  # noqa: E402


def _list_session_ids() -> list[str]:
    sessions = DEMO_BACKEND.data_root / "sessions"
    if not sessions.is_dir():
        return []
    return sorted(
        p.name for p in sessions.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


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


@cl.on_chat_start
async def on_chat_start() -> None:
    sessions = _list_session_ids()
    if not cl.user_session.get("session_id"):
        cl.user_session.set("session_id", sessions[-1] if sessions else None)
    cl.user_session.set("turns", 0)

    actions = [
        Action(name="pick_session", payload={"sid": sid}, label=sid[:16])
        for sid in reversed(sessions[-20:])
    ]
    actions.append(Action(name="new_session", payload={}, label="New session"))
    cur = cl.user_session.get("session_id") or "(set on first message)"
    await cl.Message(
        content=(
            f"**Session:** `{cur}`\n\n"
            "Runs use **Temporal** (start dev server + worker). Pick a session or start new, "
            "then send a message."
        ),
        actions=actions,
    ).send()


@cl.action_callback("pick_session")
async def on_pick_session(action: Action) -> None:
    sid = str((action.payload or {}).get("sid", ""))
    if sid:
        cl.user_session.set("session_id", sid)
        cl.user_session.set("turns", 0)
        await _show_session_history(sid)


@cl.action_callback("new_session")
async def on_new_session(_action: Action) -> None:
    import uuid

    sid = uuid.uuid4().hex[:12]
    cl.user_session.set("session_id", sid)
    cl.user_session.set("turns", 0)
    await cl.Message(content=f"New session `{sid}`.").send()


async def _show_session_history(sid: str) -> None:
    """Load prior stream log for this session if present."""
    path = DEMO_BACKEND.data_root / "sessions" / sid / "logs" / "events.ndjson"
    if not path.is_file():
        await cl.Message(content=f"Session `{sid}` (no saved stream log yet).").send()
        return
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        await cl.Message(content=f"Could not read history: {e}").send()
        return
    lines = [ln for ln in raw.splitlines() if ln.strip()][:400]
    await cl.Message(
        content=f"### Saved stream ({len(lines)} lines) — session `{sid}`\n\n```json\n"
        + "\n".join(lines[:80])
        + ("\n…" if len(lines) > 80 else "")
        + "\n```"
    ).send()


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


async def _run_temporal_stream(message: str, sid: str) -> None:
    from test_demo.temporal.client import start_orchestrator_workflow
    from test_demo.temporal.stream_events import poll_workflow_stream_events

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
                        await _emit_render_files_ui(sid, st.input or "{}")
        elif k == "agent":
            an = rec.get("name", "agent")
            async with Step(name="agent", type="undefined") as st4:
                st4.output = str(an)

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


@cl.on_message
async def on_message(message: cl.Message) -> None:
    import uuid

    sid = cl.user_session.get("session_id") or uuid.uuid4().hex[:12]
    cl.user_session.set("session_id", sid)
    turns = int(cl.user_session.get("turns") or 0)

    try:
        await _run_temporal_stream(message.content, sid)
    except Exception as e:
        await cl.ErrorMessage(
            content=(
                f"Temporal run failed ({e!s}). Start `temporal server start-dev` and "
                "`uv run --extra demo python -m test_demo.temporal.worker`, then try again."
            )
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


@cl.set_chat_profiles
async def chat_profiles() -> list[cl.ChatProfile]:
    """Per-agent tool-log lens (mirrors Chainlit chat profiles pattern)."""
    sid = cl.user_session.get("session_id")
    names = ["orchestrator"]
    if isinstance(sid, str) and sid:
        names = sorted(_tool_logs_by_agent(sid).keys()) or names
    return [
        cl.ChatProfile(
            name=n,
            markdown_description=f"After each message, tail tool JSON logs for **{n}**.",
        )
        for n in names[:12]
    ]


@cl.on_chat_resume
async def on_chat_resume(thread: dict) -> None:
    """Restore session id from persisted thread when using Chainlit data layer."""
    tid = thread.get("id")
    if tid:
        cl.user_session.set("session_id", str(tid))


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    """Chat profile selection → filter which agent summary we emphasize."""
    prof = settings.get("chat_profile")
    if isinstance(prof, str) and prof.strip():
        cl.user_session.set("preferred_log_agent", prof.strip())
