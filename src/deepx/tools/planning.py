from __future__ import annotations

import json
from datetime import datetime, timezone

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext
from deepx.models import Todo, TodoStatus


@function_tool
def write_todos(ctx: RunContextWrapper[AgentContext], todos: list[str]) -> str:
    """Replace the current todo list. Call before multi-step work."""
    ctx.context.plan.todos = [Todo(title=t) for t in todos]
    ctx.context.plan.updated_at = datetime.now(timezone.utc).isoformat()
    _persist_plan(ctx)
    if ctx.context.debug:
        entry = {
            "timestamp": ctx.context.plan.updated_at,
            "agent": ctx.context.agent_name,
            "todos": todos,
        }
        ctx.context.backend.append_plan_log(
            ctx.context.session_id, json.dumps(entry)
        )
    lines = [
        f"[{i + 1}] ({t.status.value}) {t.title}"
        for i, t in enumerate(ctx.context.plan.todos)
    ]
    return "Plan saved:\n" + "\n".join(lines)


@function_tool
def mark_done(ctx: RunContextWrapper[AgentContext], index: int) -> str:
    """Mark a todo completed by 1-based index."""
    todos = ctx.context.plan.todos
    if index < 1 or index > len(todos):
        return f"Error: index {index} out of range. Plan has {len(todos)} items."
    todos[index - 1].status = TodoStatus.completed
    ctx.context.plan.updated_at = datetime.now(timezone.utc).isoformat()
    _persist_plan(ctx)
    return f"Marked done: {todos[index - 1].title}"


@function_tool
def read_todos(ctx: RunContextWrapper[AgentContext]) -> str:
    """Read todos and statuses."""
    todos = ctx.context.plan.todos
    if not todos:
        return "No plan yet. Call write_todos to create one."
    lines = [
        f"[{i + 1}] ({t.status.value}) {t.title}" for i, t in enumerate(todos)
    ]
    return "\n".join(lines)


def _persist_plan(ctx: RunContextWrapper[AgentContext]) -> None:
    ctx.context.plan.agent_name = ctx.context.agent_name or ctx.context.plan.agent_name
    ctx.context.backend.save_plan(
        ctx.context.session_id,
        ctx.context.plan.agent_name,
        ctx.context.plan.to_json(),
    )
