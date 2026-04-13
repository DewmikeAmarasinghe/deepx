from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepx.context import AgentContext
from deepx.middleware.logs import (
    run_log_append_plan_event,
    run_log_save_plan,
)


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"


class Todo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    title: str
    status: TodoStatus = TodoStatus.pending


class Plan(BaseModel):
    session_id: str
    agent_name: str
    tasks: list[str] = Field(default_factory=list)
    todos: list[Todo] = Field(default_factory=list)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def ensure_todo_ids(self) -> Plan:
        new: list[Todo] = []
        for i, t in enumerate(self.todos):
            nid = t.id if t.id else str(i + 1)
            new.append(t.model_copy(update={"id": nid}))
        self.todos = new
        return self

    def pending(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.pending]

    def in_progress(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.in_progress]

    def completed(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.completed]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class TodoInput(BaseModel):
    content: str
    status: str = "pending"


class TodoPatch(BaseModel):
    id: str = Field(description="Numeric todo id from the plan, e.g. '1', '2'.")
    title: str | None = None
    status: str | None = None


WRITE_TODOS_TOOL_DESCRIPTION = """\
Replace the entire task list (initial plan or full reset). Assigns todo ids `1`, `2`, `3`, … in order.

**When to call:** First time you need a plan for multi-step work, or when you want to discard the
current list and start fresh.

**Prefer `update_todos`** for small changes after a plan exists (mark one step completed, tweak a title).

**Rules:**
- Pass every step in order. Mark the first step `in_progress`, later steps `pending` until you advance.
- When a step completes, use `update_todos` to set it `completed` and the next `in_progress`.
- Never call `write_todos` in parallel with other planning tools.

**States:** `pending`, `in_progress`, `completed`, `cancelled`
"""


UPDATE_TODOS_TOOL_DESCRIPTION = """\
Patch existing todos by id (ids are `1`, `2`, `3`, … as shown in the plan). Use this for most updates
after the plan exists — cheaper than rewriting the full list.

For each patch, provide `id` and optionally `title` and/or `status`. Omitted fields stay unchanged.

**Examples:**
- Mark step 1 done and step 2 active: two patches `{id: "1", status: "completed"}` and `{id: "2", status: "in_progress"}`.
- Rename step 3: `{id: "3", title: "New title"}`.

If an id does not exist, the tool returns an error listing valid ids.
"""


@function_tool(description_override=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    ctx: RunContextWrapper[AgentContext],
    todos: list[TodoInput],
) -> str:
    """Replace the full todo list; ids are reassigned 1..n."""
    ctx.context.plan.todos = [
        Todo(
            id=str(i + 1),
            title=t.content,
            status=_safe_status(t.status),
        )
        for i, t in enumerate(todos)
    ]
    ctx.context.plan.updated_at = datetime.now(timezone.utc).isoformat()
    _persist_plan(ctx)
    if ctx.context.debug:
        entry = {
            "timestamp": ctx.context.plan.updated_at,
            "agent": ctx.context.agent_name,
            "todos": [{"id": t.id, "content": t.title, "status": t.status.value} for t in ctx.context.plan.todos],
        }
        run_log_append_plan_event(
            ctx.context.backend, ctx.context.session_id, json.dumps(entry)
        )
    return _format_plan(ctx)


@function_tool(description_override=UPDATE_TODOS_TOOL_DESCRIPTION)
def update_todos(
    ctx: RunContextWrapper[AgentContext],
    patches: list[TodoPatch],
) -> str:
    """Apply patches to existing todos by id."""
    if not ctx.context.plan.todos:
        return "No plan yet. Call write_todos first."
    by_id = {t.id: t for t in ctx.context.plan.todos}
    valid = ", ".join(
        sorted(by_id.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    )
    for p in patches:
        if p.id not in by_id:
            return f"Unknown todo id {p.id!r}. Valid ids: {valid}"
    for p in patches:
        t = by_id[p.id]
        if p.title is not None:
            t.title = p.title
        if p.status is not None:
            t.status = _safe_status(p.status)
    ctx.context.plan.updated_at = datetime.now(timezone.utc).isoformat()
    _persist_plan(ctx)
    if ctx.context.debug:
        entry = {
            "timestamp": ctx.context.plan.updated_at,
            "agent": ctx.context.agent_name,
            "todos": [{"id": t.id, "content": t.title, "status": t.status.value} for t in ctx.context.plan.todos],
        }
        run_log_append_plan_event(
            ctx.context.backend, ctx.context.session_id, json.dumps(entry)
        )
    return _format_plan(ctx)


def _format_plan(ctx: RunContextWrapper[AgentContext]) -> str:
    lines = [
        f"[{t.id}] ({t.status.value}) {t.title}"
        for t in ctx.context.plan.todos
    ]
    return "Plan saved:\n" + "\n".join(lines)


def _safe_status(value: str) -> TodoStatus:
    try:
        return TodoStatus(value)
    except ValueError:
        return TodoStatus.pending


def _persist_plan(ctx: RunContextWrapper[AgentContext]) -> None:
    ctx.context.plan.agent_name = ctx.context.agent_name or ctx.context.plan.agent_name
    run_log_save_plan(
        ctx.context.backend,
        ctx.context.session_id,
        ctx.context.plan.agent_name,
        ctx.context.plan.to_json(),
    )
