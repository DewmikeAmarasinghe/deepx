from __future__ import annotations

import json
from datetime import datetime, timezone

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel

from deepx.context import AgentContext
from deepx.models import Todo, TodoStatus


class TodoInput(BaseModel):
    content: str
    status: str = "pending"

WRITE_TODOS_TOOL_DESCRIPTION = (
    "Create or update your structured task list. Call this tool whenever your work involves more than "
    "a single direct answer — any tool use, delegation, or multi-step task of any kind.\n\n"
    "**Before building the list:** review the descriptions of available tools and subagents. "
    "Design steps that match what each tool or subagent can accomplish in a single invocation. "
    "Do not create one todo step per task when one subagent call can cover all of them."
    "   For example: creating a todo list like step-1: delegate task1 to subagent1"
    "   step-2: delegate task2 to subagent1"
    "   step-3: delegate task3 to subagent1 and so on when we can delegate all of"
    "those tasks to the subagent1 in one step\n\n" 
    "**After each step completes:** call `list_todos` first to read the current state, then call "
    "`write_todos` to mark the finished step `completed` and advance the next to `in_progress`.\n\n"
    "## Rules\n\n"
    "- Always pass the **complete list** — never omit existing entries.\n"
    "- Never call this tool multiple times in parallel.\n"
    "- Mark the first step `in_progress` when you create the plan; keep exactly one step `in_progress` "
    "unless running genuinely parallel independent work.\n"
    "- ONLY mark a step `completed` when fully done. If blocked, keep it `in_progress` and add a new "
    "step describing the blocker.\n"
    "- When new work appears mid-run, append it and update statuses in the same call.\n"
    "- Keep completed steps in the list for visibility — do not delete them.\n\n"
    "## Task states\n\n"
    "- `pending` — not yet started\n"
    "- `in_progress` — currently being worked on\n"
    "- `completed` — fully done\n\n"
    "The only time you may skip this tool is for a purely conversational reply with zero tool use."
)


@function_tool
def list_todos(ctx: RunContextWrapper[AgentContext]) -> str:
    """Return the current todo list for this agent.

    Call this after completing a step to review progress before updating write_todos.
    """
    todos = ctx.context.plan.todos
    if not todos:
        return "No todos yet. Use write_todos to create your plan."
    lines = [
        f"[{i + 1}] ({t.status.value}) {t.title}"
        for i, t in enumerate(todos)
    ]
    return "Current plan:\n" + "\n".join(lines)


@function_tool(description_override=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    ctx: RunContextWrapper[AgentContext],
    todos: list[TodoInput],
) -> str:
    """Replace the current todo list with the provided items."""
    ctx.context.plan.todos = [
        Todo(
            title=t.content,
            status=_safe_status(t.status),
        )
        for t in todos
    ]
    ctx.context.plan.updated_at = datetime.now(timezone.utc).isoformat()
    _persist_plan(ctx)
    if ctx.context.debug:
        entry = {
            "timestamp": ctx.context.plan.updated_at,
            "agent": ctx.context.agent_name,
            "todos": [{"content": t.title, "status": t.status.value} for t in ctx.context.plan.todos],
        }
        ctx.context.backend.append_plan_log(
            ctx.context.session_id, json.dumps(entry)
        )
    lines = [
        f"[{i + 1}] ({t.status.value}) {t.title}"
        for i, t in enumerate(ctx.context.plan.todos)
    ]
    return "Plan saved:\n" + "\n".join(lines)


def _safe_status(value: str) -> TodoStatus:
    try:
        return TodoStatus(value)
    except ValueError:
        return TodoStatus.pending


def _persist_plan(ctx: RunContextWrapper[AgentContext]) -> None:
    ctx.context.plan.agent_name = ctx.context.agent_name or ctx.context.plan.agent_name
    ctx.context.backend.save_plan(
        ctx.context.session_id,
        ctx.context.plan.agent_name,
        ctx.context.plan.to_json(),
    )
