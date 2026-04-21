from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepx.context import AgentContext
from deepx.middleware.logs import (
    run_log_append_plan_event,
    run_log_save_plan,
)


def _slug_id(title: str, *, used: set[str]) -> str:
    s = (title or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if not s:
        s = "task"
    base = s[:48]
    cand = base
    n = 2
    while cand in used:
        suf = f"-{n}"
        cand = base[: max(1, 48 - len(suf))] + suf
        n += 1
    used.add(cand)
    return cand


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"


class Todo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    content: str = Field(default="", description="Todo description.")
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
        used: set[str] = set()
        new: list[Todo] = []
        for t in self.todos:
            nid = (t.id or "").strip()
            if not nid:
                nid = _slug_id(t.content, used=used)
            elif nid in used:
                nid = _slug_id(t.content, used=used)
            else:
                used.add(nid)
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


WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool
It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - There are unresolved issues or errors
     - Work is partial or incomplete
     - You encountered blockers that prevent completion
     - You couldn't find necessary resources or dependencies
     - Quality standards haven't been met

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.

**Full list replace:** Each call replaces the entire todo list. Pass every item you want to keep, in order.

**Parallel calls:** Never call `write_todos` multiple times in parallel in the same model turn.
"""

THINK_TOOL_DESCRIPTION = """\
Use this tool to reason explicitly before a large plan change, after a surprising tool result, or
when you are uncertain which branch to take next.

**What to put in `reflection` (short, structured):**
- What you observed and whether it matched expectations
- New constraints or blockers
- Whether the current todo list should change
- The next concrete action (not a full rehash of the whole run)

Afterwards, if the plan should change, call `write_todos` with a **full replace** of the list. \
This tool is optional; do not call it on every step.\
"""


@function_tool(description_override=THINK_TOOL_DESCRIPTION)
def think_tool(ctx: RunContextWrapper[AgentContext], reflection: str) -> str:
    """Structured reflection; prefer updating `write_todos` when the plan must change."""
    todos = [
        {"id": t.id, "content": t.content, "status": t.status.value}
        for t in ctx.context.plan.todos
    ]
    plan_block = (
        json.dumps(todos, indent=2)
        if todos
        else "(no plan yet — write_todos has not been called)"
    )
    return (
        f"Current plan:\n{plan_block}\n\n"
        "If the plan needs updating, call write_todos with the full list. "
        f"Reflection:\n{reflection}\n"
    )


@function_tool(description_override=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    ctx: RunContextWrapper[AgentContext],
    todos: list[TodoInput],
) -> str:
    """Create and manage a structured task list for your current work session (full replace)."""
    used: set[str] = set()
    ctx.context.plan.todos = [
        Todo(
            id=_slug_id(t.content, used=used),
            content=t.content,
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
            "todos": [
                {"id": t.id, "content": t.content, "status": t.status.value}
                for t in ctx.context.plan.todos
            ],
        }
        run_log_append_plan_event(
            ctx.context.backend, ctx.context.session_id, json.dumps(entry)
        )
    return _format_plan(ctx)


def _format_plan(ctx: RunContextWrapper[AgentContext]) -> str:
    lines = [f"[{t.id}] ({t.status.value}) {t.content}" for t in ctx.context.plan.todos]
    return "Plan saved:\n" + "\n".join(lines)


def _safe_status(value: str) -> TodoStatus:
    try:
        return TodoStatus(value)
    except ValueError:
        return TodoStatus.pending


def _persist_plan(ctx: RunContextWrapper[AgentContext]) -> None:
    ctx.context.plan.agent_name = ctx.context.agent_name or ctx.context.plan.agent_name
    if not ctx.context.debug:
        return
    run_log_save_plan(
        ctx.context.backend,
        ctx.context.session_id,
        ctx.context.plan.agent_name,
        ctx.context.plan.to_json(),
    )
