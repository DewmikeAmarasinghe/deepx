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


WRITE_TODOS_TOOL_DESCRIPTION = """\
Create or replace the current task list. This is your primary planning tool.

**When to call:**
Call this for any task involving more than a single direct answer — tool use, subagent delegation,
multi-step work, or even a sequence of reads followed by a response. If in doubt, write the plan.

**Before building the list — understand your capabilities:**
Review all available tools and subagent descriptions first. Design steps that match what each tool
or subagent can accomplish in a single invocation. Do not fragment work unnecessarily:
- BAD: one step per research sub-topic when a single subagent call can handle all of them together.
- GOOD: one delegation step covering the full scope, followed by a synthesis step.

**Lifecycle — all five cases:**

1. **Creating the plan** — write all steps you can foresee. Mark the first step `in_progress`.
   Subsequent steps should be `pending`.

2. **Step completes** — mark it `completed` and mark the next step `in_progress` in the same call.
   Never leave the plan between states — always update in one atomic write.

3. **Plan changes mid-run** — when `think_tool` reveals new information or a different approach is
   needed, revise the todos: update descriptions, insert new steps, reorder, or split steps. Mark
   the correct next step `in_progress`. Pass the full list including already-completed steps.

4. **Blocked** — keep the current step `in_progress`. Append a new step describing the specific
   blocker (what failed, what is needed to proceed). Do not mark as completed.

5. **New work discovered** — append the new steps and update statuses in the same call.
   Never call write_todos separately just to add steps.

**Rules:**
- Always pass the **complete list** — never omit existing entries. The full history is visible
  to you and helps track what has been done.
- Never call `write_todos` multiple times in parallel.
- Keep completed steps — do not delete them.
- Maintain exactly **one** step `in_progress` unless running genuinely parallel independent work.

**States:**
- `pending`     — not yet started
- `in_progress` — currently being worked on (exactly one at a time, normally)
- `completed`   — fully done and verified

The only time you may skip this tool entirely is for a purely conversational reply with zero tool use.\
"""

_READ_TODOS_DESCRIPTION = """\
Read your current plan state. Call this BEFORE every action step.

This is a mandatory checkpoint before using any tool, calling any subagent, or writing any file.
Confirm:
- Which step is currently `in_progress` and exactly what it requires you to do.
- What has been completed and what comes after the current step.
- Whether you are about to take the correct next action.

The plan may have been revised by a prior `think_tool` call — always read the latest state before
acting. Never assume you know what the plan says; always read it.\
"""


@function_tool(description_override=_READ_TODOS_DESCRIPTION)
def read_todos(ctx: RunContextWrapper[AgentContext]) -> str:
    """Read the current todo list for this agent."""
    todos = ctx.context.plan.todos
    if not todos:
        return "No todos yet. Use write_todos to create your plan before taking any action."
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
