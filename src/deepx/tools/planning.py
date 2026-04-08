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
    "Use this tool to create and manage a structured task list for your current work session. "
    "This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.\n\n"
    "**Multi-agent and research orchestration:** If you will use the `task` tool to delegate to subagents, "
    "or you are running a research-then-synthesize-then-write pipeline, you **must** call `write_todos` "
    "first — even when the high-level step count is small. Plan delegations, then refresh the list when "
    "each subagent returns or when new work appears.\n\n"
    "Only skip this tool for **truly trivial** single-shot requests (e.g. one direct answer with no "
    "delegation and no multi-phase workflow). If the user's request is trivial and takes less than 3 steps "
    "with no subagents, it is better to NOT use this tool and just do the task directly.\n\n"
    "## When to Use This Tool\n"
    "Use this tool in these scenarios:\n\n"
    "1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n"
    "2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations\n"
    "3. User explicitly requests todo list - When the user directly asks you to use the todo list\n"
    "4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)\n"
    "5. The plan may need future revisions or updates based on results from the first few steps\n\n"
    "## How to Use This Tool\n"
    "1. When you start working on a task - Mark it as in_progress BEFORE beginning work.\n"
    "2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.\n"
    "3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.\n"
    "4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.\n\n"
    "## When NOT to Use This Tool\n"
    "It is important to skip using this tool when:\n"
    "1. There is only a single, straightforward task\n"
    "2. The task is trivial and tracking it provides no benefit\n"
    "3. The task can be completed in less than 3 trivial steps\n"
    "4. The task is purely conversational or informational\n\n"
    "## Task States and Management\n\n"
    "1. **Task States**: Use these states to track progress:\n"
    "   - pending: Task not yet started\n"
    "   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)\n"
    "   - completed: Task finished successfully\n\n"
    "2. **Task Management**:\n"
    "   - Update task status in real-time as you work\n"
    "   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)\n"
    "   - Complete current tasks before starting new ones\n"
    "   - Remove tasks that are no longer relevant from the list entirely\n"
    "   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!\n"
    "   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.\n\n"
    "3. **Task Completion Requirements**:\n"
    "   - ONLY mark a task as completed when you have FULLY accomplished it\n"
    "   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress\n"
    "   - When blocked, create a new task describing what needs to be resolved\n\n"
    "4. **Task Breakdown**:\n"
    "   - Create specific, actionable items\n"
    "   - Break complex tasks into smaller, manageable steps\n"
    "   - Use clear, descriptive task names\n\n"
    "Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully\n"
    "Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all."
)


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
