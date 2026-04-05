from __future__ import annotations
from agents import function_tool, RunContextWrapper
from deepx.context import AgentContext
from deepx.models import Todo, TodoStatus


@function_tool
def write_todos(ctx: RunContextWrapper[AgentContext], todos: list[str]) -> str:
    """Replace the current plan with a new list of todo items. Call this FIRST before starting
    any multi-step task. Each item should be a clear, actionable step. All items start as pending.
    Returns a formatted list of the saved todos."""
    ctx.context.plan.todos = [Todo(title=t) for t in todos]
    _persist_plan(ctx)
    lines = [f"[{i+1}] ({t.status}) {t.title}" for i, t in enumerate(ctx.context.plan.todos)]
    return "Plan saved:\n" + "\n".join(lines)


@function_tool
def mark_done(ctx: RunContextWrapper[AgentContext], index: int) -> str:
    """Mark a todo item as completed by its 1-based index. Call this after finishing each step."""
    todos = ctx.context.plan.todos
    if index < 1 or index > len(todos):
        return f"Error: index {index} out of range. Plan has {len(todos)} items."
    todos[index - 1].status = TodoStatus.completed
    _persist_plan(ctx)
    return f"Marked done: {todos[index - 1].title}"


@function_tool
def read_todos(ctx: RunContextWrapper[AgentContext]) -> str:
    """Read the current plan and all todo items with their status."""
    todos = ctx.context.plan.todos
    if not todos:
        return "No plan yet. Call write_todos to create one."
    lines = [f"[{i+1}] ({t.status}) {t.title}" for i, t in enumerate(todos)]
    return "\n".join(lines)


def _persist_plan(ctx: RunContextWrapper[AgentContext]) -> None:
    ctx.context.backend.save_plan(ctx.context.session_id, ctx.context.plan.to_json())