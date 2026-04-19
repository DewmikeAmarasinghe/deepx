from __future__ import annotations

import json

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext

THINK_DESCRIPTION = """\
Use this tool when you need to think out loud before deciding what to do next.

Good times to call it:
- After an unexpected or surprising result from a tool or subagent.
- When you've been running autonomously for several steps and want to sanity-check the plan.
- When you're unsure whether the current step is still correct given new information.
- Before making a major revision to the plan.

What to write in `reflection`:
- What happened? Was the result what you expected?
- What new information or constraints did this reveal?
- Does the current plan still make sense, or does it need revising?
- What is the correct next step?

After calling this tool, update the plan if needed (`update_todos` or `write_todos`) before acting.

You do NOT need to call this after every tool result. Only call it when it would genuinely
help you reason through an ambiguous or unexpected situation.\
"""


@function_tool(description_override=THINK_DESCRIPTION)
def think_tool(ctx: RunContextWrapper[AgentContext], reflection: str) -> str:
    """Think through the current situation before acting or before updating your plan."""
    todos = [
        {"id": t.id, "title": t.title, "status": t.status.value}
        for t in ctx.context.plan.todos
    ]
    plan_block = (
        json.dumps(todos, indent=2)
        if todos
        else "(no plan yet — write_todos has not been called)"
    )

    return (
        f"Current plan:\n{plan_block}\n\n"
        "If the plan needs updating based on your reflection, call update_todos or write_todos. "
        "Otherwise continue with the next step."
        f"Reflection: {reflection}\n\n"
    )
