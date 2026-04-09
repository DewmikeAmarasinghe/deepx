from __future__ import annotations

import json

from agents import RunContextWrapper, function_tool

from deepx.context import AgentContext

_THINK_DESCRIPTION = """\
Mandatory reflection tool. Call this after EVERY tool result and EVERY subagent response.

Purpose: force structured reasoning before your next action. This tool exists because agents
tend to follow stale plans even when the latest result has changed the situation. It catches
that divergence and gives you the current plan state so you can decide what to do.

What to write in `reflection`:
- What did the tool/subagent return? Was it what you expected?
- What new information, requirements, or constraints did this reveal?
- Does the current plan still make sense, or does it need updating?
- What is the correct next step given what you now know?

After calling this tool, you MUST:
1. Call `read_todos` to confirm the exact current plan state.
2. Either advance the plan (mark current step completed, next in_progress) — or revise it
   (update/add/reorder steps with write_todos) if the result changed your understanding.

Never skip this tool. Never proceed to the next action without calling it first.\
"""


@function_tool(description_override=_THINK_DESCRIPTION)
def think_tool(ctx: RunContextWrapper[AgentContext], reflection: str) -> str:
    """Think through the current situation before acting or before updating your plan."""
    todos = [
        {"step": t.title, "status": t.status.value}
        for t in ctx.context.plan.todos
    ]
    plan_block = json.dumps(todos, indent=2) if todos else "(no plan yet — write_todos has not been called)"

    return (
        f"Reflection recorded: {reflection}\n\n"
        f"Current plan:\n{plan_block}\n\n"
        "Act on this reflection now:\n"
        "1. Call `read_todos` to confirm the latest plan state.\n"
        "2. If the step just completed as expected:\n"
        "   → Call `write_todos` to mark it `completed` and the next step `in_progress`.\n"
        "3. If this result revealed new information, changed requirements, or a different\n"
        "   approach is needed:\n"
        "   → Call `write_todos` to revise the plan (update, insert, or reorder steps as\n"
        "     needed) and mark the correct next step `in_progress`.\n"
        "   → Example: a website instructs you to visit another URL not in your original plan.\n"
        "     Add that step before continuing.\n"
        "Never proceed to your next action without updating the plan."
    )
