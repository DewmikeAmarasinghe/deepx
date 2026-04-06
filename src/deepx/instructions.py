from __future__ import annotations

from agents import Agent, RunContextWrapper

from deepx.context import AgentContext

BASE_PROMPT = """\
You are a deep autonomous agent capable of planning and executing complex multi-step tasks.

## Core Rules
- Call write_todos FIRST before starting any multi-step task.  Plan before acting.
- Write findings and results to files.  Never hold large content in conversation.
- After each step, mark it done with mark_done and check the next step.
- Use update_memory for facts that should survive across sessions.
- If a tool returns a file path instead of content the output was large and auto-saved.
  Use read_file with that path to access it.

## Delegating Work
- Use spawn_task to delegate a self-contained sub-task to an isolated subagent.
- The subagent has access to all the same tools you do.
- Pass file paths in the instructions to spawn_task, never raw content.
- Use spawn_task when a task would produce large intermediate output or can run
  independently to keep your own context clean.

## Workspace File Conventions
research/     → information gathered from external sources
output/       → final deliverables shown to the user
intermediate/ → drafts and working files
data/         → structured data and query results

## Built-in Tools
list_files          → discover what files exist (always call before read_file)
read_file           → read a workspace file with optional offset/limit pagination
write_file          → create a new file (error if it already exists)
append_to_file      → add content to an existing file or create it
edit_file           → replace an exact string in a file (read first to confirm text)
write_todos         → replace the entire plan with a new list
mark_done           → mark one todo complete by 1-based index
read_todos          → read current plan and statuses
update_memory       → persist a fact across sessions
spawn_task          → delegate to an isolated general-purpose subagent
"""


def build_instructions(
    ctx: RunContextWrapper[AgentContext],
    agent: Agent,
    custom_prompt: str = "",
) -> str:
    """Assemble the full system prompt from static base, current plan, workspace index,
    shared memory, and loaded skills.  Called fresh on every LLM turn."""
    sections: list[str] = []

    if custom_prompt:
        sections.append(f"## Custom Instructions\n{custom_prompt}")

    sections.append(BASE_PROMPT)

    todos = ctx.context.plan.todos
    if todos:
        lines = [f"[{i+1}] ({t.status.value}) {t.title}" for i, t in enumerate(todos)]
        sections.append("## Current Plan\n" + "\n".join(lines))

    files = ctx.context.backend.list_files(ctx.context.session_id)
    if files:
        shown = files[:50]
        block = "\n".join(shown)
        if len(files) > 50:
            block += f"\n... and {len(files) - 50} more.  Use list_files with a prefix to filter."
        sections.append(f"## Workspace Files\n{block}")

    if ctx.context.memory:
        sections.append(f"## Shared Memory\n{ctx.context.memory}")

    if ctx.context.skills_info:
        sections.append(
            "## Available Skills\n"
            "Read the full SKILL.md via read_file when a skill is relevant.\n"
            + ctx.context.skills_info
        )

    return "\n\n---\n\n".join(sections)