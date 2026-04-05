from __future__ import annotations
from agents import RunContextWrapper, Agent
from deepx.context import AgentContext


BASE_PROMPT = """You are a deep autonomous agent capable of planning and executing complex,
multi-step tasks.

## Core Rules
- Call write_todos FIRST before starting any multi-step task. Plan before acting.
- Write findings and results to files using write_file. Never hold large content in conversation.
- Pass file paths to subagents, not raw content. Subagents read files themselves.
- When tasks are independent of each other, invoke multiple tools in a single response.
- After each major step, mark it done with mark_done and verify the result.
- Use update_memory for facts that should persist across sessions.
- If a tool returns a file path instead of content, the output was large and was auto-saved.
  Use read_file with the path to access it.

## Delegating Work
- Use spawn_task to delegate self-contained sub-tasks to an isolated general-purpose subagent.
- The subagent has access to all the same tools you do.
- Always pass file paths in instructions to spawn_task, never raw content.
- Use spawn_task when a task would produce large output or can run independently.

## Workspace File Organization
research/          → information gathered from external sources
output/            → final deliverables
intermediate/      → working files, drafts, intermediate results
data/              → structured data, query results

## Tool Usage
- list_files: always call before reading to discover what exists
- read_file: supports offset/limit pagination for large files
- write_file: creates new files (errors if exists)
- append_to_file: adds to existing files
- edit_file: exact string replacement (read first, then edit)
- write_todos: replace entire plan
- mark_done: mark a single todo complete by index
- read_todos: check current plan status
- update_memory: persist cross-session facts
- spawn_task: delegate to an isolated general-purpose subagent
"""


def build_instructions(
    ctx: RunContextWrapper[AgentContext],
    agent: Agent,
    custom_prompt: str = "",
) -> str:
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
        displayed = files[:50]
        extra = len(files) - len(displayed)
        file_block = "\n".join(displayed)
        if extra:
            file_block += f"\n... and {extra} more. Use list_files with a prefix to filter."
        sections.append(f"## Workspace Files\n{file_block}")

    if ctx.context.memory:
        sections.append(f"## Shared Memory\n{ctx.context.memory}")

    if ctx.context.skills_info:
        sections.append(
            "## Available Skills\nRead the full SKILL.md via read_file when a skill applies.\n"
            + ctx.context.skills_info
        )

    return "\n\n---\n\n".join(sections)