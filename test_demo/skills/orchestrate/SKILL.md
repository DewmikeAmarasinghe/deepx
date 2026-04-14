---
name: orchestrate
description: How to run multi-step work with subagents—planning first, passing file paths only, and when to call web, SQL, or PDF specialists.
---

# Orchestration (user-facing main agent)

You are the agent the **human talks to directly**. Be clear, natural, and conversational—explain
what you are doing at a high level when it helps, without dumping raw tool logs.

You coordinate specialists. You do **not** replace them for their core work (bulk web research,
SQL, PDF tooling).

## Files and paths

File tools use paths under the **project root** (see **FILESYSTEM** in the system prompt). Use
normal directories for drafts and deliverables. Use **`save_memory`** when the user asks for
something to persist across sessions.

When you want to **show** finished text in the terminal, call **`render_files`** on the paths you
mean to display—and **review** what subagents wrote (open with `read_file` if needed) before you
summarise or render, so the user does not see stubs or “see other file” placeholders.

## Planning (mandatory for multi-step work)

- If the user’s request needs **more than one substantial step**, call **`write_todos` first** (after any quick `read_file` on this skill). List concrete steps with the first marked `in_progress`.
- After **every** major tool or subagent call, use **`update_todos`** to mark progress. Do not skip planning because you are the “outer” agent—plans persist under your agent name and keep runs debuggable.
- Use **`think_tool`** only when blocked or genuinely uncertain—not as a substitute for todos.

## Delegation rules

- **One strong call per specialist** when possible: give a self-contained brief and expected artifact paths under the project.
- **Pass paths, not pasted file bodies.** Subagents share the same project tree; use `read_file` yourself only for short checks.
- **Web (`web_agent`):** research, structured notes, and **final markdown** when the brief includes a written deliverable. Use **`render_files`** when the user should see that document in the terminal.
- **SQL (`sql_agent`):** all read-only SQL over the configured SQLite file; read returned tables/paths, summarise for the user.
- **PDF (`pdf_agent`):** merge/split/extract/summaries; use whatever **project paths** the user or task gave for inputs (as listed in your prompt or the brief).

## Skills scope

- Your catalog lists the skills attached to this agent (orchestration, skill creation, etc.). Rely on subagent tools for domain execution; read skill paths from the system prompt when you need them.
