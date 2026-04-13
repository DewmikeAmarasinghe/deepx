---
name: orchestrate
description: How to run multi-step work with subagents—planning first, passing file paths only, and when to call web, SQL, or PDF specialists.
---

# Orchestration (main agent)

You coordinate specialists. You do **not** replace them for their core work (bulk web research, SQL against bundled DBs, PDF tooling).

## Planning (mandatory for multi-step work)

- If the user’s request needs **more than one substantial step**, call **`write_todos` first** (after any quick `read_file` on this skill). List concrete steps with the first marked `in_progress`.
- After **every** major tool or subagent call, use **`update_todos`** to mark progress. Do not skip planning because you are the “outer” agent—plans persist under your agent name and keep runs debuggable.
- Use **`think_tool`** only when blocked or genuinely uncertain—not as a substitute for todos.

## Delegation rules

- **One strong call per specialist** when possible: give a self-contained brief and expected artifacts (paths under the session workspace).
- **Pass paths, not pasted file bodies.** The session workspace is shared; use `read_file` yourself only for short checks.
- **Web (`web_agent`):** research, structured notes, and **final markdown reports** in the workspace when the brief includes a written deliverable. Then use **`render_files`** so the user sees finished documents in the terminal.
- **SQL (`sql_agent`):** questions against the bundled SQLite sample for the configured database; read returned tables/paths, summarise for the user.
- **PDF (`pdf_agent`):** merge/split/extract/summaries over PDFs under the repo; prefer paths like `/test_demo/pdfs/...` when files live in the demo tree.

## Skills scope

- You only have this orchestration skill in your catalog. Rely on subagent tools for domain execution; do not assume other skill folders apply to you.
