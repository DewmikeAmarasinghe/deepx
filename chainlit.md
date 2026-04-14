# Deepx demo 🚀🤖

This UI drives the **Deepx** multi-agent demo: an **orchestrator** plus **web**, **SQL**, and
**PDF** subagents on the OpenAI **Agents SDK**, with **Temporal** for durable runs (model/tool
steps appear as Temporal activities via `OpenAIAgentsPlugin`).

## How to run

1. From the repo root: `uv sync --extra demo`
2. Start Temporal: `temporal server start-dev`
3. Worker: `python -m test_demo.temporal.worker`
4. UI: `uv run chainlit run test_demo/ui/app.py`

## What to try

- Pick a **session** from the actions bar or start **New session**.
- Use **Chat profiles** (top) to focus the post-run tool-log summary on one agent.
- After each turn, stream events are saved under
  `.deepx/sessions/<session_id>/logs/events.ndjson`.

For orchestrator norms and skills, see `test_demo/skills/orchestrate/SKILL.md`.
