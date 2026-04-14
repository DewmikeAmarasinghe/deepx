# Deepx demo

This UI drives the **Deepx** multi-agent demo: an **orchestrator** plus **web**, **SQL**, and
**PDF** subagents on the OpenAI **Agents SDK**. You can run the agent **locally** (streaming in
this process) or via **Temporal** (durable run; the UI polls discrete tool/LLM events).

## How to run

1. From the repo root: `uv sync --extra demo`
2. Set **`CHAINLIT_AUTH_SECRET`** in the environment (required for login cookies). Use a **long**
   secret (at least 32 characters) to avoid weak HMAC keys, for example:
   `export CHAINLIT_AUTH_SECRET="$(openssl rand -hex 32)"`
3. UI: `uv run chainlit run test_demo/ui/app.py`

Optional **Temporal** (default on unless you turn the **Use Temporal** switch off or set
`DEEPX_USE_TEMPORAL=false`):

- `temporal server start-dev`
- `uv run --extra demo python -m test_demo.temporal.worker`

Chat history uses a SQLite database at **`test_demo/ui/chainlit.db`** (override with
`CHAINLIT_DATABASE_URL`). The demo runs **idempotent `CREATE TABLE IF NOT EXISTS`** on startup for
SQLite so Chainlit’s `SQLAlchemyDataLayer` tables exist (the stock data layer does not migrate
automatically). If you still see schema errors from an old file, remove `test_demo/ui/chainlit.db`
and restart.

### Login

Chainlit does **not** auto-submit credentials. After persistence works, use the password form
with **`admin` / `admin`**.

## Commands and settings

- **Chat settings** — **Use Temporal for agent runs** switch (default follows `DEEPX_USE_TEMPORAL`,
  normally on). The choice is stored on the thread when supported.
- **`/temporal`** / **`/local`** — optional shortcuts for the same mode.
- **Chat profiles** (top) — focus the post-run tool-log summary on one agent.

After each turn, events are saved under `.deepx/sessions/<session_id>/logs/events.ndjson`.

For orchestrator norms and skills, see `test_demo/skills/orchestrate/SKILL.md`.
