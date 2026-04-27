# Deepx CLI (`deepx_cli`)

This package ships **optional** interactive helpers built on **`rich`** and **`prompt_toolkit`**. It is included in the same wheel as **`deepx`** (see `pyproject.toml` `[tool.hatch.build.targets.wheel]`) but **requires the `demo` extra** for those dependencies:

```bash
uv sync --extra demo
```

Repository overview: **[`README.md`](../../README.md)**. Framework details: **[`src/deepx/README.md`](../deepx/README.md)**.

---

## Exports (`deepx_cli/__init__.py`)

| Symbol | Module | Purpose |
|--------|--------|---------|
| **`run_chat_stream`** | `deepx_cli.chat_stream` | Interactive loop using **`Runner.run_streamed`**; prints assistant text deltas. |
| **`run_chat_sync`** | `deepx_cli.chat_sync` | Same REPL pattern without token streaming. |

Both use **`deepx_cli.session.run_interactive_repl`**.

---

## Session REPL (`deepx_cli/session.py`)

- **`run_interactive_repl(runner, session_id=..., run_turn=...)`** — Creates a **`Hitl`** via **`create_terminal_hitl`**, generates or resumes a **session id** (`uuid.uuid4().hex[:12]` if not provided), loops: read user line, **`runner.bind(sid, resume=(resuming or turn > 0), hitl=hitl)`**, invoke **`run_turn(binding, user_input, console)`**.
- **`parse_cli_session_arg()`** — Reads **`--session`** from **`sys.argv`** with **`argparse.parse_known_args`** so demos can pass **`python -m test_demo.orchestrator --chat --session <id>`**.
- **`/bye`** exits and prints the resume hint.

---

## Streaming vs sync

- **`chat_stream.py`** — **`run_stream_until_settled`** drains **`stream.stream_events()`**; with **`stream_text=True`**, prints **`ResponseTextDeltaEvent`** chunks.
- **`chat_sync.py`** — **`run_turn`** awaits **`binding.run(user_input)`** (non-streamed **`RunResult`**).

---

## Terminal HITL (`deepx_cli/hitl.py`)

**`create_terminal_hitl(console)`** returns **`deepx.middleware.hitl.Hitl`** with a blocking policy:

1. Prints agent name, tool name, and pretty-printed JSON arguments when possible.
2. Prompts **`[1] Reject`**, **`[2] Allow once`**, **`[3] Allow for rest of this session (this tool name)`** (also accepts short aliases like `y` / `n` / `always` per implementation).
3. Runs the blocking **`input()`** in **`asyncio.to_thread`** so it does not block the event loop.

Choice **3** maps to **`HitlDecision.ALLOW_ALWAYS`**, which persists **`(agent_name, tool_name)`** through **`Hitl`** (see **`src/deepx/README.md`** — persistence uses the **tool runner’s** `backend`).

---

## Usage pattern (demo)

Demos construct a **`DeepAgentRunner`** with **`create_deep_agent`**, then:

```python
from deepx_cli.chat_stream import run_chat_stream

run_chat_stream(orchestrator_runner, session_id=None)  # or pass explicit id
```

The **`--session`** flag is typically parsed inside **`run_chat_stream`** via **`parse_cli_session_arg`** when **`session_id`** is omitted.

---

## Files

| File | Role |
|------|------|
| **`__init__.py`** | Re-exports **`run_chat_stream`**, **`run_chat_sync`**. |
| **`session.py`** | Shared REPL, session id resolution, **`--session`** parsing. |
| **`chat_stream.py`** | Streaming turn runner. |
| **`chat_sync.py`** | Non-streaming turn runner. |
| **`hitl.py`** | **`create_terminal_hitl`**. |
