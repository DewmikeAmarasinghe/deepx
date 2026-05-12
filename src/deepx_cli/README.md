# deepx_cli

Minimal **terminal REPL** for Deepx agents: **`run_interactive_cli`** in [`cli.py`](cli.py) → [`repl.py`](repl.py) → [`run.py`](run.py).

## Streaming layout (`--chat`)

Tool invocations render as full-width **`Panel`** rows titled **`{agent} · {tool}`**.

When **`stream_text=True`** (default for **`--chat`**), the SDK streams reasoning and answer text into **live-updating** panels (Rich **`Live`**):

- **`{agent} · thinking`** — model reasoning / summary stream (grey border).
- **`{agent} · response`** — final answer stream (green border).

With **`--chat_sync`**, only tool panels are printed for tool calls; there is no token stream.

## REPL / plans / resume

- **Enter** sends; **Esc+Enter** or **Alt+Enter** inserts a newline ([`repl.py`](repl.py)).
- Each user line uses **`runner.bind(session_id, resume=..., hitl=...)`**. **`resume`** is **True** only when the user launched with **`--session <id>`**, so follow-up messages in the same REPL session **do not** reload the previous turn’s **`write_todos`** plan into **`AgentContext`** (see **`FilesystemHooks`** in [`src/deepx/README.md`](../deepx/README.md)).
- **`/bye`** exits; a short **resume** hint is printed.

## HITL

[`hitl.py`](hitl.py) implements blocking **`input()`** for **`interrupt_on`** tools (reject / allow once / allow for session).
