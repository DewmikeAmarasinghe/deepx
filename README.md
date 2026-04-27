# Deepx

**Deepx** is a Python library that layers a **sandboxed workspace**, **built-in tools**, **skills**, **session storage**, and **human-in-the-loop (HITL)** on top of the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) (`openai-agents`). You build agents with `create_deep_agent`, bind a conversation session, and run with the SDK’s `Runner` as usual—the framework supplies filesystem tools, planning helpers, optional shell execution, and prompt assembly.

- **Python:** 3.11+ (see `pyproject.toml`).
- **Install:** `uv sync` (core) or `uv sync --extra demo` for the interactive CLI, Rich, and the `test_demo` stack.

## Documentation layout

| Document | Contents |
|----------|----------|
| **[`src/deepx/README.md`](src/deepx/README.md)** | Framework overview, backends, middleware, package file map, `create_deep_agent` parameters, `AgentContext`, sessions, outputs (`/_outputs/`, `/.deepx/`, large tool results). |
| **[`src/deepx_cli/README.md`](src/deepx_cli/README.md)** | Terminal chat entrypoints (`run_chat_stream`, `run_chat_sync`), REPL session ids, `--session` resume, terminal HITL. |
| **[`test_demo/README.md`](test_demo/README.md)** | Demo orchestrator and specialist agents; how the demo uses repo paths, `/_outputs/` for deliverables, and sample DBs. |

## Quick start (library)

```python
from pathlib import Path
from deepx import create_deep_agent, FilesystemBackend

runner = create_deep_agent(
    name="assistant",
    system_prompt="You help with files under the project.",
    backend=FilesystemBackend(Path.cwd()),
    checkpointer=":memory:",
    debug=False,
)
binding = runner.bind("my-session-id", resume=False, hitl=None)
# Then use agents.Runner.run / run_streamed with binding.agent, binding.session, binding.hooks, context=binding.ctx
```

See **`src/deepx/README.md`** for the full API and behavior (including **`debug=True`** and when **`sessions/&lt;id&gt;/logs`** are written).

## Repository layout (high level)

- **`src/deepx/`** — Core package: factory, backends, tools, middleware, prompts, sessions.
- **`src/deepx_cli/`** — Optional CLI helpers (requires the `demo` extra).
- **`test_demo/`** — Example multi-agent demo (orchestrator + specialists); not required to use the library.

## License and dependencies

Core runtime dependencies are declared in `pyproject.toml` (e.g. `openai-agents`, `pydantic`, `aiosqlite`, `pyyaml`, `langsmith`, `wcmatch`). Tracing hooks run when `LANGSMITH_API_KEY` is set (see `deepx.middleware.observability.setup_observability`).
