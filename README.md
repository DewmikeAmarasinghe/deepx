# Deepx

> Long-running, tool-using agents on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python): **filesystem-scoped backends**, **skills**, **HITL**, **subagents as tools**, SQLite **session + compaction**, and **LangSmith** tracing when configured.

A **backend** defines what the agent can touch (`root_dir`, optional shell, in-memory dict). Everything else—tools, prompts, hooks, eviction, **`interrupt_on`**—is implemented against **`BackendProtocol`**.

**`deepx_cli`** is a **minimal terminal REPL** (streaming or sync chat, `--session` resume, Rich HITL). It is **not** a full product CLI. See [`src/deepx_cli/README.md`](src/deepx_cli/README.md).

**Design history / removed experiments:** [`removed_features.md`](removed_features.md).

---

## Documentation map

| File                                                     | Contents                                                           |
| -------------------------------------------------------- | ------------------------------------------------------------------ |
| **[`README.md`](README.md)** (this file)                 | Install, env vars, repo layout, quick start                        |
| **[`src/deepx/README.md`](src/deepx/README.md)**         | Framework: factory, backends, middleware, tools, sessions, prompts |
| **[`src/deepx_cli/README.md`](src/deepx_cli/README.md)** | Terminal chat, HITL policy, REPL internals                         |
| **[`test_demo/README.md`](test_demo/README.md)**         | Demo orchestrator + specialists                                    |

---

## Installation

**Python ≥ 3.11**, [uv](https://github.com/astral-sh/uv) recommended.

```bash
uv sync                 # core library
uv sync --extra demo    # deepx_cli (rich, prompt_toolkit), demo deps
```

Copy **`.env.example`** → **`.env`** and set keys.

---

## Quick start (library)

```python
from pathlib import Path
from deepx import create_deep_agent, FilesystemBackend

runner = create_deep_agent(
    name="assistant",
    system_prompt="You are a helpful coding agent.",
    backend=FilesystemBackend(Path.cwd()),
    checkpointer="agent.db",
)
result = runner.run_sync("List Python files in this project.")
print(result.output)
```

Use **`runner.bind(session_id, resume=..., hitl=...)`** with **`binding.run`** / **`binding.run_streamed`** when you need full control (see [`src/deepx/README.md`](src/deepx/README.md)).

### Interactive chat (demo)

The **maintained** entrypoint with **`--chat`** / **`--session`** is the orchestrator:

```bash
python -m test_demo.orchestrator --chat
python -m test_demo.orchestrator --chat_sync
python -m test_demo.orchestrator --chat --session <session_id>
```

Individual specialist modules (**`web_agent.py`**, etc.) are imported as libraries by the orchestrator; they do not all define the same CLI flags—check each file if you run it directly.

---

## Repository layout

**Text tree** (source + typical runtime dirs at repo root when `root_dir` is the repo):

```text
.
├── .deepx/                          # metadata (backend data_root); see src/deepx/README.md
│   ├── AGENTS.md                    # save_memory
│   └── sessions/<id>/
│       ├── approvals.json           # HITL persistance (optional)
│       └── logs/                    # if create_deep_agent(..., debug=True)
│           ├── plans/
│           └── tools/<tool>/<n>.json
├── _outputs/                        # agent path /_outputs/ — deliverables + large_tool_results/
├── src/
│   ├── deepx/                       # core framework (see src/deepx/README.md)
│   └── deepx_cli/                   # terminal REPL
├── test_demo/                       # orchestrator, agents, skills, dbs/
├── pyproject.toml
├── README.md
└── removed_features.md
```

### `.deepx/` vs `debug=True`

- **Always (when used):** **`AGENTS.md`** (append via **`save_memory`**); **`approvals.json`** after HITL **allow always** (via the **tool runner’s** backend).
- **Only if `debug=True`:** **`logs/plans/`**, **`logs/tools/`**, plan **`events.json`**.

### `_outputs/`

Human-facing artifacts under agent path **`/_outputs/`**; **`large_tool_results/`** holds evicted oversized tool returns (see middleware **`tool_pipeline`**).

---

## Environment variables

| Variable                   | Required          | Purpose                                   |
| -------------------------- | ----------------- | ----------------------------------------- |
| **`OPENAI_API_KEY`**       | For OpenAI models | API access                                |
| **`LANGSMITH_API_KEY`**    | Optional          | LangSmith tracing (`setup_observability`) |
| **`HF_TOKEN`**             | Optional          | Hugging Face Hub demo agent               |
| **`TAVILY_API_KEY`** / CLI | Optional          | Tavily / `tvly` for web demo              |

---

## Development

```bash
uv run ruff check src/
uv run ty check src/deepx
```

(There is no guaranteed **`pytest`** suite in-tree; use the checks above unless you add tests.)

---

## Dependencies

Declared in **`pyproject.toml`** (e.g. **`openai-agents`**, **`pydantic`**, **`aiosqlite`**, **`pyyaml`**, **`langsmith`**, **`wcmatch`**).
