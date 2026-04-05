# deepx — Agent Harness Built on OpenAI Agents SDK

## What This Is

`deepx` is a generic Python agent harness, inspired by `langchain/deepagents`, built on top of the
`openai-agents` SDK. It gives any agent built with this framework:

- A planning tool (write_todos, mark_done, read_todos)
- A workspace filesystem (read_file, write_file, edit_file, list_files, append_to_file)
- Shared memory across agents (AGENTS.md loaded at startup)
- Skills (SKILL.md files discovered at startup, read on demand by the agent)
- Automatic tool I/O logging for observability
- Automatic large-output eviction (replaces large results with file reference + preview)
- Automatic context compaction via OpenAI Responses API
- Langfuse tracing via OTEL (automatic, zero decoration needed)
- Swappable session storage (in-memory default, SQLite for persistence)
- Human-in-the-loop hook

This framework is generic. It does not contain domain-specific logic (no visited_urls,
no SQL tools, no web scrapers). Those belong in the agent built on top of it.

---

## Core Architecture Decisions

### Naming

- The shared storage folder for all non-message artifacts is called the **workspace**,
  not VFS. The workspace root defaults to `.deepx/` relative to cwd, configurable via
  `DEEPX_WORKSPACE` env var or `workspace_path` parameter.
- Session storage (conversation history / messages array) is separate from the workspace.
- The framework uses the word `workspace` in code, docs, and user-facing APIs.

### What Goes Where

**Session (openai-agents `SQLiteSession` or in-memory):**
Only conversation history — the messages array. The session handles resuming conversations.
By default uses an in-memory session (ephemeral). Pass `db_path` to use SQLite.

**Workspace (filesystem folder):**
Everything else. Organized as:
```
.deepx/                              ← workspace root (DEEPX_WORKSPACE or .deepx/)
├── memory/
│   └── AGENTS.md                    ← shared memory loaded at startup for all agents
├── skills/
│   └── {skill-name}/
│       └── SKILL.md                 ← skill definition with YAML frontmatter
└── sessions/
    └── {session_id}/                ← per-session isolation
        ├── plan.json                ← Pydantic Plan model (todos + metadata)
        ├── files/                   ← files written by the agent via write_file
        │   └── ...user-created files
        └── tools/                   ← auto-logged tool call I/O for observability
            └── {tool_name}/
                └── {call_id}.json   ← {input, output, timestamp, agent_name, chars}
```

The `memory/` folder is shared across all sessions. The `sessions/{session_id}/` folder
is isolated per session. Subagents share the same session_id folder as their parent (same
workspace context). The `tools/` subfolder is written automatically by the framework
middleware, not by the agent.

### Middleware as RunHooks

`langchain/deepagents` uses an `AgentMiddleware` ABC with `before_agent`, `wrap_model_call`,
`wrap_tool_call` hooks. In `openai-agents`, the equivalent is subclassing `RunHooks` and
overriding `on_tool_start`, `on_tool_end`, `on_agent_start`, `on_agent_end`.

This framework implements all middleware as `RunHooks` subclasses or, for dynamic system
prompt injection, as a callable `instructions` function on the `Agent`.

### Dynamic Instructions (System Prompt)

Instead of multiple middlewares each appending to a system message, a single callable
`build_instructions(ctx, agent) -> str` reads from `AgentContext` and builds the full
system prompt fresh on every LLM call. Sections included:

1. The framework base prompt (core behavior rules, file organization, tool usage)
2. User-provided custom instructions (prepended)
3. Current plan / todos block (from `ctx.plan`)
4. Workspace files index (from `ctx.workspace_backend.list_files(session_id)`)
5. Memory (from `ctx.memory`)
6. Skills listing (names, descriptions, paths — read full SKILL.md via read_file on demand)

### Compaction

Handled by wrapping the session with `OpenAIResponsesCompactionSession` from the
`openai-agents` SDK. This wraps any `Session` and uses the OpenAI Responses API to compact
the conversation once enough turns accumulate. The middleware layer does this transparently
before passing the session to `Runner.run()`. Users do not configure it — it just works.

### Auto-Eviction of Large Tool Outputs

When `on_tool_end` fires and `len(result) > LARGE_OUTPUT_THRESHOLD` chars (~80k), the
framework:
1. Writes the full output to `sessions/{session_id}/tools/{tool_name}/{call_id}.json`
   (the output field contains the full text)
2. Writes the full output to `sessions/{session_id}/files/{tool_name}_{call_id}.txt`
   (a file the agent can read via `read_file`)
3. Replaces the result seen by the agent with a preview (first 30 lines) + message:
   `"Output too large ({n} chars). Full output saved to: sessions/{session_id}/files/{tool_name}_{call_id}.txt — use read_file to access it."`

This means the agent always knows where to find large outputs, and the main context window
is not flooded. The agent is instructed in the base prompt to pass file paths (not raw
content) to subagents.

### Observability (Langfuse)

The `openinference-instrumentation-openai-agents` library automatically captures OpenAI Agents operations and exports OpenTelemetry spans to Langfuse with zero decoration of agent code.

Setup is three env vars + two lines of code at process startup (in `deepx/observability.py`):
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic {base64(public:secret)}
```
```python
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
OpenAIAgentsInstrumentor().instrument()
```

Call `setup_observability()` from `deepx.observability` at the top of `graph.py` if the
env vars are present. Silently skipped if not configured. No user action needed.

### Parallelism

The openai-agents SDK does not have a special parallel primitive. Use `asyncio.gather` for parallel execution when you want a deterministic order, and agent-as-tool when you want the orchestrator to dynamically decide. The framework uses `asyncio.gather` inside subagent tool wrappers. When the orchestrator LLM emits multiple tool calls in one response, the SDK executes them concurrently automatically.

### Subagents

Each subagent registered with `create_deep_agent` becomes a `@function_tool`. The tool's
docstring is the subagent's description (which becomes the tool description for the LLM).
When called, it runs `Runner.run(subagent, input=description, context=ctx)` and returns
`result.final_output`. The subagent gets the same `AgentContext` as the parent (shared
workspace), but the subagent's internal conversation history is NOT stored in the parent's
session — only the final return value is added as a `ToolMessage` to the parent.

### Pydantic Models for Structured State

Todos and plan use Pydantic models so they serialize cleanly to JSON and have clear status
transitions. Tool I/O logs are also Pydantic models serialized to JSON.

---

## File Structure

```
deepx/                               ← repo root
├── pyproject.toml
├── README.md
├── AGENTS.md                        ← this file
└── src/
    └── deepx/
        ├── __init__.py              ← public API exports
        ├── _version.py
        ├── graph.py                 ← create_deep_agent() entry point
        ├── observability.py         ← setup_observability() — Langfuse via OTEL
        ├── context.py               ← AgentContext dataclass
        ├── models.py                ← Pydantic models: Plan, Todo, TodoStatus, ToolLog
        ├── instructions.py          ← build_instructions(ctx, agent) -> str
        ├── backends/
        │   ├── __init__.py
        │   ├── protocol.py          ← WorkspaceBackend ABC
        │   ├── filesystem.py        ← FilesystemBackend (default)
        │   └── memory_backend.py    ← InMemoryBackend (ephemeral, for tests)
        ├── middleware/
        │   ├── __init__.py
        │   ├── workspace.py         ← WorkspaceHooks(RunHooks) — auto-logs + auto-evicts
        │   ├── hitl.py              ← HumanInTheLoopHooks(RunHooks)
        │   └── _utils.py            ← shared helpers (preview generation, path sanitization)
        ├── tools/
        │   ├── __init__.py          ← WORKSPACE_TOOLS, PLANNING_TOOLS, MEMORY_TOOLS
        │   ├── workspace_tools.py   ← read_file, write_file, edit_file, list_files, append_to_file
        │   ├── planning_tools.py    ← write_todos, mark_done, read_todos
        │   └── memory_tools.py      ← update_memory, read_memory
        ├── sessions/
        │   ├── __init__.py
        │   └── factory.py           ← create_session(session_id, db_path) -> Session
        └── skills.py                ← SkillsLoader.discover(), SkillsLoader.format_for_prompt()
```

---

## Implementation Details

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "deepx"
dynamic = ["version"]
requires-python = ">=3.11"
dependencies = [
    "openai-agents>=0.0.19",
    "pydantic>=2.0",
    "aiosqlite>=0.19",
    "pyyaml>=6.0",
    "openinference-instrumentation-openai-agents>=0.1.0",
    "opentelemetry-exporter-otlp-proto-http>=1.0.0",
]

[project.optional-dependencies]
redis = ["openai-agents[redis]"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff", "pyright"]

[tool.hatch.version]
path = "src/deepx/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/deepx"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### `src/deepx/_version.py`

```python
__version__ = "0.1.0"
```

### `src/deepx/models.py`

```python
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timezone


class TodoStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"


class Todo(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    status: TodoStatus = TodoStatus.pending
    notes: str = ""


class Plan(BaseModel):
    session_id: str
    todos: list[Todo] = Field(default_factory=list)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def pending(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.pending]

    def in_progress(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.in_progress]

    def completed(self) -> list[Todo]:
        return [t for t in self.todos if t.status == TodoStatus.completed]


class ToolLog(BaseModel):
    call_id: str
    tool_name: str
    agent_name: str
    session_id: str
    timestamp: str
    input_preview: str
    output_preview: str
    output_chars: int
    saved_to: str | None = None
```

### `src/deepx/context.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from deepx.models import Plan


@dataclass
class AgentContext:
    session_id: str
    plan: Plan = field(init=False)
    memory: str = ""
    skills_info: str = ""

    def __post_init__(self) -> None:
        self.plan = Plan(session_id=self.session_id)
```

`AgentContext` is the single object passed to every `Runner.run()` call via the `context`
parameter. All tools, hooks, and the instructions callable read from and write to this object.

### `src/deepx/backends/protocol.py`

```python
from __future__ import annotations
import abc


class WorkspaceBackend(abc.ABC):
    @abc.abstractmethod
    def read(self, session_id: str, path: str) -> str | None: ...

    @abc.abstractmethod
    def write(self, session_id: str, path: str, content: str) -> None: ...

    @abc.abstractmethod
    def append(self, session_id: str, path: str, content: str) -> None: ...

    @abc.abstractmethod
    def exists(self, session_id: str, path: str) -> bool: ...

    @abc.abstractmethod
    def list_files(self, session_id: str, prefix: str = "") -> list[str]: ...

    @abc.abstractmethod
    def read_shared(self, path: str) -> str | None: ...

    @abc.abstractmethod
    def write_shared(self, path: str, content: str) -> None: ...

    @abc.abstractmethod
    def list_shared(self, prefix: str = "") -> list[str]: ...
```

`read`/`write`/`list_files` operate under `sessions/{session_id}/files/`.
`read_shared`/`write_shared`/`list_shared` operate under `memory/`.

### `src/deepx/backends/filesystem.py`

```python
from __future__ import annotations
import json
from pathlib import Path
from deepx.backends.protocol import WorkspaceBackend


class FilesystemBackend(WorkspaceBackend):
    def __init__(self, root: str | Path = ".deepx") -> None:
        self._root = Path(root)

    def _files_path(self, session_id: str, path: str) -> Path:
        return self._root / "sessions" / session_id / "files" / path

    def _tools_path(self, session_id: str, tool_name: str, call_id: str) -> Path:
        return self._root / "sessions" / session_id / "tools" / tool_name / f"{call_id}.json"

    def _plan_path(self, session_id: str) -> Path:
        return self._root / "sessions" / session_id / "plan.json"

    def _shared_path(self, path: str) -> Path:
        return self._root / "memory" / path

    def read(self, session_id: str, path: str) -> str | None:
        p = self._files_path(session_id, path)
        return p.read_text() if p.exists() else None

    def write(self, session_id: str, path: str, content: str) -> None:
        p = self._files_path(session_id, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def append(self, session_id: str, path: str, content: str) -> None:
        p = self._files_path(session_id, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content)

    def exists(self, session_id: str, path: str) -> bool:
        return self._files_path(session_id, path).exists()

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        base = self._root / "sessions" / session_id / "files"
        if not base.exists():
            return []
        results = [
            str(p.relative_to(base))
            for p in sorted(base.rglob("*"))
            if p.is_file() and str(p.relative_to(base)).startswith(prefix)
        ]
        return results

    def read_shared(self, path: str) -> str | None:
        p = self._shared_path(path)
        return p.read_text() if p.exists() else None

    def write_shared(self, path: str, content: str) -> None:
        p = self._shared_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def list_shared(self, prefix: str = "") -> list[str]:
        base = self._root / "memory"
        if not base.exists():
            return []
        return [
            str(p.relative_to(base))
            for p in sorted(base.rglob("*"))
            if p.is_file() and str(p.relative_to(base)).startswith(prefix)
        ]

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        tool_name = log_data["tool_name"]
        call_id = log_data["call_id"]
        p = self._tools_path(session_id, tool_name, call_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(log_data, indent=2))

    def save_plan(self, session_id: str, plan_json: str) -> None:
        p = self._plan_path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(plan_json)

    def load_plan(self, session_id: str) -> str | None:
        p = self._plan_path(session_id)
        return p.read_text() if p.exists() else None
```

### `src/deepx/backends/memory_backend.py`

```python
from __future__ import annotations
import json
from deepx.backends.protocol import WorkspaceBackend


class InMemoryBackend(WorkspaceBackend):
    def __init__(self) -> None:
        self._files: dict[str, dict[str, str]] = {}
        self._shared: dict[str, str] = {}
        self._tools: dict[str, list] = {}
        self._plans: dict[str, str] = {}

    def read(self, session_id: str, path: str) -> str | None:
        return self._files.get(session_id, {}).get(path)

    def write(self, session_id: str, path: str, content: str) -> None:
        self._files.setdefault(session_id, {})[path] = content

    def append(self, session_id: str, path: str, content: str) -> None:
        existing = self._files.get(session_id, {}).get(path, "")
        self._files.setdefault(session_id, {})[path] = existing + content

    def exists(self, session_id: str, path: str) -> bool:
        return path in self._files.get(session_id, {})

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        return sorted(
            k for k in self._files.get(session_id, {})
            if k.startswith(prefix)
        )

    def read_shared(self, path: str) -> str | None:
        return self._shared.get(path)

    def write_shared(self, path: str, content: str) -> None:
        self._shared[path] = content

    def list_shared(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._shared if k.startswith(prefix))

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        self._tools.setdefault(session_id, []).append(log_data)

    def save_plan(self, session_id: str, plan_json: str) -> None:
        self._plans[session_id] = plan_json

    def load_plan(self, session_id: str) -> str | None:
        return self._plans.get(session_id)
```

### `src/deepx/middleware/_utils.py`

```python
from __future__ import annotations


LARGE_OUTPUT_THRESHOLD = 80_000
PREVIEW_LINES = 30


def generate_preview(content: str, max_lines: int = PREVIEW_LINES) -> str:
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    preview_lines = lines[:max_lines]
    omitted = len(lines) - max_lines
    return "\n".join(preview_lines) + f"\n\n... [{omitted} more lines omitted]"


def sanitize_path_component(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:64]
```

### `src/deepx/middleware/workspace.py`

This is the most important middleware file. It is a `RunHooks` subclass that:
1. On `on_agent_start`: loads shared memory (AGENTS.md) and plan into `AgentContext`
2. On `on_tool_start`: logs tool input to `tools/{tool_name}/{call_id}.json`
3. On `on_tool_end`: logs tool output, if large saves to file and returns modified result

**Critical note**: `on_tool_end` in openai-agents receives the result string but CANNOT
modify what the agent sees. To intercept the result before the agent sees it, the
workspace tools themselves (`write_file`, etc.) must handle eviction. However, for tools
defined outside the framework (user-provided tools), the large output problem is handled
by instructing the agent in the base prompt.

For the framework's own tools, the tools themselves check size and save if large.
For external tools, `on_tool_end` saves to the workspace for observability and appends a
note to the step log — the agent can be told in the instructions to check `list_files`
after any large operation.

```python
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from agents import RunHooks, RunContextWrapper, Agent
from deepx.context import AgentContext
from deepx.middleware._utils import LARGE_OUTPUT_THRESHOLD, generate_preview, sanitize_path_component
from deepx.models import Plan, ToolLog


class WorkspaceHooks(RunHooks[AgentContext]):
    def __init__(self, backend) -> None:
        self._backend = backend

    async def on_agent_start(
        self, ctx: RunContextWrapper[AgentContext], agent: Agent
    ) -> None:
        if not ctx.context.memory:
            raw = self._backend.read_shared("AGENTS.md")
            if raw:
                ctx.context.memory = raw

        saved_plan = self._backend.load_plan(ctx.context.session_id)
        if saved_plan:
            ctx.context.plan = Plan.model_validate_json(saved_plan)

    async def on_tool_end(
        self, ctx: RunContextWrapper[AgentContext], agent: Agent, tool, result: str
    ) -> None:
        call_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()
        session_id = ctx.context.session_id
        tool_name = sanitize_path_component(tool.name)
        is_large = len(result) > LARGE_OUTPUT_THRESHOLD
        saved_to = None

        if is_large:
            file_path = f"{tool_name}_{call_id}.txt"
            self._backend.write(session_id, file_path, result)
            saved_to = file_path

        log = {
            "call_id": call_id,
            "tool_name": tool.name,
            "agent_name": agent.name,
            "session_id": session_id,
            "timestamp": timestamp,
            "output_chars": len(result),
            "output_preview": generate_preview(result, 10),
            "saved_to": saved_to,
        }
        self._backend.save_tool_log(session_id, log)
```

Note: The hooks `on_tool_start` with input logging would need access to the tool call
arguments. In the current openai-agents SDK, `on_tool_start` receives the `tool` object
but not the input arguments directly. Log what is available. Check openai-agents docs for
the current `on_tool_start` signature — if args are available, log them too.

### `src/deepx/middleware/hitl.py`

```python
from __future__ import annotations
from collections.abc import Callable
from agents import RunHooks, RunContextWrapper, Agent
from deepx.context import AgentContext


class HumanInTheLoopHooks(RunHooks[AgentContext]):
    def __init__(
        self,
        sensitive_tools: set[str],
        approval_fn: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._sensitive = sensitive_tools
        self._approval_fn = approval_fn or self._cli_approval

    @staticmethod
    def _cli_approval(agent_name: str, tool_name: str) -> bool:
        response = input(f"\n[HITL] Agent '{agent_name}' wants to call '{tool_name}'. Approve? [y/n]: ")
        return response.strip().lower() == "y"

    async def on_tool_start(
        self, ctx: RunContextWrapper[AgentContext], agent: Agent, tool
    ) -> None:
        if tool.name in self._sensitive:
            approved = self._approval_fn(agent.name, tool.name)
            if not approved:
                raise RuntimeError(f"Human rejected tool call: {tool.name}. Do not retry this tool.")
```

### `src/deepx/tools/planning_tools.py`

Tool docstrings here become the tool descriptions sent to the LLM. Make them clear and
instructive. Do not put implementation comments in the docstring — only the LLM instruction.

```python
from __future__ import annotations
from agents import function_tool, RunContextWrapper
from deepx.context import AgentContext
from deepx.models import Todo, TodoStatus


@function_tool
def write_todos(ctx: RunContextWrapper[AgentContext], todos: list[str]) -> str:
    """Replace the current plan with a new list of todo items. Call this FIRST before starting
    any multi-step task. Each item should be a clear, actionable step. All items start as pending.
    Returns a formatted list of the saved todos."""
    ctx.context.plan.todos = [Todo(title=t) for t in todos]
    _persist_plan(ctx)
    lines = [f"[{i+1}] ({t.status}) {t.title}" for i, t in enumerate(ctx.context.plan.todos)]
    return "Plan saved:\n" + "\n".join(lines)


@function_tool
def mark_done(ctx: RunContextWrapper[AgentContext], index: int) -> str:
    """Mark a todo item as completed by its 1-based index. Call this after finishing each step."""
    todos = ctx.context.plan.todos
    if index < 1 or index > len(todos):
        return f"Error: index {index} out of range. Plan has {len(todos)} items."
    todos[index - 1].status = TodoStatus.completed
    _persist_plan(ctx)
    return f"Marked done: {todos[index - 1].title}"


@function_tool
def read_todos(ctx: RunContextWrapper[AgentContext]) -> str:
    """Read the current plan and all todo items with their status."""
    todos = ctx.context.plan.todos
    if not todos:
        return "No plan yet. Call write_todos to create one."
    lines = [f"[{i+1}] ({t.status}) {t.title}" for i, t in enumerate(todos)]
    return "\n".join(lines)


def _persist_plan(ctx: RunContextWrapper[AgentContext]) -> None:
    from deepx.graph import _get_backend
    backend = _get_backend()
    backend.save_plan(ctx.context.session_id, ctx.context.plan.model_dump_json())
```

Note: `_get_backend()` needs to be accessible. The cleanest approach is to store the
backend on the `AgentContext` itself:

```python
@dataclass
class AgentContext:
    session_id: str
    backend: WorkspaceBackend
    plan: Plan = field(init=False)
    memory: str = ""
    skills_info: str = ""

    def __post_init__(self) -> None:
        self.plan = Plan(session_id=self.session_id)
```

Then all tools access `ctx.context.backend` directly. Use this pattern throughout.

### `src/deepx/tools/workspace_tools.py`

```python
from __future__ import annotations
import uuid
from agents import function_tool, RunContextWrapper
from deepx.context import AgentContext
from deepx.middleware._utils import LARGE_OUTPUT_THRESHOLD, generate_preview, sanitize_path_component


@function_tool
def write_file(ctx: RunContextWrapper[AgentContext], path: str, content: str) -> str:
    """Write content to a new file in the workspace. Path is relative (e.g. 'research/topic.md').
    Returns an error if the file already exists. Use append_to_file to add to an existing file.
    Always prefer writing findings to files rather than holding them in conversation."""
    if ctx.context.backend.exists(ctx.context.session_id, path):
        return f"Error: '{path}' already exists. Use edit_file to modify it."
    ctx.context.backend.write(ctx.context.session_id, path, content)
    return f"Written: {path} ({len(content)} chars)"


@function_tool
def read_file(ctx: RunContextWrapper[AgentContext], path: str, offset: int = 0, limit: int = 100) -> str:
    """Read a file from the workspace with pagination. Path is relative (e.g. 'research/topic.md').
    Use offset and limit to read large files in sections. Returns line-numbered content."""
    content = ctx.context.backend.read(ctx.context.session_id, path)
    if content is None:
        shared = ctx.context.backend.read_shared(path)
        if shared is None:
            return f"Error: '{path}' not found in workspace or shared memory."
        content = shared
    lines = content.splitlines()
    selected = lines[offset: offset + limit]
    if not selected:
        return f"Error: offset {offset} exceeds file length ({len(lines)} lines)."
    numbered = "\n".join(f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(selected))
    if len(lines) > offset + limit:
        numbered += f"\n\n[{len(lines) - offset - limit} more lines. Use offset={offset + limit} to continue.]"
    return numbered


@function_tool
def append_to_file(ctx: RunContextWrapper[AgentContext], path: str, content: str) -> str:
    """Append content to an existing file. Creates the file if it does not exist.
    Use this to accumulate research findings, logs, or any growing data."""
    ctx.context.backend.append(ctx.context.session_id, path, "\n" + content)
    return f"Appended to: {path}"


@function_tool
def edit_file(ctx: RunContextWrapper[AgentContext], path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing an exact string. The old_string must appear exactly once.
    Always read the file first to confirm the exact text before editing."""
    content = ctx.context.backend.read(ctx.context.session_id, path)
    if content is None:
        return f"Error: '{path}' not found."
    count = content.count(old_string)
    if count == 0:
        return f"Error: string not found in '{path}'."
    if count > 1:
        return f"Error: string appears {count} times. Provide more context to make it unique."
    ctx.context.backend.write(ctx.context.session_id, path, content.replace(old_string, new_string, 1))
    return f"Edited: {path}"


@function_tool
def list_files(ctx: RunContextWrapper[AgentContext], prefix: str = "") -> str:
    """List all files in the workspace for this session. Optionally filter by prefix.
    Returns a sorted list of relative file paths. Always call this before reading to discover files."""
    files = ctx.context.backend.list_files(ctx.context.session_id, prefix)
    if not files:
        return "No files found." + (f" (prefix='{prefix}')" if prefix else "")
    return "\n".join(files)
```

### `src/deepx/tools/memory_tools.py`

```python
from __future__ import annotations
from agents import function_tool, RunContextWrapper
from deepx.context import AgentContext


@function_tool
def update_memory(ctx: RunContextWrapper[AgentContext], note: str) -> str:
    """Add a persistent note to shared memory. This memory is loaded at the start of every session
    and is shared across all agents. Use for important facts, preferences, credentials, patterns,
    or anything that should persist across sessions. Do not store secrets or API keys."""
    ctx.context.memory = (ctx.context.memory or "") + f"\n- {note}"
    ctx.context.backend.write_shared("AGENTS.md", ctx.context.memory)
    return f"Memory updated: {note[:100]}"


@function_tool
def read_memory(ctx: RunContextWrapper[AgentContext]) -> str:
    """Read all notes from shared memory."""
    return ctx.context.memory or "No memory notes yet."
```

### `src/deepx/skills.py`

```python
from __future__ import annotations
import re
from pathlib import Path
from typing import TypedDict
import yaml


class SkillMetadata(TypedDict):
    name: str
    description: str
    path: str


class SkillsLoader:
    @staticmethod
    def discover(skills_root: str | Path) -> list[SkillMetadata]:
        root = Path(skills_root)
        if not root.exists():
            return []
        skills: list[SkillMetadata] = []
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text()
            metadata = SkillsLoader._parse_frontmatter(content, str(skill_md))
            if metadata:
                skills.append(metadata)
        return skills

    @staticmethod
    def _parse_frontmatter(content: str, path: str) -> SkillMetadata | None:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return None
        try:
            data = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
        if not isinstance(data, dict):
            return None
        name = str(data.get("name", "")).strip()
        description = str(data.get("description", "")).strip()
        if not name or not description:
            return None
        return SkillMetadata(name=name, description=description, path=path)

    @staticmethod
    def format_for_prompt(skills: list[SkillMetadata]) -> str:
        if not skills:
            return ""
        lines = []
        for skill in skills:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(f"  → Read `{skill['path']}` for full instructions")
        return "\n".join(lines)
```

### `src/deepx/sessions/factory.py`

```python
from __future__ import annotations
from agents import SQLiteSession
from agents.memory import OpenAIResponsesCompactionSession


def create_session(session_id: str, db_path: str | None = None):
    if db_path is None:
        from agents.memory import MemorySession
        return MemorySession()

    raw = SQLiteSession(session_id, db_path)
    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=raw,
    )
```

Note: `MemorySession` may not exist in all versions of openai-agents. Check the SDK.
If it does not exist, use `SQLiteSession(session_id)` without a `db_path` which defaults
to in-memory (`:memory:`).

### `src/deepx/observability.py`

```python
from __future__ import annotations
import base64
import os


def setup_observability() -> None:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        return

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{base_url}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

    try:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
        OpenAIAgentsInstrumentor().instrument()
    except ImportError:
        pass
```

### `src/deepx/instructions.py`

```python
from __future__ import annotations
from agents import RunContextWrapper, Agent
from deepx.context import AgentContext


BASE_PROMPT = """You are a deep autonomous agent capable of planning and executing complex,
multi-step tasks.

## Core Rules
- Call write_todos FIRST before starting any multi-step task. Plan before acting.
- Write findings and results to files using write_file. Never hold large content in conversation.
- Pass file paths to subagents, not raw content. Subagents read files themselves.
- When tasks are independent of each other, invoke multiple subagent tools in a single response.
- After each major step, mark it done with mark_done and verify the result.
- Use update_memory for facts that should persist across sessions.
- If a tool returns a file path instead of content, the output was large and was auto-saved.
  Use read_file with the path to access it.

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
```

### `src/deepx/graph.py`

This is the main entry point. Mirrors `graph.py` in langchain/deepagents.

```python
from __future__ import annotations
import asyncio
import os
from pathlib import Path
from collections.abc import Callable
from typing import Any

from agents import Agent, Runner, RunHooks
from agents.memory import MemorySession

from deepx._version import __version__
from deepx.context import AgentContext
from deepx.models import Plan
from deepx.backends.protocol import WorkspaceBackend
from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory_backend import InMemoryBackend
from deepx.instructions import build_instructions
from deepx.middleware.workspace import WorkspaceHooks
from deepx.observability import setup_observability
from deepx.sessions.factory import create_session
from deepx.skills import SkillsLoader
from deepx.tools import WORKSPACE_TOOLS, PLANNING_TOOLS, MEMORY_TOOLS


setup_observability()


def create_deep_agent(
    *,
    model: str = "gpt-4o",
    tools: list | None = None,
    subagents: list[tuple[Agent, str]] | None = None,
    system_prompt: str = "",
    skills_path: str | None = None,
    memory_path: str | None = None,
    workspace_path: str | None = None,
    db_path: str | None = None,
    max_turns: int = 200,
    hitl_hooks: RunHooks | None = None,
) -> "DeepAgent":
    workspace_root = workspace_path or os.getenv("DEEPX_WORKSPACE", ".deepx")
    backend: WorkspaceBackend = (
        FilesystemBackend(workspace_root) if workspace_root else InMemoryBackend()
    )

    if skills_path:
        skills = SkillsLoader.discover(skills_path)
        skills_info = SkillsLoader.format_for_prompt(skills)
    else:
        skills_info = ""

    if memory_path:
        mem_content = Path(memory_path).read_text() if Path(memory_path).exists() else ""
    else:
        mem_content = ""

    subagent_tools = []
    for sub_agent, description in (subagents or []):
        subagent_tools.append(_make_subagent_tool(sub_agent, description))

    base_tools = [*WORKSPACE_TOOLS, *PLANNING_TOOLS, *MEMORY_TOOLS]
    all_tools = base_tools + subagent_tools + (tools or [])

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_instructions(ctx, agent, custom_prompt=system_prompt)

    agent = Agent(
        name="orchestrator",
        instructions=instructions,
        model=model,
        tools=all_tools,
    )

    return DeepAgent(
        agent=agent,
        backend=backend,
        db_path=db_path,
        max_turns=max_turns,
        hitl_hooks=hitl_hooks,
        skills_info=skills_info,
        memory=mem_content,
    )


class DeepAgent:
    def __init__(
        self,
        agent: Agent,
        backend: WorkspaceBackend,
        db_path: str | None,
        max_turns: int,
        hitl_hooks: RunHooks | None,
        skills_info: str,
        memory: str,
    ) -> None:
        self._agent = agent
        self._backend = backend
        self._db_path = db_path
        self._max_turns = max_turns
        self._hitl_hooks = hitl_hooks
        self._skills_info = skills_info
        self._memory = memory

    async def run(
        self,
        task: str,
        *,
        session_id: str,
        resume: bool = False,
    ) -> "DeepRunResult":
        ctx = AgentContext(
            session_id=session_id,
            backend=self._backend,
        )
        ctx.memory = self._memory
        ctx.skills_info = self._skills_info

        if resume:
            saved_plan = self._backend.load_plan(session_id)
            if saved_plan:
                ctx.plan = Plan.model_validate_json(saved_plan)

        session = create_session(session_id, self._db_path)

        hooks_list: list[RunHooks] = [WorkspaceHooks(self._backend)]
        if self._hitl_hooks:
            hooks_list.append(self._hitl_hooks)

        combined_hooks = _CombinedHooks(hooks_list) if len(hooks_list) > 1 else hooks_list[0]

        result = await Runner.run(
            self._agent,
            input=task,
            context=ctx,
            session=session,
            hooks=combined_hooks,
            max_turns=self._max_turns,
        )

        return DeepRunResult(
            output=result.final_output,
            session_id=session_id,
            plan=ctx.plan,
        )

    def run_sync(self, task: str, *, session_id: str, resume: bool = False) -> "DeepRunResult":
        return asyncio.get_event_loop().run_until_complete(
            self.run(task, session_id=session_id, resume=resume)
        )


class DeepRunResult:
    def __init__(self, output: str, session_id: str, plan: Plan) -> None:
        self.output = output
        self.session_id = session_id
        self.plan = plan

    def __repr__(self) -> str:
        return f"DeepRunResult(session_id={self.session_id!r}, output={self.output[:100]!r})"


def _make_subagent_tool(sub_agent: Agent, description: str):
    from agents import function_tool

    @function_tool
    async def run_subagent(ctx: RunContextWrapper, task_description: str) -> str:
        result = await Runner.run(sub_agent, input=task_description, context=ctx.context)
        return result.final_output

    run_subagent.name = sub_agent.name
    run_subagent.__doc__ = description
    return run_subagent


class _CombinedHooks(RunHooks):
    def __init__(self, hooks: list[RunHooks]) -> None:
        self._hooks = hooks

    async def on_agent_start(self, ctx, agent) -> None:
        for h in self._hooks:
            await h.on_agent_start(ctx, agent)

    async def on_agent_end(self, ctx, agent, output) -> None:
        for h in self._hooks:
            await h.on_agent_end(ctx, agent, output)

    async def on_tool_start(self, ctx, agent, tool) -> None:
        for h in self._hooks:
            await h.on_tool_start(ctx, agent, tool)

    async def on_tool_end(self, ctx, agent, tool, result) -> None:
        for h in self._hooks:
            await h.on_tool_end(ctx, agent, tool, result)
```

### `src/deepx/tools/__init__.py`

```python
from deepx.tools.workspace_tools import (
    write_file,
    read_file,
    append_to_file,
    edit_file,
    list_files,
)
from deepx.tools.planning_tools import write_todos, mark_done, read_todos
from deepx.tools.memory_tools import update_memory, read_memory

WORKSPACE_TOOLS = [write_file, read_file, append_to_file, edit_file, list_files]
PLANNING_TOOLS = [write_todos, mark_done, read_todos]
MEMORY_TOOLS = [update_memory, read_memory]
```

### `src/deepx/__init__.py`

```python
from deepx._version import __version__
from deepx.graph import create_deep_agent, DeepAgent, DeepRunResult
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory_backend import InMemoryBackend

__all__ = [
    "__version__",
    "create_deep_agent",
    "DeepAgent",
    "DeepRunResult",
    "HumanInTheLoopHooks",
    "FilesystemBackend",
    "InMemoryBackend",
]
```

---

## Usage Example (text-to-sql agent style)

This is how a user builds an agent on top of `deepx`. Mirror the style of the
langchain/deepagents `text-to-sql-agent/agent.py` example.

```python
import os
import asyncio
from agents import Agent, function_tool
from deepx import create_deep_agent, HumanInTheLoopHooks


@function_tool
def sql_query(ctx, query: str) -> str:
    """Execute a read-only SQL SELECT query and return results as a formatted table."""
    ...


@function_tool
def web_search(ctx, query: str) -> str:
    """Search the web for current information. Returns a summary of top results."""
    ...


sql_agent = Agent(
    name="sql_agent",
    instructions="You write and execute SQL queries against a database.",
    tools=[sql_query],
)

agent = create_deep_agent(
    model="gpt-4o",
    tools=[web_search],
    subagents=[(sql_agent, "Query a relational database using natural language")],
    system_prompt="You are a research assistant that combines web and database information.",
    skills_path="./skills/",
    workspace_path=".deepx/",
    db_path="agent.db",
)

result = agent.run_sync(
    "Compare pricing from competitor websites with our internal database",
    session_id="research_001",
)
print(result.output)
```

---

## Skill Definition Format

Create a `skills/` folder alongside `agent.py`. Each skill is a subfolder with `SKILL.md`:

```
skills/
└── query-writing/
    └── SKILL.md
```

`SKILL.md` format:
```markdown
---
name: query-writing
description: For writing and executing SQL queries against relational databases
---

# Query Writing Skill

## When to Use
Use this skill when...

## Workflow
1. ...
```

The framework lists skill names and descriptions in the system prompt. The agent reads
the full `SKILL.md` via `read_file("skills/query-writing/SKILL.md")` when it decides
the skill is relevant.

---

## AGENTS.md Format

Create `memory/AGENTS.md` in the workspace root for persistent cross-session memory:

```markdown
# Agent Memory

- The user prefers concise responses
- The database is PostgreSQL running on localhost:5432
- Primary contact: user@example.com
```

The framework loads this at `on_agent_start` and injects it into the system prompt.

---

## Environment Variables

```
OPENAI_API_KEY          required — OpenAI API key
DEEPX_WORKSPACE         optional — workspace root path (default: .deepx/)
LANGFUSE_PUBLIC_KEY     optional — enables Langfuse tracing
LANGFUSE_SECRET_KEY     optional — enables Langfuse tracing
LANGFUSE_BASE_URL       optional — Langfuse endpoint (default: https://cloud.langfuse.com)
```

---

## Coding Standards

- No comments anywhere in the code
- Docstrings only on `@function_tool` decorated functions — these become LLM tool descriptions
- All other functions and classes: no docstrings, no comments
- Use `from __future__ import annotations` in all files
- Pydantic for all structured data models
- Type hints everywhere
- `async`/`await` for all I/O operations
- No shell/subprocess access — do not implement an execute/shell tool
- No LangChain, no LangGraph dependencies
- Keep files small and focused — one clear responsibility per file

---

## Implementation Order

1. `pyproject.toml` — package definition
2. `src/deepx/_version.py` — version string
3. `src/deepx/models.py` — Pydantic models (Plan, Todo, TodoStatus, ToolLog)
4. `src/deepx/backends/protocol.py` — WorkspaceBackend ABC
5. `src/deepx/backends/filesystem.py` — FilesystemBackend
6. `src/deepx/backends/memory_backend.py` — InMemoryBackend
7. `src/deepx/context.py` — AgentContext with backend field
8. `src/deepx/middleware/_utils.py` — shared helpers
9. `src/deepx/middleware/workspace.py` — WorkspaceHooks
10. `src/deepx/middleware/hitl.py` — HumanInTheLoopHooks
11. `src/deepx/tools/planning_tools.py` — write_todos, mark_done, read_todos
12. `src/deepx/tools/workspace_tools.py` — read_file, write_file, etc.
13. `src/deepx/tools/memory_tools.py` — update_memory, read_memory
14. `src/deepx/tools/__init__.py` — tool group exports
15. `src/deepx/skills.py` — SkillsLoader
16. `src/deepx/sessions/factory.py` — create_session
17. `src/deepx/observability.py` — setup_observability
18. `src/deepx/instructions.py` — build_instructions
19. `src/deepx/graph.py` — create_deep_agent, DeepAgent, _make_subagent_tool
20. `src/deepx/__init__.py` — public API exports
21. `README.md` — usage documentation

---

## Key Implementation Notes for the Coding Agent

1. **`on_tool_start` args**: Check the current `openai-agents` SDK source for whether
   `on_tool_start(ctx, agent, tool)` includes the tool call arguments. If it does, log them.
   If not, only log `tool.name`. Do not guess.

2. **`_make_subagent_tool`**: The `@function_tool` decorator infers the tool name from
   the function name. Since we need the name to be the agent's name dynamically, check how
   `function_tool` handles renaming. The `name` attribute on the returned tool may be
   settable directly, or you may need to use `functools.wraps` or a closure pattern. Check
   the openai-agents SDK source before implementing.

3. **`MemorySession` availability**: The `openai-agents` SDK may export `MemorySession`
   from `agents` or from `agents.memory`. Check the SDK before writing the import.
   The default in-memory SQLite is `SQLiteSession(session_id)` with no db_path — it defaults
   to `:memory:`. Use that as the fallback if `MemorySession` does not exist.

4. **`OpenAIResponsesCompactionSession`**: Imported from `agents.memory`. Wrap it around
   any session. Default trigger is ~10 non-user items. This is fine for production.

5. **`RunContextWrapper` type parameter**: When writing hooks and tools, the type parameter
   is `AgentContext`. Use `RunContextWrapper[AgentContext]` everywhere for proper typing.

6. **Tool docstring → LLM description**: In openai-agents, the `@function_tool` decorator
   uses the function's `__doc__` (docstring) as the tool description sent to the LLM.
   This is the ONLY place docstrings belong. Make them clear, instructive, and concise.

7. **Backend on AgentContext**: The `backend` field on `AgentContext` means all tools
   have access to the workspace backend through `ctx.context.backend`. This avoids global
   state. Every tool function accesses `ctx.context.backend` directly.

8. **`_CombinedHooks`**: The openai-agents SDK `Runner.run()` takes a single `hooks`
   parameter. To combine multiple hooks objects, implement `_CombinedHooks` as shown above.
   Verify this is still needed in the current SDK — newer versions may support a list.

9. **Langfuse**: `OpenAIAgentsInstrumentor().instrument()` must be called ONCE at process
   startup, before any `Agent` or `Runner` is constructed. That's why it's in
   `setup_observability()` which is called at module import time in `graph.py`.
   The instrumentation is automatic — every `Runner.run()` call, every tool call, every
   handoff is automatically traced. No further action needed.

10. **Tool I/O observability limitation**: The `WorkspaceHooks.on_tool_end` receives the
    result AFTER it has been returned to the agent. It cannot modify the result. True
    interception (modifying what the agent sees) requires wrapping at the tool level.
    For the framework's own tools, handle size checks inside the tool function itself.
    For user-provided tools, the `on_tool_end` saves to workspace for observability,
    and the agent is instructed via the base prompt to check `list_files` if a result
    seems truncated. This is an acceptable tradeoff for framework simplicity.