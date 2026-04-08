---
./README.md
```markdown
# deepx — Agent Harness Built on OpenAI Agents SDK

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

## Quick Start

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

## Installation

```bash
pip install deepx
```

### Optional Dependencies

```bash
# For Redis session storage
pip install deepx[redis]

# For development
pip install deepx[dev]
```

## Configuration

### Environment Variables

```
OPENAI_API_KEY          required — OpenAI API key
DEEPX_WORKSPACE         optional — workspace root path (default: .deepx/)
LANGFUSE_PUBLIC_KEY     optional — enables Langfuse tracing
LANGFUSE_SECRET_KEY     optional — enables Langfuse tracing
LANGFUSE_BASE_URL       optional — Langfuse endpoint (default: https://cloud.langfuse.com)
```

### Workspace Structure

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

## Skills

Create a `skills/` folder alongside your agent code. Each skill is a subfolder with `SKILL.md`:

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

## Shared Memory

Create `memory/AGENTS.md` in the workspace root for persistent cross-session memory:

```markdown
# Agent Memory

- The user prefers concise responses
- The database is PostgreSQL running on localhost:5432
- Primary contact: user@example.com
```

## Human-in-the-Loop

```python
from deepx import HumanInTheLoopHooks

hitl = HumanInTheLoopHooks(
    sensitive_tools={"execute_payment", "delete_data"},
    approval_fn=lambda agent, tool: input(f"Allow {agent} to call {tool}? [y/n]: ").lower() == "y"
)

agent = create_deep_agent(
    # ... other args
    hitl_hooks=hitl,
)
```

## API Reference

### create_deep_agent

Main entry point for creating agents.

```python
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
) -> DeepAgent
```

### DeepAgent

The main agent class.

```python
class DeepAgent:
    async def run(self, task: str, *, session_id: str, resume: bool = False) -> DeepRunResult
    def run_sync(self, task: str, *, session_id: str, resume: bool = False) -> DeepRunResult
```

### DeepRunResult

Result of running an agent.

```python
class DeepRunResult:
    output: str
    session_id: str
    plan: Plan
```

## Built-in Tools

### Planning Tools
- `write_todos(todos: list[str])` - Create a new plan
- `mark_done(index: int)` - Mark a todo as completed
- `read_todos()` - Read current plan status

### Workspace Tools
- `write_file(path: str, content: str)` - Write a new file
- `read_file(path: str, offset: int = 0, limit: int = 100)` - Read file with pagination
- `edit_file(path: str, old_string: str, new_string: str)` - Edit file by string replacement
- `append_to_file(path: str, content: str)` - Append to existing file
- `list_files(prefix: str = "")` - List files in workspace

### Memory Tools
- `update_memory(note: str)` - Add persistent note to shared memory
- `read_memory()` - Read all shared memory notes

## Backend Options

### FilesystemBackend (default)
Persistent storage using the filesystem. Use for production.

```python
from deepx import FilesystemBackend

agent = create_deep_agent(
    workspace_path="/path/to/workspace",
)
```

### InMemoryBackend
Ephemeral storage in memory. Use for testing.

```python
from deepx import InMemoryBackend

agent = create_deep_agent(
    workspace_path="",  # Empty string triggers InMemoryBackend
)
```

## Observability

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, deepx automatically enables Langfuse tracing via OpenTelemetry. No additional configuration needed.

## Development

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Type checking
pyright src/

# Linting
ruff check src/
```

## License

MIT
```
---
./old_plan.md
```markdown
# deepx v2 — Implementation Instructions

---

## 0. Immediate bug fixes

### 0.1 Type error on `subagents`
`factory.py` — change the signature:
```python
# from
subagents: list[SubAgentDict | tuple[Agent, str]] | None = None,
# to
subagents: list[dict | tuple[Agent, str]] | None = None,
```
Keep `SubAgentDict` as a `TypedDict` for documentation only. Never use it in function signatures.

### 0.2 HITL not firing
`factory.py` — `DeepAgentRunner._make_hooks()`:
```python
def _make_hooks(self) -> RunHooksBase:
    hooks = [WorkspaceHooks(self._backend, self._debug)]
    if self._hitl:
        hooks.append(self._hitl)
    return _CombinedHooks(hooks) if len(hooks) > 1 else hooks[0]
```
Also confirm `approved_tools: set[str] = field(default_factory=set)` is in `AgentContext.__post_init__`.

### 0.3 Rename tool files — remove `_tools` suffix
- `tools/planning_tools.py` → `tools/planning.py`
- `tools/workspace_tools.py` → `tools/workspace.py`
- `tools/memory_tools.py` → `tools/memory.py`
- Fix all imports in `factory.py` and `system_prompt.py`.

---

## 1. File renames and moves

| Old | New | Reason |
|---|---|---|
| `backends/memory_backend.py` | `backends/memory.py` | Remove redundant `_backend` suffix |
| `middleware/sessions.py` | `sessions.py` (package root) | Not middleware — it's an SDK helper |
| `instructions.py` | `system_prompt.py` (package root) | Name reflects what it does |
| `middleware/skills.py` | Absorbed into `system_prompt.py` | Skills discovery only serves prompt assembly |

**Class names stay the same.** `InMemoryBackend` stays `InMemoryBackend`. No `EphemeralBackend` alias.

**Delete** `middleware/skills.py` after merging its logic into `system_prompt.py`.

---

## 2. `.deepx` folder layout

```
.deepx/
├── memory/
│   └── AGENTS.md               ← global agent memory, loaded via memory= parameter
│
└── sessions/
    └── {session_id}/
        ├── files/              ← shared VFS: all agents in this session read/write here
        │   ├── research/
        │   ├── output/
        │   └── ...
        ├── plans/              ← per-agent plan isolation
        │   ├── orchestrator.json
        │   └── {subagent_name}.json
        └── logs/               ← only written when debug=True, never listed by ls
            ├── tasks.md        ← append-only: one entry per run() call
            ├── plans.json      ← append-only JSON array: every write_todos call
            └── tools/          ← one JSON file per tool call
                └── {tool_name}/
                    └── {call_id}.json
```

**Why `files/` is shared:** In langchain/deepagents, the `StateBackend` is shared across all agents in a session. All agents read/write the same flat namespace. Files are the handoff mechanism between agents. We mirror this exactly.

**Why `plans/` is per-agent:** In langchain/deepagents source (`subagents.py`), `_EXCLUDED_STATE_KEYS = {"messages", "todos", ...}` — todos are explicitly excluded when returning subagent results to the parent. Each agent owns its own plan. The orchestrator plans at the high level; subagents plan their own subtasks independently.

**Path routing in tools:** The agent uses paths like `write_file("/research/ollama.md", ...)`. The backend stores it at `sessions/{id}/files/research/ollama.md`. The `store/` prefix routes to `.deepx/memory/`. No other path prefixes.

| Tool path | Backend stores at |
|---|---|
| `/research/ollama.md` | `sessions/{id}/files/research/ollama.md` |
| `/store/AGENTS.md` | `.deepx/memory/AGENTS.md` |

---

## 3. Backend protocol — `backends/protocol.py`

```python
class WorkspaceBackend(abc.ABC):

    # Session-scoped files (shared VFS — what ls/read_file/write_file use)
    def read(self, session_id: str, path: str) -> str | None: ...
    def write(self, session_id: str, path: str, content: str) -> None: ...
    def append(self, session_id: str, path: str, content: str) -> None: ...
    def exists(self, session_id: str, path: str) -> bool: ...
    def list_files(self, session_id: str, prefix: str = "") -> list[str]: ...

    # Cross-session memory store (store/ prefix in tool paths)
    def read_store(self, path: str) -> str | None: ...
    def write_store(self, path: str, content: str) -> None: ...
    def list_store(self, prefix: str = "") -> list[str]: ...

    # Per-agent plan persistence
    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None: ...
    def load_plan(self, session_id: str, agent_name: str) -> str | None: ...

    # Debug logging — only called when debug=True
    def append_task_log(self, session_id: str, task: str) -> None: ...
    def append_plan_log(self, session_id: str, entry_json: str) -> None: ...
    def save_tool_log(self, session_id: str, log_data: dict) -> None: ...

    # Execution support (default: not available)
    @property
    def supports_execution(self) -> bool:
        return False

    def execute(self, command: str) -> str:
        raise NotImplementedError("Shell execution requires a sandbox backend.")
```

### `FilesystemBackend` path mappings

```
sessions/{id}/files/{path}                      ← read / write / list_files
sessions/{id}/plans/{agent_name}.json           ← save_plan / load_plan
sessions/{id}/logs/tasks.md                     ← append_task_log
sessions/{id}/logs/plans.json                   ← append_plan_log
sessions/{id}/logs/tools/{tool_name}/{id}.json  ← save_tool_log
.deepx/memory/{path}                            ← read_store / write_store
```

### `InMemoryBackend` (`backends/memory.py`)

Same interface, nested dicts internally. Class name stays `InMemoryBackend`. No alias.

### `CompositeBackend` (`backends/composite.py`)

```python
class CompositeBackend(WorkspaceBackend):
    """Routes file operations to different backends by path prefix.

    Usage:
        backend = CompositeBackend(
            default=InMemoryBackend(),
            routes={"/store/": FilesystemBackend(".deepx")}
        )
    """
    def __init__(self, default: WorkspaceBackend, routes: dict[str, WorkspaceBackend]):
        self._default = default
        # Sort routes longest-first for correct prefix matching
        self._routes = sorted(routes.items(), key=lambda x: -len(x[0]))

    def _route(self, path: str) -> tuple[WorkspaceBackend, str]:
        for prefix, backend in self._routes:
            if path.startswith(prefix):
                return backend, path[len(prefix):]
        return self._default, path
```

All `WorkspaceBackend` methods delegate via `_route(path)`.

---

## 4. Context and models

### `context.py`

```python
@dataclass
class AgentContext:
    session_id: str
    agent_name: str              # identifies which plan file to use
    backend: WorkspaceBackend
    plan: Plan = field(init=False)
    memory: str = ""             # loaded AGENTS.md content
    skills_info: str = ""        # formatted skill frontmatter
    approved_tools: set[str] = field(default_factory=set)
    debug: bool = False

    def __post_init__(self) -> None:
        self.plan = Plan(session_id=self.session_id, agent_name=self.agent_name)
```

### `models.py`

```python
class Plan(BaseModel):
    session_id: str
    agent_name: str
    tasks: list[str] = Field(default_factory=list)   # accumulates across run() calls
    todos: list[Todo] = Field(default_factory=list)  # replaced on each write_todos
    updated_at: str = ...
```

---

## 5. `system_prompt.py` (replaces `instructions.py`, absorbs `middleware/skills.py`)

This file does three things:
1. Defines all prompt string constants
2. Contains skill discovery logic (moved from `middleware/skills.py`)
3. Assembles the full system prompt via `build_system_prompt()`

### Skill discovery (from langchain/deepagents `skills.py`)

Skills use **progressive disclosure** — only frontmatter is loaded at startup, not the full file body. The agent reads the full `SKILL.md` on-demand using `read_file` when it decides the skill is relevant.

```python
class SkillMetadata(TypedDict):
    name: str
    description: str
    path: str          # absolute path to the SKILL.md file
    license: str | None
    compatibility: str | None
    allowed_tools: list[str]

def discover_skills(paths: list[str]) -> list[SkillMetadata]:
    """Scan each path for SKILL.md files. Parse only YAML frontmatter.
    
    Expects structure:
        {path}/
        └── {skill-name}/
            └── SKILL.md
    
    Later paths override earlier ones for skills with the same name.
    Skills can live anywhere — inside .deepx/ or in the project directory.
    """

def format_skills_for_prompt(skills: list[SkillMetadata]) -> str:
    """Format skill list for system prompt injection.
    
    Output format matching langchain/deepagents:
    - **{name}**: {description}
      -> Read `{path}` for full instructions
    """
```

### Prompt constants

All prompt strings should closely follow the wording used in langchain/deepagents source and documentation. Specifically:

**`BASE_AGENT_PROMPT`** — use the same text as `BASE_AGENT_PROMPT` in langchain/deepagents `graph.py` and `base_prompt.md`, adapted to openai-agents SDK context. Do not invent new wording.

**`TODO_PROMPT`** — instructions for `write_todos`, `mark_done`, `read_todos`. Follow the TodoListMiddleware prompt style from langchain/deepagents.

**`FILESYSTEM_PROMPT`** — instructions for `ls`, `read_file`, `write_file`, `edit_file`, `append_to_file`, `execute`. Use the `FILESYSTEM_SYSTEM_PROMPT` and `EXECUTION_SYSTEM_PROMPT` wording from `middleware/filesystem.py` in langchain/deepagents as the reference.

**`TASK_PROMPT`** — instructions for the `task` tool (subagent spawner). Use `TASK_SYSTEM_PROMPT` from `middleware/subagents.py` in langchain/deepagents as the reference.

**`MEMORY_PROMPT_TEMPLATE`** — wraps the loaded `AGENTS.md` content. Use the `MEMORY_SYSTEM_PROMPT` from `middleware/memory.py` in langchain/deepagents including the `<agent_memory>` tags and `<memory_guidelines>` section.

**`SKILLS_SYSTEM_PROMPT`** — instructions for using skills (progressive disclosure). Use `SKILLS_SYSTEM_PROMPT` from `middleware/skills.py` in langchain/deepagents as reference.

### `build_system_prompt()` — section order

Match langchain/deepagents prompt assembly order exactly (from `context-engineering` docs):

```python
def build_system_prompt(
    ctx: RunContextWrapper[AgentContext],
    agent: Agent,
    custom_prompt: str = "",
) -> str:
    sections = []

    # 1. User's custom system_prompt (prepended)
    if custom_prompt:
        sections.append(custom_prompt)

    # 2. BASE_AGENT_PROMPT
    sections.append(BASE_AGENT_PROMPT)

    # 3. Todo list prompt
    sections.append(TODO_PROMPT)

    # 4. Memory (AGENTS.md content) — only if memory= was provided
    if ctx.context.memory:
        sections.append(MEMORY_PROMPT_TEMPLATE.format(agent_memory=ctx.context.memory))

    # 5. Skills — only if skills= was provided
    if ctx.context.skills_info:
        sections.append(SKILLS_SYSTEM_PROMPT.format(...))

    # 6. Filesystem tool prompt (always)
    sections.append(FILESYSTEM_PROMPT)

    # 7. Task tool prompt (always — subagent delegation)
    sections.append(TASK_PROMPT)

    # 8. HITL prompt — only if interrupt_on= is set
    if ctx.context.hitl_tools:
        sections.append(HITL_PROMPT.format(tools=ctx.context.hitl_tools))

    # 9. Current plan todos (if any exist)
    if ctx.context.plan.todos:
        lines = [f"[{i+1}] ({t.status.value}) {t.title}"
                 for i, t in enumerate(ctx.context.plan.todos)]
        sections.append("## Current Plan\n" + "\n".join(lines))

    # 10. Current file listing
    files = ctx.context.backend.list_files(ctx.context.session_id)
    if files:
        shown = files[:50]
        block = "\n".join(shown)
        if len(files) > 50:
            block += f"\n... and {len(files) - 50} more. Use ls with a prefix to filter."
        sections.append(f"## Workspace Files\n{block}")

    return "\n\n---\n\n".join(sections)
```

---

## 6. Tools

### `tools/workspace.py`

Tool names must match langchain/deepagents exactly: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`. Tool descriptions must mirror or closely follow the descriptions in `middleware/filesystem.py` of langchain/deepagents (`LIST_FILES_TOOL_DESCRIPTION`, `READ_FILE_TOOL_DESCRIPTION`, etc.).

**`ls(path: str = "/") -> str`**
Lists the session's `files/` directory. If path starts with `/store/`, lists the memory store. Output includes file type and size:
```
research/              DIR
research/ollama.md     2.1 KB    2026-04-06 10:24
```

**`read_file(path: str, offset: int = 0, limit: int = 100) -> str`**
Path `/store/X` → `backend.read_store(X)`, else → `backend.read(session_id, X)`.
Returns line-numbered output (`cat -n` format):
```
     1  # Heading
     2
     3  Content here
```
If the file is an image (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`), return it as a base64 multimodal content block — same as langchain/deepagents `FilesystemMiddleware._create_read_file_tool()`.
Pagination footer when truncated: `[N more lines — use offset=M to continue]`

**`write_file(path: str, content: str) -> str`**
Path routing as above. Error if file already exists (exact wording from langchain/deepagents: `"Cannot write to {path} because it already exists. Read and then make an edit, or write to a new path."`).

**`edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str`**
Path routing. If `replace_all=False` and old_string appears more than once: error. Exact error wording from langchain/deepagents: `"Error: String '{old_string}' appears {n} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."`

**`append_to_file(path: str, content: str) -> str`**
Path routing. Creates if not exists.

**`glob(pattern: str, path: str = "/") -> str`**
Find files matching a glob pattern. Supports `*`, `**`, `?`. Returns matching paths.

**`grep(pattern: str, path: str | None = None, glob: str | None = None, output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches") -> str`**
Literal text search (not regex). Same three output modes as langchain/deepagents.

**`list_files(path: str = "/") -> str`** — deprecated alias calling `ls`. Keep for backward compat.

**`execute(command: str) -> str`**
Always registered. Returns `"Shell execution is not available. Use a sandbox backend to enable it."` if `backend.supports_execution` is False.

### `tools/planning.py`

Tool descriptions should follow the TodoListMiddleware prompt style from langchain/deepagents.

**`write_todos(todos: list[str]) -> str`**
Replaces `ctx.context.plan.todos`. Saves plan via `backend.save_plan(session_id, agent_name, plan_json)`.
When `ctx.context.debug` is True: also calls `backend.append_plan_log(session_id, entry_json)`.
Log entry: `{"timestamp": "...", "agent": "...", "todos": [...]}`
The `logs/plans.json` file is a JSON array — each call appends one object.

**`mark_done(index: int) -> str`** — marks todo at 1-based index as completed.

**`read_todos() -> str`** — returns current plan todos with statuses.

### `tools/memory.py`

**`update_memory(note: str) -> str`**
Appends `\n- {note}` to `ctx.context.memory`. Writes full updated content to `store/AGENTS.md` via `backend.write_store("AGENTS.md", ...)`.

**`read_memory() -> str`** — returns `ctx.context.memory`.

**`read_store(path: str) -> str`** — reads from cross-session memory store.

**`write_store(path: str, content: str) -> str`** — writes to cross-session memory store.

### `tools/__init__.py`

```python
from deepx.tools.workspace import ls, read_file, write_file, edit_file, append_to_file, glob, grep, list_files, execute
from deepx.tools.planning import write_todos, mark_done, read_todos
from deepx.tools.memory import update_memory, read_memory, read_store, write_store

WORKSPACE_TOOLS = [ls, read_file, write_file, edit_file, append_to_file, glob, grep, list_files, execute]
PLANNING_TOOLS = [write_todos, mark_done, read_todos]
MEMORY_TOOLS = [update_memory, read_memory, read_store, write_store]
```

---

## 7. `WorkspaceHooks` — `middleware/workspace.py`

### Fix duplicate logging
`on_tool_end` handles ONLY large-output eviction (saves content to file, replaces return value with path pointer + preview). It NEVER writes tool log JSON.

`wrap_tools_for_logging` is the ONLY logger. Only called when `debug=True`.

### `on_agent_start` must set `agent_name`

```python
async def on_agent_start(self, context: AgentHookContext, agent: Agent) -> None:
    context.context.agent_name = agent.name
    saved = self._backend.load_plan(context.context.session_id, agent.name)
    if saved:
        context.context.plan = Plan.model_validate_json(saved)
    if not context.context.memory:
        raw = self._backend.read_store("AGENTS.md")
        if raw:
            context.context.memory = raw
```

### Tool log format (written when `debug=True`)

Written to `sessions/{id}/logs/tools/{tool_name}/{call_id}.json`:
```json
{
  "call_id": "abc123",
  "tool_name": "web_search",
  "agent_name": "researcher",
  "session_id": "research_001",
  "timestamp": "2026-04-06T10:23:11Z",
  "input": {"query": "Ollama GPU support"},
  "output_chars": 4200,
  "output": "..."
}
```

### Large-output eviction

When a tool returns more than ~80,000 characters:
1. Write content to `files/large_tool_results/{call_id}.txt`
2. Replace the return value with: `"[Output was large and saved to {path}. Use read_file to access it. Preview:\n{first_10_lines}]"`

This matches langchain/deepagents' `FilesystemMiddleware.wrap_tool_call()` eviction behavior.

---

## 8. `create_deep_agent()` — `factory.py`

```python
def create_deep_agent(
    model: str = "gpt-4o-mini",
    tools: list | None = None,
    *,
    name: str = "orchestrator",
    system_prompt: str = "",
    subagents: list[dict | tuple[Agent, str]] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: type | None = None,
    backend: WorkspaceBackend | None = None,
    db_path: str = ":memory:",
    interrupt_on: list[str] | None = None,
    debug: bool = False,
    max_turns: int = 1000,
) -> "DeepAgentRunner":
```

Parameter notes:

**`memory`** — list of file paths to `AGENTS.md` files. Read each file at agent creation time using `backend.read_store(path)` or direct filesystem read. Concatenate all into a single string stored in `AgentContext.memory`. This is injected into the system prompt on every turn via `MEMORY_PROMPT_TEMPLATE`. Mirrors `MemoryMiddleware.before_agent()` which calls `download_files()` on all sources.

**`skills`** — list of directory paths. Each directory is scanned for `{skill-name}/SKILL.md` subdirectories. Only the YAML frontmatter is parsed (same as `_list_skills()` in langchain/deepagents `skills.py`). The full `SKILL.md` body is NOT loaded at startup. Skills can live inside `.deepx/` or anywhere in the project. Discovery happens at agent creation time and the formatted frontmatter goes into `AgentContext.skills_info`.

**`backend`** — defaults to `FilesystemBackend(".deepx")` when None.

**`response_format`** — passed as `output_type` to the `Agent`.

**`max_turns=1000`** — matches langchain/deepagents `recursion_limit=1000`.

### Building subagents — `_build_subagent()`

Every subagent gets the full base tools: `WORKSPACE_TOOLS + PLANNING_TOOLS + MEMORY_TOOLS`. This mirrors how langchain/deepagents applies `TodoListMiddleware` and `FilesystemMiddleware` to every subagent.

Custom subagents do NOT inherit the parent's user tools (same as langchain/deepagents — custom subagents declare their own `tools`).

The general-purpose subagent inherits parent's user tools and skills. It is auto-added unless the user provides a subagent named `"general-purpose"`.

```python
def _build_subagent(spec: dict, default_model: str, base_tools: list, user_tools: list) -> Agent:
    sub_tools = base_tools + list(spec.get("tools", []))   # custom: no parent tools
    sub_model = spec.get("model", default_model)
    sub_skills = discover_skills(spec.get("skills", []))
    sub_skills_info = format_skills_for_prompt(sub_skills)

    def instructions(ctx, agent):
        return build_system_prompt(ctx, agent, custom_prompt=spec.get("system_prompt", ""))

    return Agent(name=spec["name"], instructions=instructions, model=sub_model, tools=sub_tools)
```

General-purpose subagent spec:
```python
gp_spec = {
    "name": "general-purpose",
    "description": "General-purpose agent for isolated multi-step tasks. Has access to all the same tools as the main agent.",
    "system_prompt": "",   # uses BASE_AGENT_PROMPT from build_system_prompt
    "tools": user_tools,   # inherits parent's user tools
    "skills": skills or [],  # inherits parent's skills
}
```

### The `task` tool

Rename `spawn_task` → `task` to match langchain/deepagents.

Built via `agent.as_tool()` from the openai-agents SDK. The orchestrator calls:
```
task(subagent_type="researcher", description="Research Ollama GPU support...")
```

The subagent runs with an isolated context window. Only `final_output` is returned to the orchestrator. The full message history of the subagent is never exposed to the parent.

---

## 9. `DeepAgentRunner`

```python
async def run(
    self,
    task: str,
    *,
    session_id: str | None = None,
    resume: bool = False,
) -> DeepRunResult:
    sid = session_id or uuid.uuid4().hex
    ctx = self._make_ctx(sid, resume)
    ctx.plan.tasks.append(task)

    if self._debug:
        self._backend.append_task_log(sid, task)

    session = create_session(sid, self._db_path)
    agent = self._make_agent(sid)
    hooks = self._make_hooks()

    result = await Runner.run(
        agent, input=task,
        context=ctx, session=session,
        hooks=hooks, max_turns=self._max_turns,
    )
    return DeepRunResult(output=result.final_output, session_id=sid, plan=ctx.plan)

def run_sync(self, task: str, *, session_id=None, resume=False) -> DeepRunResult:
    """Synchronous wrapper — safe to call from scripts."""
    ...

async def run_stream(self, task: str, *, session_id=None, resume=False):
    """Async generator yielding StreamEvent objects.
    
    Uses Runner.run_streamed() from openai-agents SDK.
    
    Usage:
        async for event in agent.run_stream(task, session_id="x"):
            if event.type == "raw_response_event":
                print(event.data, end="", flush=True)
    """
    sid = session_id or uuid.uuid4().hex
    ctx = self._make_ctx(sid, resume)
    ctx.plan.tasks.append(task)

    if self._debug:
        self._backend.append_task_log(sid, task)

    session = create_session(sid, self._db_path)
    agent = self._make_agent(sid)
    hooks = self._make_hooks()

    stream = Runner.run_streamed(
        agent, input=task,
        context=ctx, session=session,
        hooks=hooks, max_turns=self._max_turns,
    )
    async for event in stream.stream_events():
        yield event
```

---

## 10. Observability — `middleware/observability.py`

Keep in `middleware/` since it integrates with the SDK's trace processor. Called from `create_deep_agent()`.

```python
def setup_observability() -> None:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "deepx")
    from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
    set_trace_processors([OpenAIAgentsTracingProcessor()])
```

---

## 11. `__init__.py` exports

```python
from deepx.factory import create_deep_agent, DeepAgentRunner, DeepRunResult
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory import InMemoryBackend
from deepx.backends.composite import CompositeBackend

DeepAgent = create_deep_agent   # alias so users can call DeepAgent(model=...)

__all__ = [
    "create_deep_agent", "DeepAgent",
    "DeepAgentRunner", "DeepRunResult",
    "HumanInTheLoopHooks",
    "FilesystemBackend", "InMemoryBackend", "CompositeBackend",
]
```

---

## 12. Final file tree

```
src/deepx/
├── __init__.py
├── factory.py          # create_deep_agent, DeepAgentRunner, DeepRunResult
├── context.py          # AgentContext (agent_name, debug)
├── models.py           # Plan (agent_name, tasks), Todo, TodoStatus
├── system_prompt.py    # BASE_AGENT_PROMPT, all prompt constants, discover_skills,
│                       # format_skills_for_prompt, build_system_prompt
├── sessions.py         # create_session (moved from middleware/)
├── _version.py
│
├── backends/
│   ├── __init__.py     # WorkspaceBackend, FilesystemBackend, InMemoryBackend, CompositeBackend
│   ├── protocol.py     # WorkspaceBackend ABC
│   ├── filesystem.py   # FilesystemBackend
│   ├── memory.py       # InMemoryBackend (renamed from memory_backend.py)
│   └── composite.py    # CompositeBackend (new)
│
├── middleware/
│   ├── __init__.py
│   ├── observability.py  # setup_observability() — LangSmith
│   ├── workspace.py      # WorkspaceHooks, wrap_tools_for_logging
│   └── hitl.py           # HumanInTheLoopHooks (interrupt_on naming)
│
└── tools/
    ├── __init__.py
    ├── planning.py         # write_todos, mark_done, read_todos
    ├── workspace.py        # ls, read_file, write_file, edit_file, append_to_file,
    │                       # glob, grep, list_files (deprecated), execute
    └── memory.py           # update_memory, read_memory, read_store, write_store
```

---

## 13. Implementation order

| # | Task | Files |
|---|---|---|
| 1 | Fix `subagents` type annotation | `factory.py` |
| 2 | Fix HITL `_make_hooks()` | `factory.py`, `middleware/hitl.py` |
| 3 | Rename tool files, fix all imports | `tools/` |
| 4 | Update `models.py` — `agent_name`, `tasks` on `Plan` | `models.py` |
| 5 | Update `context.py` — `agent_name`, `debug`, `hitl_tools` fields | `context.py` |
| 6 | Rewrite `backends/protocol.py` — new interface | `backends/protocol.py` |
| 7 | Rename `backends/memory_backend.py` → `backends/memory.py`, update interface | `backends/memory.py` |
| 8 | Rewrite `backends/filesystem.py` — new path layout | `backends/filesystem.py` |
| 9 | Add `backends/composite.py` | `backends/composite.py` |
| 10 | Update `backends/__init__.py` | `backends/__init__.py` |
| 11 | Move `middleware/sessions.py` → `sessions.py`, no logic changes | `sessions.py` |
| 12 | Create `system_prompt.py` — absorb `middleware/skills.py` logic, rename from `instructions.py`, all prompt constants referencing langchain/deepagents wording, `build_system_prompt()` in correct section order | `system_prompt.py` |
| 13 | Delete `middleware/skills.py` (logic now in `system_prompt.py`) | — |
| 14 | Rewrite `tools/workspace.py` — `ls`, `glob`, `grep`, updated `read_file` (line numbers, image support, path routing), `write_file`/`edit_file`/`append_to_file` with path routing, `execute` always registered, `list_files` deprecated alias | `tools/workspace.py` |
| 15 | Update `tools/memory.py` — add `read_store`, `write_store` | `tools/memory.py` |
| 16 | Update `tools/planning.py` — `write_todos` appends to `logs/plans.json` when `debug=True`, uses `agent_name`-scoped plan | `tools/planning.py` |
| 17 | Update `tools/__init__.py` — add `glob`, `grep` to `WORKSPACE_TOOLS` | `tools/__init__.py` |
| 18 | Rewrite `middleware/workspace.py` — `on_agent_start` sets `agent_name`, loads correct plan; `on_tool_end` eviction-only; `wrap_tools_for_logging` sole logger (debug only) | `middleware/workspace.py` |
| 19 | Move/rewrite `middleware/observability.py` — LangSmith | `middleware/observability.py` |
| 20 | Update `middleware/__init__.py` | `middleware/__init__.py` |
| 21 | Rewrite `factory.py` — new signature, `_build_subagent()`, general-purpose subagent logic, `task` tool replacing `spawn_task`, `DeepAgentRunner` with `run`/`run_sync`/`run_stream`, debug logging, default `FilesystemBackend(".deepx")` | `factory.py` |
| 22 | Update `src/deepx/__init__.py` | `__init__.py` |
| 23 | Rewrite `agent.py` test | `agent.py` |```
---
./tests/demo.py
```python
"""Demo: Multi-agent research and architecture decision record (ADR) generation.

Demonstrates:
- write_todos for structured planning
- Parallel task delegation via the task tool
- web_agent subagent (web_search, web_extract, think_tool)
- Writer subagent producing a polished document from research files
- HITL session approval (approve once per tool name; declines return tool messages)
- Debug tool logging
- Skills from tests/skills (deep_researcher orchestration)

Usage:
    OPENAI_API_KEY=... TAVILY_API_KEY=... python tests/demo.py
"""

from __future__ import annotations

import os
import uuid

import httpx
from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv

from deepx import create_deep_agent

load_dotenv()

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")

THINK_TOOL_DESCRIPTION = (
    "Tool for strategic reflection on research progress and decision-making.\n\n"
    "Use this tool after each web_search or web_extract to analyze results and plan next steps.\n\n"
    "When to use:\n"
    "- After search or extract results: what key information did you find?\n"
    "- Before deciding next steps: do you have enough to answer comprehensively?\n"
    "- Update mental model of todos / gaps in the research.\n"
    "- When assessing gaps: what specific information is still missing?\n"
    "- Before concluding: can you provide a complete answer now?\n\n"
    "Reflection should address:\n"
    "1. Analysis of current findings — what concrete information have you gathered?\n"
    "2. Gap assessment — what crucial information is still missing?\n"
    "3. Quality evaluation — sufficient evidence/examples for a good write-up?\n"
    "4. Strategic decision — continue searching or finalize findings to files?\n\n"
    "Args:\n"
    "- reflection: Your detailed reflection on progress, findings, gaps, and next steps.\n\n"
    "Returns:\n"
    "- Confirmation that reflection was recorded for decision-making."
)


@function_tool(description_override=THINK_TOOL_DESCRIPTION)
def think_tool(reflection: str) -> str:
    """Record structured reflection (no side effects beyond returning confirmation)."""
    return (
        f"Reflection: {reflection}. \n\n"
        "Now update your plan:"
        "call list_todos to review progress, then write_todos if anything needs updating."
    )


@function_tool
async def web_search(ctx: RunContextWrapper, query: str) -> str:
    """Search the web for current information. Returns titles, URLs, and snippets."""
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot perform web search."
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query, "max_results": 5},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "No results found."
    return "\n\n---\n\n".join(
        f"[{i + 1}] {res['title']}\nURL: {res['url']}\n{res['content']}"
        for i, res in enumerate(results)
    )


@function_tool
async def web_extract(ctx: RunContextWrapper, url: str) -> str:
    """Extract the full text content from a URL."""
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot extract content."
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/extract",
            json={"api_key": TAVILY_KEY, "urls": [url]},
            timeout=25,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "Could not extract content from that URL."
    return results[0].get("raw_content", "No content extracted.")


web_agent_spec = {
    "name": "web_agent",
    "description": (
        "Web research specialist with web_search and web_extract. "
        "Give it a complete list of topics or questions to investigate. "
        "It writes distilled findings (not raw tool dumps) to research/`file_name`.md and returns those paths."
    ),
    "system_prompt": (
        "You are a web research specialist. You have web_search, web_extract, and think_tool.\n"
        "**CRITICAL: Call think_tool after each web_search or web_extract** to reflect on results "
        "and plan next steps.\n"
        "For each topic: search, extract the most relevant URLs as needed, then write a **synthesized** "
        "markdown file at research/<topic-slug>.md with clear headings, bullet summaries, inline citations "
        "[1], [2] mapping to sources, and only short quoted excerpts — do **not** paste full raw search "
        "or extract payloads into files.\n"
        "When done, return a message listing every file path you wrote, e.g.: "
        "'Saved research to: research/foo.md, research/bar.md'."
    ),
    "tools": [think_tool, web_search, web_extract],
}

writer = {
    "name": "writer",
    "description": (
        "Technical writer. Reads research files specified by the caller, then produces "
        "a polished document and returns the full content as its final response."
    ),
    "system_prompt": (
        "You are a professional technical writer. "
        "You will be given a task description and one or more research file paths to read. "
        "Read all specified research files using read_file, then write the requested document. "
        "Return the complete document content in your final response — do not save it to a file."
    ),
}

agent = create_deep_agent(
    model="gpt-4o-mini",
    name="deep_researcher",
    subagents=[web_agent_spec, writer],
    tools=[think_tool],
    skills=["tests/skills"],
    system_prompt=(
        "You are the deep_researcher: coordinate multi-step web research and writing for the user.\n"
        "Follow the deep_researcher skill (read its SKILL.md when the task matches). "
        "Use write_todos immediately to plan, then delegate research to web_agent and writing to writer. "
        "Use think_tool after major delegations or when you need to reason about gaps before the next step."
    ),
    interrupt_on=["web_search"],
    debug=True,
    db_path="tests/demo.db",
)

TASK = """
Investigate the long-term viability of sodium-ion batteries as a sustainable alternative to lithium-ion technology in the electric vehicle market. The research should analyze current energy density limitations, the geopolitical stability of the raw material supply chain, and the existing manufacturing hurdles for large-scale adoption. Additionally, the agent should identify key startups leading the sector and provide a cost-benefit analysis comparing the lifecycle environmental impact of sodium versus lithium extraction.
After investigating write a report.
"""

if __name__ == "__main__":
    session_id = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    result = agent.run_sync(TASK, session_id=session_id)

    print("\n" + "=" * 70)
    print(result.output)
    print("=" * 70)
    print(
        f"\nSession: {result.session_id}  (resume with SESSION_ID={result.session_id})"
    )
```
---
./tests/skills/orchestration/SKILL.md
```markdown
---
name: deep_researcher
description: Multi-agent web research pipeline. Coordinate web_agent and writer with write_todos, files, and citations for deep research tasks.
---

# Deep research workflow

Follow this workflow when the user wants thorough research plus a written deliverable (report, ADR, memo, comparison, etc.):

1. **Plan**: Create a todo list with `write_todos` — break the work into concrete steps (research, synthesize, write, verify).
2. **Save the request** (recommended): Use `write_file` to save the user's research question or brief to `/research_request.md` so you can verify coverage later.
3. **Research**: Delegate to the **`web_agent`** subagent via `task(subagent_type="web_agent", ...)`. The web_agent has **`web_search`**, **`web_extract`**, and **`think_tool`**. It must use **`think_tool` after each search or extract** to reflect and plan next steps. It writes **distilled** findings to `research/<topic-slug>.md` (structured prose, headings, short quotes, inline `[n]` citations) — **not** full raw dumps of search/extract tool output.
4. **Synthesize**: Read the web_agent's returned file paths (and files if needed). Consolidate citation numbers if multiple files overlap (each unique URL one number across findings).
5. **Write**: Delegate to the **`writer`** subagent with the user's deliverable instructions and the **exact** research file paths. The writer returns the full document text in its final message.
6. **Verify**: Re-read `/research_request.md` (if you created it) and confirm the writer's output addresses every part with appropriate structure and citations.

## Delegation strategy

- **Default: one `web_agent`** for most questions (single coherent topic, one comprehensive research pass).
- **Parallel `web_agent` tasks only** when the query **explicitly** requires comparison of distinct entities or clearly independent aspects (e.g. three-company comparison → up to three parallel tasks). Prefer one thorough delegation over many narrow ones.
- Use at most **3** parallel `task()` calls to `web_agent` per iteration unless the user explicitly needs more.

## Report and citation expectations

When coordinating the final document (via `writer`):

- Cite sources inline using `[1]`, `[2]`, …
- End with a `### Sources` section: one line per source, `[n] Title: URL`
- Prefer professional report tone; avoid meta narration ("I searched…") in the final doc unless the user asked for process.

## Rules

- **web_agent** must finish (files written, paths returned) before **writer** starts.
- Pass **explicit absolute paths** to `writer`. Do not ask it to discover files on its own.
- Return the writer's content to the user as requested; do not silently rewrite it unless asked.

## Available tools (reference)

- **deep_researcher** (you): planning, filesystem, `task`, optional `think_tool`.
- **web_agent**: `web_search`, `web_extract`, `think_tool`, filesystem for `research/*.md`.
- **writer**: filesystem reads + final prose in the return message.
```
---
./src/deepx/tools/__init__.py
```python
from deepx.tools.filesystem import edit_file, execute, glob, grep, ls, read_file, write_file
from deepx.tools.planning import list_todos, write_todos

FILESYSTEM_TOOLS = [
    ls,
    read_file,
    write_file,
    edit_file,
    glob,
    grep,
    execute,
]
PLANNING_TOOLS = [write_todos, list_todos]

__all__ = [
    "FILESYSTEM_TOOLS",
    "PLANNING_TOOLS",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
    "write_todos",
    "list_todos",
]
```
---
./src/deepx/tools/planning.py
```python
from __future__ import annotations

import json
from datetime import datetime, timezone

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel

from deepx.context import AgentContext
from deepx.models import Todo, TodoStatus


class TodoInput(BaseModel):
    content: str
    status: str = "pending"

WRITE_TODOS_TOOL_DESCRIPTION = (
    "Create or update your structured task list. Call this tool whenever your work involves more than "
    "a single direct answer — any tool use, delegation, or multi-step task of any kind.\n\n"
    "**Before building the list:** review the descriptions of available tools and subagents. "
    "Design steps that match what each tool or subagent can accomplish in a single invocation. "
    "Do not create one todo step per task when one subagent call can cover all of them."
    "   For example: creating a todo list like step-1: delegate task1 to subagent1"
    "   step-2: delegate task2 to subagent1"
    "   step-3: delegate task3 to subagent1 and so on when we can delegate all of"
    "those tasks to the subagent1 in one step\n\n" 
    "**After each step completes:** call `list_todos` first to read the current state, then call "
    "`write_todos` to mark the finished step `completed` and advance the next to `in_progress`.\n\n"
    "## Rules\n\n"
    "- Always pass the **complete list** — never omit existing entries.\n"
    "- Never call this tool multiple times in parallel.\n"
    "- Mark the first step `in_progress` when you create the plan; keep exactly one step `in_progress` "
    "unless running genuinely parallel independent work.\n"
    "- ONLY mark a step `completed` when fully done. If blocked, keep it `in_progress` and add a new "
    "step describing the blocker.\n"
    "- When new work appears mid-run, append it and update statuses in the same call.\n"
    "- Keep completed steps in the list for visibility — do not delete them.\n\n"
    "## Task states\n\n"
    "- `pending` — not yet started\n"
    "- `in_progress` — currently being worked on\n"
    "- `completed` — fully done\n\n"
    "The only time you may skip this tool is for a purely conversational reply with zero tool use."
)


@function_tool
def list_todos(ctx: RunContextWrapper[AgentContext]) -> str:
    """Return the current todo list for this agent.

    Call this after completing a step to review progress before updating write_todos.
    """
    todos = ctx.context.plan.todos
    if not todos:
        return "No todos yet. Use write_todos to create your plan."
    lines = [
        f"[{i + 1}] ({t.status.value}) {t.title}"
        for i, t in enumerate(todos)
    ]
    return "Current plan:\n" + "\n".join(lines)


@function_tool(description_override=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    ctx: RunContextWrapper[AgentContext],
    todos: list[TodoInput],
) -> str:
    """Replace the current todo list with the provided items."""
    ctx.context.plan.todos = [
        Todo(
            title=t.content,
            status=_safe_status(t.status),
        )
        for t in todos
    ]
    ctx.context.plan.updated_at = datetime.now(timezone.utc).isoformat()
    _persist_plan(ctx)
    if ctx.context.debug:
        entry = {
            "timestamp": ctx.context.plan.updated_at,
            "agent": ctx.context.agent_name,
            "todos": [{"content": t.title, "status": t.status.value} for t in ctx.context.plan.todos],
        }
        ctx.context.backend.append_plan_log(
            ctx.context.session_id, json.dumps(entry)
        )
    lines = [
        f"[{i + 1}] ({t.status.value}) {t.title}"
        for i, t in enumerate(ctx.context.plan.todos)
    ]
    return "Plan saved:\n" + "\n".join(lines)


def _safe_status(value: str) -> TodoStatus:
    try:
        return TodoStatus(value)
    except ValueError:
        return TodoStatus.pending


def _persist_plan(ctx: RunContextWrapper[AgentContext]) -> None:
    ctx.context.plan.agent_name = ctx.context.agent_name or ctx.context.plan.agent_name
    ctx.context.backend.save_plan(
        ctx.context.session_id,
        ctx.context.plan.agent_name,
        ctx.context.plan.to_json(),
    )
```
---
./src/deepx/tools/filesystem.py
```python
from __future__ import annotations

import base64
import fnmatch
from datetime import datetime, timezone
from typing import Literal

from agents import RunContextWrapper, function_tool

from deepx.backends.filesystem import FilesystemBackend
from deepx.context import AgentContext

_IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


def _route_path(path: str) -> tuple[str, str]:
    p = path.strip().replace("\\", "/")
    if p.startswith("/store/"):
        return "store", p[7:].lstrip("/")
    if p.startswith("store/"):
        return "store", p[6:].lstrip("/")
    return "files", p.lstrip("/")


def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _path_ext(rel: str) -> str:
    if "." not in rel:
        return ""
    return "." + rel.rsplit(".", 1)[-1].lower()


def _mime(ext: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")


def _child_map(all_rels: list[str], base_prefix: str) -> dict[str, bool]:
    base_prefix = base_prefix.strip("/")
    pfx = f"{base_prefix}/" if base_prefix else ""
    children: dict[str, bool] = {}
    for rel in all_rels:
        if base_prefix:
            if rel == base_prefix:
                continue
            if not rel.startswith(pfx):
                continue
            rest = rel[len(pfx):]
        else:
            rest = rel
        if not rest:
            continue
        head, _, tail = rest.partition("/")
        is_dir = bool(tail) or any(o.startswith(f"{pfx}{head}/") for o in all_rels)
        if head in children:
            children[head] = children[head] or is_dir
        else:
            children[head] = is_dir
    return children


def _format_ls_lines(
    ctx: RunContextWrapper[AgentContext],
    children: dict[str, bool],
    pfx: str,
    *,
    store: bool,
) -> str:
    b = ctx.context.backend
    sid = ctx.context.session_id
    fs_b = b if isinstance(b, FilesystemBackend) else None
    lines: list[str] = []
    for name in sorted(children):
        is_dir = children[name]
        if is_dir:
            lines.append(f"{name}/{' ' * max(0, 22 - len(name))}DIR")
            continue
        rel_path = f"{pfx}{name}" if pfx else name
        ts = ""
        sz = ""
        if store:
            raw = b.read_store(rel_path)
            if raw is not None:
                sz = _format_size(len(raw.encode("utf-8")))
        elif fs_b:
            p_stat = fs_b._file_path(sid, rel_path)
            if p_stat.is_file():
                st = p_stat.stat()
                sz = _format_size(st.st_size)
                ts = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                )
        else:
            raw = b.read(sid, rel_path)
            if raw is not None:
                sz = _format_size(len(raw.encode("utf-8")))
        pad = max(0, 18 - len(name))
        lines.append(f"{name}{' ' * pad}{sz}{' ' * 4}{ts}".rstrip())
    return "\n".join(lines) if lines else "(empty)"


def _run_ls(ctx: RunContextWrapper[AgentContext], path: str) -> str:
    kind, rel = _route_path(path)
    base_prefix = rel.strip("/")
    if kind == "store":
        all_rels = ctx.context.backend.list_store("")
        children = _child_map(all_rels, base_prefix)
        pfx = f"{base_prefix}/" if base_prefix else ""
        return _format_ls_lines(ctx, children, pfx, store=True)
    all_rels = ctx.context.backend.list_files(ctx.context.session_id, "")
    children = _child_map(all_rels, base_prefix)
    pfx = f"{base_prefix}/" if base_prefix else ""
    return _format_ls_lines(ctx, children, pfx, store=False)


@function_tool
def ls(ctx: RunContextWrapper[AgentContext], path: str = "/") -> str:
    """Lists all files in a directory.

    This is useful for exploring the filesystem and finding the right file to read or edit.
    You should almost ALWAYS use this tool before using the read_file or edit_file tools."""
    return _run_ls(ctx, path)


@function_tool
def read_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """Reads a file from the filesystem.

    Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

    Usage:
    - By default, it reads up to 100 lines starting from the beginning of the file
    - **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
      - First scan: read_file(path, limit=100) to see file structure
      - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
      - Only omit limit (read full file) when necessary for editing
    - Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
    - Results are returned using cat -n format, with line numbers starting at 1
    - You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
    - If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
    - Image files (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) are returned as base64-encoded data.

    - You should ALWAYS make sure a file has been read before editing it."""
    kind, rel = _route_path(path)
    b = ctx.context.backend
    sid = ctx.context.session_id
    ext = _path_ext(rel)

    if kind == "store":
        if ext in _IMAGE_EXT and isinstance(b, FilesystemBackend):
            raw = b.read_store_bytes(rel)
            if raw is None:
                return f"Error: '{path}' not found."
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return f"[Image base64 {ext}]\ndata:{_mime(ext)};base64,{b64}"
        content = b.read_store(rel)
    else:
        if ext in _IMAGE_EXT and isinstance(b, FilesystemBackend):
            raw = b.read_session_bytes(sid, rel)
            if raw is None:
                return f"Error: '{path}' not found."
            b64 = base64.standard_b64encode(raw).decode("ascii")
            return f"[Image base64 {ext}]\ndata:{_mime(ext)};base64,{b64}"
        content = b.read(sid, rel)

    if content is None:
        return f"Error: '{path}' not found."

    if not content.strip():
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    selected = lines[offset: offset + limit]
    if not selected and lines:
        return f"Error: offset {offset} exceeds file length ({len(lines)} lines)."
    numbered = "\n".join(
        f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(selected)
    )
    if len(lines) > offset + limit:
        numbered += f"\n\n[{len(lines) - offset - limit} more lines — use offset={offset + limit} to continue]"
    return numbered


@function_tool
def write_file(
    ctx: RunContextWrapper[AgentContext], path: str, content: str
) -> str:
    """Writes to a new file in the filesystem.

    Usage:
    - The write_file tool will create a new file.
    - Prefer to edit existing files (with the edit_file tool) over creating new ones when possible."""
    kind, rel = _route_path(path)
    b = ctx.context.backend
    sid = ctx.context.session_id
    if kind == "store":
        if b.read_store(rel) is not None:
            return (
                f"Cannot write to {path} because it already exists. "
                "Read and then make an edit, or write to a new path."
            )
        b.write_store(rel, content)
    else:
        if b.exists(sid, rel):
            return (
                f"Cannot write to {path} because it already exists. "
                "Read and then make an edit, or write to a new path."
            )
        b.write(sid, rel, content)
    return f"Updated file {path}"


@function_tool
def edit_file(
    ctx: RunContextWrapper[AgentContext],
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Performs exact string replacements in files.

    Usage:
    - You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
    - When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
    - ALWAYS prefer editing existing files over creating new ones.
    - Only use emojis if the user explicitly requests it."""
    kind, rel = _route_path(path)
    b = ctx.context.backend
    sid = ctx.context.session_id
    if kind == "store":
        content = b.read_store(rel)
    else:
        content = b.read(sid, rel)
    if content is None:
        return f"Error: '{path}' not found."
    count = content.count(old_string)
    if count == 0:
        return f"Error: string not found in '{path}'."
    if not replace_all and count > 1:
        return (
            f"Error: String '{old_string}' appears {count} times in file. "
            "Use replace_all=True to replace all instances, or provide a more "
            "specific string with surrounding context."
        )
    new_content = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )
    if kind == "store":
        b.write_store(rel, new_content)
    else:
        b.write(sid, rel, new_content)
    return f"Successfully replaced {count if replace_all else 1} instance(s) of the string in '{path}'"


@function_tool
def glob(
    ctx: RunContextWrapper[AgentContext], pattern: str, path: str = "/"
) -> str:
    """Find files matching a glob pattern.

    Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character).
    Returns a list of file paths that match the pattern.

    Examples:
    - `**/*.py` - Find all Python files
    - `*.txt` - Find all text files in root
    - `/subdir/**/*.md` - Find all markdown files under /subdir"""
    kind, rel = _route_path(path)
    base_prefix = rel.strip("/")
    pfx = f"{base_prefix}/" if base_prefix else ""
    if kind == "store":
        candidates = ctx.context.backend.list_store("")
        if base_prefix:
            candidates = [
                c
                for c in candidates
                if c == base_prefix or c.startswith(pfx)
            ]
    else:
        sid = ctx.context.session_id
        candidates = ctx.context.backend.list_files(sid, prefix=pfx if pfx else "")
    matches = sorted({p for p in candidates if fnmatch.fnmatch(p, pattern)})
    return "\n".join(matches) if matches else "(no matches)"


@function_tool
def grep(
    ctx: RunContextWrapper[AgentContext],
    pattern: str,
    path: str | None = None,
    glob_pattern: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    """Search for a text pattern across files.

    Searches for literal text (not regex) and returns matching files or content based on output_mode.
    Special characters like parentheses, brackets, pipes, etc. are treated as literal characters, not regex operators.

    Examples:
    - Search all files: `grep(pattern="TODO")`
    - Search Python files only: `grep(pattern="import", glob_pattern="*.py")`
    - Show matching lines: `grep(pattern="error", output_mode="content")`
    - Search for code with special chars: `grep(pattern="def __init__(self):")`"""
    sid = ctx.context.session_id
    b = ctx.context.backend

    if path:
        kind, rel = _route_path(path)
        base_prefix = rel.strip("/")
        pfx = f"{base_prefix}/" if base_prefix else ""
        if kind == "store":
            files = b.list_store("")
            if base_prefix:
                files = [
                    f
                    for f in files
                    if f == base_prefix or f.startswith(pfx)
                ]
        else:
            files = b.list_files(sid, prefix=pfx if pfx else "")
    else:
        files = b.list_files(sid, "")
        kind = "files"

    if glob_pattern:
        files = [f for f in files if fnmatch.fnmatch(f, glob_pattern)]

    results: list[str] = []
    for fp in sorted(files):
        if path and _route_path(path)[0] == "store":
            raw = b.read_store(fp)
        else:
            raw = b.read(sid, fp)
        if raw is None:
            continue
        count = raw.count(pattern)
        if count == 0:
            continue
        if output_mode == "count":
            results.append(f"{fp}:{count}")
        elif output_mode == "files_with_matches":
            results.append(fp)
        else:
            for i, line in enumerate(raw.splitlines(), 1):
                if pattern in line:
                    results.append(f"{fp}:{i}:{line}")
    return "\n".join(results) if results else "(no matches)"


@function_tool
def execute(ctx: RunContextWrapper[AgentContext], command: str) -> str:
    """Executes a shell command in an isolated sandbox environment.

    Usage:
    Executes a given command in the sandbox environment with proper handling and security measures.
    Before executing the command, please follow these steps:
    1. Directory Verification:
       - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
    2. Command Execution:
       - Always quote file paths that contain spaces with double quotes
       - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines
    Usage notes:
      - Commands run in an isolated sandbox environment
      - Returns combined stdout/stderr output with exit code
      - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.

    Note: This tool is only available if the backend supports execution."""
    b = ctx.context.backend
    if not b.supports_execution:
        return (
            "Shell execution is not available. Use a sandbox backend to enable it."
        )
    return b.execute(command)
```
---
./src/deepx/__init__.py
```python
from deepx.backends.composite import CompositeBackend
from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory import InMemoryBackend
from deepx.backends.protocol import BackendProtocol
from deepx.factory import (
    DeepAgent,
    DeepAgentRunner,
    DeepRunResult,
    SubAgentDict,
    create_deep_agent,
)
from deepx.middleware.hitl import HumanInTheLoopHooks

__all__ = [
    "create_deep_agent",
    "DeepAgent",
    "DeepAgentRunner",
    "DeepRunResult",
    "SubAgentDict",
    "HumanInTheLoopHooks",
    "BackendProtocol",
    "FilesystemBackend",
    "InMemoryBackend",
    "CompositeBackend",
]
```
---
./src/deepx/backends/__init__.py
```python
from deepx.backends.composite import CompositeBackend
from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.memory import InMemoryBackend
from deepx.backends.protocol import BackendProtocol

__all__ = [
    "BackendProtocol",
    "FilesystemBackend",
    "InMemoryBackend",
    "CompositeBackend",
]
```
---
./src/deepx/backends/protocol.py
```python
from __future__ import annotations

import abc


class BackendProtocol(abc.ABC):
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
    def read_store(self, path: str) -> str | None: ...

    @abc.abstractmethod
    def write_store(self, path: str, content: str) -> None: ...

    @abc.abstractmethod
    def list_store(self, prefix: str = "") -> list[str]: ...

    @abc.abstractmethod
    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None: ...

    @abc.abstractmethod
    def load_plan(self, session_id: str, agent_name: str) -> str | None: ...

    @abc.abstractmethod
    def append_task_log(self, session_id: str, task: str) -> None: ...

    @abc.abstractmethod
    def append_plan_log(self, session_id: str, entry_json: str) -> None: ...

    @abc.abstractmethod
    def save_tool_log(self, session_id: str, log_data: dict) -> None: ...

    @abc.abstractmethod
    def append_system_prompt_log(self, session_id: str, agent_name: str, prompt: str) -> None: ...

    @property
    def supports_execution(self) -> bool:
        return False

    def execute(self, command: str) -> str:
        raise NotImplementedError("Shell execution requires a sandbox backend.")
```
---
./src/deepx/backends/composite.py
```python
from __future__ import annotations

from deepx.backends.protocol import BackendProtocol


class CompositeBackend(BackendProtocol):
    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol],
    ) -> None:
        self._default = default
        self._routes = sorted(
            ((p, b) for p, b in routes.items() if p),
            key=lambda x: -len(x[0]),
        )

    def _pick(self, path: str) -> tuple[BackendProtocol, str]:
        for prefix, backend in self._routes:
            if path.startswith(prefix):
                return backend, path[len(prefix) :].lstrip("/")
        return self._default, path

    def read(self, session_id: str, path: str) -> str | None:
        b, p = self._pick(path)
        return b.read(session_id, p)

    def write(self, session_id: str, path: str, content: str) -> None:
        b, p = self._pick(path)
        b.write(session_id, p, content)

    def append(self, session_id: str, path: str, content: str) -> None:
        b, p = self._pick(path)
        b.append(session_id, p, content)

    def exists(self, session_id: str, path: str) -> bool:
        b, p = self._pick(path)
        return b.exists(session_id, p)

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        b, p = self._pick(prefix or "")
        return b.list_files(session_id, p)

    def read_store(self, path: str) -> str | None:
        return self._default.read_store(path)

    def write_store(self, path: str, content: str) -> None:
        self._default.write_store(path, content)

    def list_store(self, prefix: str = "") -> list[str]:
        return self._default.list_store(prefix)

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        self._default.save_plan(session_id, agent_name, plan_json)

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        return self._default.load_plan(session_id, agent_name)

    def append_task_log(self, session_id: str, task: str) -> None:
        self._default.append_task_log(session_id, task)

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        self._default.append_plan_log(session_id, entry_json)

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        self._default.save_tool_log(session_id, log_data)

    def append_system_prompt_log(self, session_id: str, agent_name: str, prompt: str) -> None:
        self._default.append_system_prompt_log(session_id, agent_name, prompt)

    @property
    def supports_execution(self) -> bool:
        return self._default.supports_execution

    def execute(self, command: str) -> str:
        return self._default.execute(command)
```
---
./src/deepx/backends/memory.py
```python
from __future__ import annotations

import json

from deepx.backends.protocol import BackendProtocol


class InMemoryBackend(BackendProtocol):
    def __init__(self) -> None:
        self._files: dict[tuple[str, str], str] = {}
        self._store: dict[str, str] = {}
        self._plans: dict[tuple[str, str], str] = {}
        self._task_logs: dict[str, list[str]] = {}
        self._plan_logs: dict[str, list[dict]] = {}
        self._tool_logs: dict[str, list[dict]] = {}

    def read(self, session_id: str, path: str) -> str | None:
        key = (session_id, path.lstrip("/"))
        return self._files.get(key)

    def write(self, session_id: str, path: str, content: str) -> None:
        key = (session_id, path.lstrip("/"))
        self._files[key] = content

    def append(self, session_id: str, path: str, content: str) -> None:
        key = (session_id, path.lstrip("/"))
        self._files[key] = self._files.get(key, "") + content

    def exists(self, session_id: str, path: str) -> bool:
        key = (session_id, path.lstrip("/"))
        return key in self._files

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        prefix = prefix.lstrip("/")
        keys = [p for sid, p in self._files if sid == session_id]
        return sorted(k for k in keys if not prefix or k.startswith(prefix))

    def read_store(self, path: str) -> str | None:
        return self._store.get(path.lstrip("/"))

    def write_store(self, path: str, content: str) -> None:
        self._store[path.lstrip("/")] = content

    def list_store(self, prefix: str = "") -> list[str]:
        prefix = prefix.lstrip("/")
        return sorted(k for k in self._store if not prefix or k.startswith(prefix))

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        self._plans[(session_id, agent_name)] = plan_json

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        return self._plans.get((session_id, agent_name))

    def append_task_log(self, session_id: str, task: str) -> None:
        self._task_logs.setdefault(session_id, []).append(task)

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        try:
            obj = json.loads(entry_json)
        except json.JSONDecodeError:
            obj = {"raw": entry_json}
        self._plan_logs.setdefault(session_id, []).append(obj)

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        self._tool_logs.setdefault(session_id, []).append(log_data)

    def append_system_prompt_log(self, session_id: str, agent_name: str, prompt: str) -> None:
        pass
```
---
./src/deepx/backends/filesystem.py
```python
from __future__ import annotations

import json
import re
from pathlib import Path

from deepx.backends.protocol import BackendProtocol


def _safe_agent_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name) or "agent"


class FilesystemBackend(BackendProtocol):
    def __init__(self, root: str | Path = ".deepx") -> None:
        self._root = Path(root)

    def _files_base(self, session_id: str) -> Path:
        return self._root / "sessions" / session_id / "files"

    def _file_path(self, session_id: str, path: str) -> Path:
        rel = path.lstrip("/").replace("\\", "/")
        return self._files_base(session_id) / rel

    def _memory_base(self) -> Path:
        return self._root / "memory"

    def _memory_path(self, path: str) -> Path:
        rel = path.lstrip("/").replace("\\", "/")
        return self._memory_base() / rel

    def _plan_path(self, session_id: str, agent_name: str) -> Path:
        safe = _safe_agent_name(agent_name)
        return self._root / "sessions" / session_id / "plans" / f"{safe}.json"

    def _logs_dir(self, session_id: str) -> Path:
        return self._root / "sessions" / session_id / "logs"

    def _tool_log_path(self, session_id: str, tool_name: str, call_id: str) -> Path:
        return self._logs_dir(session_id) / "tools" / tool_name / f"{call_id}.json"

    def read(self, session_id: str, path: str) -> str | None:
        p = self._file_path(session_id, path)
        return p.read_text() if p.is_file() else None

    def read_session_bytes(self, session_id: str, path: str) -> bytes | None:
        p = self._file_path(session_id, path)
        return p.read_bytes() if p.is_file() else None

    def read_store_bytes(self, path: str) -> bytes | None:
        p = self._memory_path(path)
        return p.read_bytes() if p.is_file() else None

    def write(self, session_id: str, path: str, content: str) -> None:
        p = self._file_path(session_id, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def append(self, session_id: str, path: str, content: str) -> None:
        p = self._file_path(session_id, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content)

    def exists(self, session_id: str, path: str) -> bool:
        return self._file_path(session_id, path).is_file()

    def list_files(self, session_id: str, prefix: str = "") -> list[str]:
        base = self._files_base(session_id)
        if not base.exists():
            return []
        prefix = prefix.lstrip("/").replace("\\", "/")
        out: list[str] = []
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            if prefix and not rel.startswith(prefix):
                continue
            out.append(rel)
        return out

    def read_store(self, path: str) -> str | None:
        p = self._memory_path(path)
        return p.read_text() if p.is_file() else None

    def write_store(self, path: str, content: str) -> None:
        p = self._memory_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def list_store(self, prefix: str = "") -> list[str]:
        base = self._memory_base()
        if not base.exists():
            return []
        prefix = prefix.lstrip("/").replace("\\", "/")
        out: list[str] = []
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            if prefix and not rel.startswith(prefix):
                continue
            out.append(rel)
        return out

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        p = self._plan_path(session_id, agent_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(plan_json)

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        p = self._plan_path(session_id, agent_name)
        return p.read_text() if p.is_file() else None

    def append_task_log(self, session_id: str, task: str) -> None:
        p = self._logs_dir(session_id) / "tasks.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(task + "\n")

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        p = self._logs_dir(session_id) / "plans.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                arr = json.loads(p.read_text())
                if not isinstance(arr, list):
                    arr = []
            except json.JSONDecodeError:
                arr = []
        else:
            arr = []
        try:
            arr.append(json.loads(entry_json))
        except json.JSONDecodeError:
            arr.append({"raw": entry_json})
        p.write_text(json.dumps(arr, indent=2))

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        tool_name = str(log_data["tool_name"])
        call_id = str(log_data["call_id"])
        path = self._tool_log_path(session_id, tool_name, call_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log_data, indent=2))

    def append_system_prompt_log(self, session_id: str, agent_name: str, prompt: str) -> None:
        from datetime import datetime, timezone
        p = self._logs_dir(session_id) / "system_prompts.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                arr = json.loads(p.read_text())
                if not isinstance(arr, list):
                    arr = []
            except json.JSONDecodeError:
                arr = []
        else:
            arr = []
        arr.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "prompt": prompt,
        })
        p.write_text(json.dumps(arr, indent=2))
```
---
./src/deepx/context.py
```python
from __future__ import annotations

from dataclasses import dataclass, field

from deepx.backends.protocol import BackendProtocol
from deepx.models import Plan


@dataclass
class AgentContext:
    session_id: str
    backend: BackendProtocol
    agent_name: str = ""
    plan: Plan = field(init=False)
    memory: str = ""
    skills_info: str = ""
    debug: bool = False
    hitl_tools: list[str] = field(default_factory=list)
    resume: bool = False

    def __post_init__(self) -> None:
        an = self.agent_name or "agent"
        self.plan = Plan(session_id=self.session_id, agent_name=an)
```
---
./src/deepx/models.py
```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


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
    agent_name: str
    tasks: list[str] = Field(default_factory=list)
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

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
```
---
./src/deepx/_version.py
```python
__version__ = "0.1.0"
```
---
./src/deepx/system_prompt.py
```python
from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import yaml
from agents import Agent, RunContextWrapper

from deepx.context import AgentContext

HARD_LIMITS_PROMPT = """These are non-negotiable rules. Violating any of them is a critical failure.

1. **ALWAYS call `write_todos` before starting work.** For any task that involves more than a single direct answer — tool use, delegation, multi-step work of any kind — call `write_todos` first. No exceptions.
2. **Understand your tools and subagents before planning.** Before calling `write_todos`, read the available subagent descriptions. Design todo steps that match what each subagent can do in one call. Do not over-split work a single subagent or tool can handle in one invocation.
3. **After each completed step: call `list_todos`, then `write_todos`.** Use `list_todos` to read the current state, mark the finished step `completed`, advance the next step to `in_progress`.
4. **The workspace filesystem is internal. Never tell the user to open a file.** All session files are private agent-to-agent coordination storage the user cannot access. When your task produces a deliverable (document, code, report, analysis), write the **full content inline** in your response to the user. Never reference `/research/`, `sandbox:/`, or any workspace path in a user-facing response."""

BASE_AGENT_PROMPT = """You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- Don't say "I'll now do X" — just do it.
- If the request is ambiguous, ask questions before acting.
- If asked how to approach something, explain first, then act.

## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation

## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.

Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it. Only yield back to the user when the task is done or you're genuinely blocked.

**When things go wrong:**
- If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask for guidance.

## Deliverables go in your response, not in files

The agent session filesystem is private, AI-to-AI coordination storage. The user cannot access it. When your task produces a document, code, report, analysis, or any other deliverable, write the **full content inline** in your response — do not reference, link to, or ask the user to read files from the workspace.

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next."""

TODO_SYSTEM_PROMPT = """## Planning with `write_todos` and `list_todos`

Planning is **mandatory** for any task that involves more than a single direct answer. This applies equally to coding tasks, data analysis, multi-step tool use, subagent delegations, and everything in between.

**Before calling `write_todos`, review your available tools and subagents.** Read their descriptions to understand what each can accomplish in a single invocation. Build a plan whose steps match those real capabilities — do not create one step per query when a single subagent call can cover all of them.

### Lifecycle

1. **Start of work** → call `write_todos` with all steps. Mark the first step `in_progress`.
2. **Step completes** → call `list_todos` to review current state, then call `write_todos` to mark it `completed` and advance the next step to `in_progress`.
3. **New work discovered** → call `write_todos` to append new steps and update statuses in the same call.
4. **Blocked** → keep the step `in_progress`, add a new step describing the blocker.

### Rules for `write_todos`

- Always pass the **complete list** — never omit existing entries.
- Never call `write_todos` multiple times in parallel.
- Keep completed items in the list for visibility — do not delete them.
- Keep **exactly one** step `in_progress` unless you are running genuinely parallel independent work."""

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem conventions

- Read files before editing — understand existing content before making changes.
- Mimic existing style, naming conventions, and patterns.

## Tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem for session-scoped work. All file paths must start with `/`.
Use pagination (`offset`/`limit`) when reading large files.

- `ls` — list files in a directory (requires absolute path)
- `read_file` — read a file
- `write_file` — write a file
- `edit_file` — edit a file by string replacement
- `glob` — find files matching a pattern (e.g. `**/*.py`)
- `grep` — search for text within files

## Large tool results

When a tool result is too large, it is offloaded to the filesystem. Use `read_file` to inspect it in chunks, or `grep` within `/large_tool_results/`. Offloaded results are stored under `/large_tool_results/<tool_call_id>`."""

EXECUTION_SYSTEM_PROMPT = """## Tool: `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- `execute` — run a shell command in the sandbox (returns output and exit code)"""

TASK_SYSTEM_PROMPT = """## Delegating to subagents

Your available subagents appear in your tool list by name. Call each directly with an `input` parameter describing the task. Each subagent runs in isolation and returns a single result.

- Call `write_todos` before delegating to lay out the plan.
- Give each subagent a complete, self-contained prompt — it cannot ask follow-up questions.
- After a subagent returns, call `list_todos` then `write_todos` to update your plan.
- Parallelize subagent calls only when they have no data dependency on each other.
- Subagents are highly capable — delegate fully and trust the result."""

MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files in your filesystem. As you learn from your interactions with the user, you can save new knowledge by calling the `edit_file` tool.

    **Learning from feedback:**
    - One of your MAIN PRIORITIES is to learn from your interactions with the user. These learnings can be implicit or explicit. This means that in the future, you will remember this important information.
    - When you need to remember something, updating memory must be your FIRST, IMMEDIATE action - before responding to the user, before calling other tools, before doing anything else. Just update memory immediately.
    - When user says something is better/worse, capture WHY and encode it as a pattern.
    - Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions.
    - A great opportunity to update your memories is when the user interrupts a tool call and provides feedback. You should update your memories immediately before revising the tool call.
    - Look for the underlying principle behind corrections, not just the specific mistake.
    - The user might not explicitly ask you to remember something, but if they provide information that is useful for future use, you should update your memories immediately.

    **Asking for information:**
    - If you lack context to perform an action you should explicitly ask the user for this information.
    - It is preferred for you to ask for information, don't assume anything that you do not know!
    - When the user provides information that is useful for future use, you should update your memories immediately.

    **When to update memories:**
    - When the user explicitly asks you to remember something (e.g., "remember my email", "save this preference")
    - When the user describes your role or how you should behave (e.g., "you are a web researcher", "always do X")
    - When the user gives feedback on your work - capture what was wrong and how to improve
    - When the user provides information required for tool use (e.g., slack channel ID, email addresses)
    - When the user provides context useful for future tasks, such as how to use tools, or which actions to take in a particular situation
    - When you discover new patterns or preferences (coding styles, conventions, workflows)

    **When to NOT update memories:**
    - When the information is temporary or transient (e.g., "I'm running late", "I'm on my phone right now")
    - When the information is a one-time task request (e.g., "Find me a recipe", "What's 25 * 4?")
    - When the information is a simple question that doesn't reveal lasting preferences (e.g., "What day is it?", "Can you explain X?")
    - When the information is an acknowledgment or small talk (e.g., "Sounds good!", "Hello", "Thanks for that")
    - When the information is stale or irrelevant in future conversations
    - Never store API keys, access tokens, passwords, or any other credentials in any file, memory, or system prompt.
    - If the user asks where to put API keys or provides an API key, do NOT echo or save it.
</memory_guidelines>
"""

SKILLS_SYSTEM_PROMPT = """You have access to a skills library that provides specialized capabilities and domain knowledge.

**Available Skills:**

{skills_list}

**How to use skills:**

1. **Recognize when a skill applies** — check if the user's task matches a skill's description
2. **Read the full instructions** — use the path shown next to the skill
3. **Follow the skill's workflow** — `SKILL.md` contains step-by-step guidance, best practices, and examples

When in doubt, check if a skill exists for the task."""

HITL_PROMPT = """The following tools require explicit human approval before they run in this session: {tools}
Once approved in this session, a tool will not prompt for approval again.

If a tool returns a message starting with `[Human-in-the-loop]` stating that the human **declined** approval, that tool did **not** run. Acknowledge the decline, call `list_todos` to review your plan, update `write_todos` to reflect the change, and proceed differently — use non-sensitive tools, ask the user for guidance, or revise your approach. Do **not** blindly retry the same tool."""


class SkillMetadata(TypedDict, total=False):
    name: str
    description: str
    path: str
    license: str | None
    compatibility: str | None
    allowed_tools: list[str]


def discover_skills(paths: list[str]) -> list[SkillMetadata]:
    by_name: dict[str, SkillMetadata] = {}
    for raw in paths:
        root = Path(raw)
        if not root.exists():
            continue
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            meta = _parse_skill_frontmatter(
                skill_md.read_text(), str(skill_md.resolve())
            )
            if meta and meta.get("name") and meta.get("description"):
                by_name[str(meta["name"])] = meta
    return list(by_name.values())


def format_skills_for_prompt(skills: list[SkillMetadata]) -> str:
    if not skills:
        return ""
    lines: list[str] = []
    for skill in skills:
        lines.append(f"- **{skill['name']}**: {skill['description']}\n")
        lines.append(f"  -> Read `{skill['path']}` for full instructions")
    return "\n".join(lines)


def _parse_skill_frontmatter(content: str, path: str) -> SkillMetadata | None:
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
    meta: SkillMetadata = {
        "name": name,
        "description": description,
        "path": path,
    }
    if data.get("license") is not None:
        meta["license"] = str(data.get("license"))
    if data.get("compatibility") is not None:
        meta["compatibility"] = str(data.get("compatibility"))
    at = data.get("allowed-tools") or data.get("allowed_tools")
    if isinstance(at, list):
        meta["allowed_tools"] = [str(x) for x in at]
    elif isinstance(at, str):
        meta["allowed_tools"] = [at]
    return meta


def _discover_deepx_skills() -> list[SkillMetadata]:
    deepx_skills_path = Path(".deepx") / "skills"
    if deepx_skills_path.exists():
        return discover_skills([str(deepx_skills_path)])
    return []


def _labeled(title: str, content: str) -> str:
    """Prefix a section with a visible title so the separator blocks are clearly labeled."""
    return f"# {title}\n\n{content}"


def build_system_prompt(
    ctx: RunContextWrapper[AgentContext],
    agent: Agent,
    custom_prompt: str = "",
) -> str:
    sections: list[str] = []

    sections.append(_labeled("HARD LIMITS", HARD_LIMITS_PROMPT))

    if custom_prompt:
        sections.append(_labeled("AGENT ROLE", custom_prompt))

    sections.append(_labeled("CORE BEHAVIOR", BASE_AGENT_PROMPT))

    if ctx.context.memory:
        sections.append(
            _labeled("AGENT MEMORY", MEMORY_SYSTEM_PROMPT.format(agent_memory=ctx.context.memory))
        )

    all_skills_info = ctx.context.skills_info
    deepx_skills = _discover_deepx_skills()
    if deepx_skills:
        deepx_skills_text = format_skills_for_prompt(deepx_skills)
        all_skills_info = (
            (deepx_skills_text + "\n" + all_skills_info).strip()
            if all_skills_info
            else deepx_skills_text
        )

    if all_skills_info:
        sections.append(
            _labeled(
                "SKILLS",
                SKILLS_SYSTEM_PROMPT.format(skills_list=all_skills_info),
            )
        )

    sections.append(_labeled("FILESYSTEM", FILESYSTEM_SYSTEM_PROMPT))

    if ctx.context.backend.supports_execution:
        sections.append(_labeled("EXECUTION", EXECUTION_SYSTEM_PROMPT))

    sections.append(_labeled("PLANNING & DELEGATION", TODO_SYSTEM_PROMPT + "\n\n" + TASK_SYSTEM_PROMPT))

    if ctx.context.hitl_tools:
        sections.append(
            _labeled(
                "HUMAN-IN-THE-LOOP",
                HITL_PROMPT.format(tools=", ".join(ctx.context.hitl_tools)),
            )
        )

    if ctx.context.plan.todos:
        lines = [
            f"[{i + 1}] ({t.status.value}) {t.title}"
            for i, t in enumerate(ctx.context.plan.todos)
        ]
        sections.append(_labeled("CURRENT PLAN", "\n".join(lines)))

    files = ctx.context.backend.list_files(ctx.context.session_id)
    if files:
        shown = files[:50]
        block = "\n".join(shown)
        if len(files) > 50:
            block += (
                f"\n... and {len(files) - 50} more. Use ls with a prefix to filter."
            )
        sections.append(_labeled("SESSION FILES", block))

    _section_sep = "\n\n" + "=" * 80 + "\n\n"
    prompt = _section_sep.join(sections)

    if ctx.context.debug:
        try:
            ctx.context.backend.append_system_prompt_log(
                ctx.context.session_id, ctx.context.agent_name, prompt
            )
        except Exception:
            pass

    return prompt
```
---
./src/deepx/middleware/__init__.py
```python
from deepx.middleware.filesystem import (
    FilesystemHooks,
    apply_tool_pipeline,
    wrap_tools_for_logging,
    wrap_tools_with_large_output_eviction,
)
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.middleware.observability import setup_observability

__all__ = [
    "FilesystemHooks",
    "HumanInTheLoopHooks",
    "apply_tool_pipeline",
    "wrap_tools_for_logging",
    "wrap_tools_with_large_output_eviction",
    "setup_observability",
]
```
---
./src/deepx/middleware/observability.py
```python
from __future__ import annotations

import os


def setup_observability() -> None:
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "deepx")
    try:
        from langsmith.integrations.openai_agents_sdk import (
            OpenAIAgentsTracingProcessor,
        )
        from agents.tracing import set_trace_processors

        set_trace_processors([OpenAIAgentsTracingProcessor()])  # type: ignore[list-item]
    except Exception:
        return
```
---
./src/deepx/middleware/_utils.py
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
---
./src/deepx/middleware/hitl.py
```python
from __future__ import annotations

import asyncio
from collections.abc import Callable


class HumanInTheLoopHooks:
    """Human approval for sensitive tools, enforced inside the tool invoke path.

    Approval is remembered for the lifetime of this instance — once a tool is
    approved it will not be asked again. An asyncio.Lock serializes the
    check-and-add so concurrent tool calls cannot both slip through before the
    first approval is recorded.

    If the human declines, ``gate_tool`` returns a rejection message (the model
    sees it as normal tool output) instead of raising.

    Args:
        sensitive_tools: Tool names that require approval before they run.
        approval_fn: Optional override for the approval prompt. Receives
            ``(agent_name, tool_name)`` and returns ``True`` to approve.
            Defaults to a CLI ``input()`` prompt.
    """

    def __init__(
        self,
        sensitive_tools: list[str],
        approval_fn: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._sensitive = set(sensitive_tools)
        self._approved: set[str] = set()
        self._lock = asyncio.Lock()
        self._approval_fn = approval_fn or self._cli_approval

    @staticmethod
    def _cli_approval(agent_name: str, tool_name: str) -> bool:
        response = input(
            f"\n[HITL] Agent '{agent_name}' wants to call '{tool_name}'. Approve? [y/n]: "
        )
        return response.strip().lower() == "y"

    async def gate_tool(self, agent_name: str, tool_name: str) -> str | None:
        """Return ``None`` if the tool may run; otherwise a rejection message for the model."""
        if tool_name not in self._sensitive:
            return None
        async with self._lock:
            if tool_name in self._approved:
                return None
            loop = asyncio.get_event_loop()
            approved = await loop.run_in_executor(
                None, self._approval_fn, agent_name, tool_name
            )
            if not approved:
                return (
                    f"[Human-in-the-loop] The human declined approval for tool {tool_name!r}. "
                    "Do not retry this exact tool call without changing your approach or asking the user. "
                    "Use write_todos to update your plan and continue with other steps or tools."
                )
            self._approved.add(tool_name)
        return None
```
---
./src/deepx/middleware/filesystem.py
```python
from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from agents.agent import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext
from agents.tool import FunctionTool, Tool

from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.models import Plan

LARGE_OUTPUT_THRESHOLD = 80_000


class FilesystemHooks(RunHooksBase[AgentContext, Agent[AgentContext]]):
    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend

    async def on_agent_start(
        self,
        context: AgentHookContext[AgentContext],
        agent: Agent[AgentContext],
    ) -> None:
        context.context.agent_name = agent.name
        context.context.plan.agent_name = agent.name
        if context.context.resume:
            saved = self._backend.load_plan(context.context.session_id, agent.name)
            if saved:
                context.context.plan = Plan.model_validate_json(saved)
        if not context.context.memory:
            raw = self._backend.read_store("AGENTS.md")
            if raw:
                context.context.memory = raw


def _make_evicting_invoke(original_invoke: Any, backend: BackendProtocol) -> Any:
    async def invoke(ctx: Any, args_json: str) -> Any:
        result = await original_invoke(ctx, args_json)
        text = str(result)
        if len(text) <= LARGE_OUTPUT_THRESHOLD:
            return result
        session_id = ctx.context.session_id
        call_id = uuid.uuid4().hex[:12]
        rel = f"large_tool_results/{call_id}.txt"
        backend.write(session_id, rel, text)
        preview = "\n".join(text.splitlines()[:10])
        return (
            f"[Output was large and saved to /{rel}. Use read_file to access it. "
            f"Preview:\n{preview}]"
        )

    return invoke


def wrap_tools_with_large_output_eviction(
    tools: list[Tool],
    backend: BackendProtocol,
) -> list[Tool]:
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            inv = _make_evicting_invoke(tool.on_invoke_tool, backend)
            out.append(dataclasses.replace(tool, on_invoke_tool=inv))
        else:
            out.append(tool)
    return out


def _make_logged_invoke(
    original_invoke: Any,
    tool_name: str,
    agent_name: str,
    backend: BackendProtocol,
) -> Any:
    async def logged_invoke(ctx: Any, args_json: str) -> Any:
        result = await original_invoke(ctx, args_json)
        session_id = ctx.context.session_id
        call_id = uuid.uuid4().hex[:12]
        backend.save_tool_log(
            session_id,
            {
                "call_id": call_id,
                "tool_name": tool_name,
                "agent_name": agent_name,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": json.loads(args_json) if args_json else {},
                "output": str(result),
                "output_chars": len(str(result)),
            },
        )
        return result

    return logged_invoke


def wrap_tools_for_logging(
    tools: list[Tool],
    backend: BackendProtocol,
    agent_name: str,
) -> list[Tool]:
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            logged = _make_logged_invoke(
                tool.on_invoke_tool, tool.name, agent_name, backend
            )
            out.append(dataclasses.replace(tool, on_invoke_tool=logged))
        else:
            out.append(tool)
    return out


def wrap_tools_for_hitl(
    tools: list[Tool],
    hitl: HumanInTheLoopHooks,
) -> list[Tool]:
    """Wrap sensitive FunctionTools so approval runs before invoke; declines become tool output."""
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool) and tool.name in hitl._sensitive:
            inner = tool.on_invoke_tool
            name = tool.name

            async def hitl_invoke(
                ctx: Any,
                args_json: str,
                *,
                _inner: Any = inner,
                _hitl: HumanInTheLoopHooks = hitl,
                _name: str = name,
            ) -> Any:
                agent_name = getattr(ctx.context, "agent_name", "") or "agent"
                msg = await _hitl.gate_tool(agent_name, _name)
                if msg is not None:
                    return msg
                return await _inner(ctx, args_json)

            out.append(dataclasses.replace(tool, on_invoke_tool=hitl_invoke))
        else:
            out.append(tool)
    return out


def apply_tool_pipeline(
    tools: list[Tool],
    backend: BackendProtocol,
    *,
    agent_name: str,
    debug: bool,
    hitl: HumanInTheLoopHooks | None = None,
) -> list[Tool]:
    wrapped = wrap_tools_with_large_output_eviction(tools, backend)
    if debug:
        wrapped = wrap_tools_for_logging(wrapped, backend, agent_name)
    if hitl is not None:
        wrapped = wrap_tools_for_hitl(wrapped, hitl)
    return wrapped
```
---
./src/deepx/sessions.py
```python
from __future__ import annotations

from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession


def create_session(session_id: str, db_path: str = ":memory:"):
    raw = SQLiteSession(session_id, db_path)
    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=raw,
    )
```
---
./src/deepx/factory.py
```python
from __future__ import annotations

import asyncio
import dataclasses
import re
import uuid
from pathlib import Path
from typing import TypedDict

from agents import Agent, RunContextWrapper, Runner, function_tool
from agents.agent import Agent as AgentType
from agents.lifecycle import RunHooksBase
from agents.result import RunResultStreaming
from agents.tool import Tool

from deepx.backends.filesystem import FilesystemBackend
from deepx.backends.protocol import BackendProtocol
from deepx.context import AgentContext
from deepx.middleware.filesystem import FilesystemHooks, apply_tool_pipeline
from deepx.middleware.hitl import HumanInTheLoopHooks
from deepx.middleware.observability import setup_observability
from deepx.models import Plan
from deepx.sessions import create_session
from deepx.system_prompt import (
    build_system_prompt,
    discover_skills,
    format_skills_for_prompt,
)
from deepx.tools import FILESYSTEM_TOOLS, PLANNING_TOOLS

_HookList = list[RunHooksBase[AgentContext, AgentType[AgentContext]]]


class SubAgentDict(TypedDict, total=False):
    name: str
    description: str
    system_prompt: str
    tools: list
    model: str
    skills: list[str]


def create_deep_agent(
    model: str = "gpt-4o-mini",
    tools: list | None = None,
    *,
    name: str = "agent",
    system_prompt: str = "",
    subagents: list[dict | tuple[Agent, str]] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: type | None = None,
    backend: BackendProtocol | None = None,
    db_path: str = ":memory:",
    interrupt_on: list[str] | None = None,
    debug: bool = False,
    max_turns: int = 1000,
) -> DeepAgentRunner:
    setup_observability()

    resolved_backend = backend or FilesystemBackend(".deepx")
    mem_content = _load_memory(memory, resolved_backend)
    skills_paths = skills or []
    skills_info_main = format_skills_for_prompt(discover_skills(skills_paths))
    user_tools = list(tools or [])
    base_tools = _build_base_tools()

    sub_specs = list(subagents or [])
    has_gp = any(
        isinstance(s, dict) and s.get("name") == "general_purpose" for s in sub_specs
    )
    if not has_gp:
        sub_specs.append(
            {
                "name": "general_purpose",
                "description": (
                    "General-purpose agent for isolated multi-step tasks. "
                    "Has access to the same filesystem and planning tools as the main agent."
                ),
                "system_prompt": "",
                "tools": user_tools,
                "skills": skills_paths,
            }
        )

    skills_by_agent: dict[str, str] = {}
    registry: dict[str, Agent] = {}
    descriptions: dict[str, str] = {}

    for spec in sub_specs:
        if isinstance(spec, tuple):
            ag, desc = spec
            registry[ag.name] = ag
            descriptions[ag.name] = desc
            skills_by_agent[ag.name] = ""
            continue
        an = spec["name"]
        spaths = skills_paths if an == "general_purpose" else list(spec.get("skills", []))
        skills_by_agent[an] = format_skills_for_prompt(discover_skills(spaths))
        descriptions[an] = spec["description"]
        registry[an] = _build_subagent_from_dict(
            spec,
            model=model,
            base_tools=base_tools,
            user_tools=user_tools,
            response_format=response_format,
        )

    hitl = HumanInTheLoopHooks(interrupt_on) if interrupt_on else None
    interrupt_list = list(interrupt_on or [])

    subagent_tools: list[Tool] = []
    for spec in sub_specs:
        an = spec["name"] if isinstance(spec, dict) else spec[0].name
        tool_name = re.sub(r"[^a-zA-Z0-9_]", "_", an)
        subagent_tools.append(
            _make_subagent_tool(
                agent=registry[an],
                tool_name=tool_name,
                description=descriptions[an],
                db_path=db_path,
                max_turns=max_turns,
                backend=resolved_backend,
                hitl=hitl,
                debug=debug,
                memory_default=mem_content,
                skills_info=skills_by_agent.get(an, ""),
                interrupt_tools=interrupt_list,
            )
        )

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_system_prompt(ctx, agent, custom_prompt=system_prompt)

    main_agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=base_tools + user_tools + subagent_tools,
        output_type=response_format,
    )

    return DeepAgentRunner(
        agent=main_agent,
        backend=resolved_backend,
        db_path=db_path,
        max_turns=max_turns,
        hitl=hitl,
        skills_info=skills_info_main,
        memory=mem_content,
        debug=debug,
        agent_name=name,
        interrupt_tools=interrupt_list,
    )


def _build_subagent_from_dict(
    spec: dict,
    *,
    model: str,
    base_tools: list,
    user_tools: list,
    response_format: type | None,
) -> Agent:
    an = spec["name"]
    if an == "general_purpose":
        sub_tools = base_tools + list(spec.get("tools", user_tools))
    else:
        sub_tools = base_tools + list(spec.get("tools", []))
    sub_model = spec.get("model", model)

    def instructions(ctx: RunContextWrapper, agent: Agent) -> str:
        return build_system_prompt(
            ctx, agent, custom_prompt=spec.get("system_prompt", "")
        )

    return Agent(
        name=an,
        instructions=instructions,
        model=sub_model,
        tools=sub_tools,
        output_type=response_format,
    )


def _make_subagent_tool(
    *,
    agent: Agent,
    tool_name: str,
    description: str,
    db_path: str,
    max_turns: int,
    backend: BackendProtocol,
    hitl: HumanInTheLoopHooks | None,
    debug: bool,
    memory_default: str,
    skills_info: str,
    interrupt_tools: list[str],
) -> Tool:
    """Create a named FunctionTool that runs a subagent in an isolated context."""

    async def _invoke(ctx: RunContextWrapper[AgentContext], input: str) -> str:
        sub_ctx = AgentContext(
            session_id=ctx.context.session_id,
            backend=ctx.context.backend,
            agent_name=agent.name,
            memory=memory_default,
            skills_info=skills_info,
            debug=debug,
            hitl_tools=interrupt_tools,
        )
        sub_sid = f"{ctx.context.session_id}:{agent.name}:{uuid.uuid4().hex[:12]}"
        session = create_session(sub_sid, db_path)
        wrapped = apply_tool_pipeline(
            list(agent.tools),
            backend,
            agent_name=agent.name,
            debug=debug,
            hitl=hitl,
        )
        result = await Runner.run(
            dataclasses.replace(agent, tools=wrapped),
            input=input,
            context=sub_ctx,
            session=session,
            hooks=FilesystemHooks(backend),
            max_turns=max_turns,
        )
        return str(result.final_output)

    return function_tool(
        _invoke,
        name_override=tool_name,
        description_override=description,
        use_docstring_info=False,
    )


class DeepAgentRunner:
    def __init__(
        self,
        agent: Agent,
        backend: BackendProtocol,
        db_path: str,
        max_turns: int,
        hitl: HumanInTheLoopHooks | None,
        skills_info: str,
        memory: str,
        debug: bool,
        agent_name: str,
        interrupt_tools: list[str],
    ) -> None:
        self._agent = agent
        self._backend = backend
        self._db_path = db_path
        self._max_turns = max_turns
        self._hitl = hitl
        self._skills_info = skills_info
        self._memory = memory
        self._debug = debug
        self._agent_name = agent_name
        self._interrupt_tools = interrupt_tools

    def _make_ctx(self, session_id: str, resume: bool) -> AgentContext:
        return AgentContext(
            session_id=session_id,
            backend=self._backend,
            agent_name=self._agent_name,
            memory=self._memory,
            skills_info=self._skills_info,
            debug=self._debug,
            hitl_tools=self._interrupt_tools,
            resume=resume,
        )

    def _make_hooks(self) -> RunHooksBase[AgentContext, AgentType[AgentContext]]:
        return FilesystemHooks(self._backend)

    def _prepare_agent(self, ctx: AgentContext) -> Agent:
        wrapped = apply_tool_pipeline(
            list(self._agent.tools),
            self._backend,
            agent_name=self._agent.name,
            debug=self._debug,
            hitl=self._hitl,
        )
        return dataclasses.replace(self._agent, tools=wrapped)

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
    ) -> DeepRunResult:
        sid = session_id or uuid.uuid4().hex
        ctx = self._make_ctx(sid, resume)
        if self._debug:
            self._backend.append_task_log(sid, task)
        session = create_session(sid, self._db_path)
        agent = self._prepare_agent(ctx)
        hooks = self._make_hooks()
        result = await Runner.run(
            agent,
            input=task,
            context=ctx,
            session=session,
            hooks=hooks,
            max_turns=self._max_turns,
        )
        return DeepRunResult(output=result.final_output, session_id=sid, plan=ctx.plan)

    def run_sync(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
    ) -> DeepRunResult:
        coro = self.run(task, session_id=session_id, resume=resume)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)  # type: ignore[return-value]
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()  # type: ignore[return-value]

    async def run_stream(
        self,
        task: str,
        *,
        session_id: str | None = None,
        resume: bool = False,
    ):
        sid = session_id or uuid.uuid4().hex
        ctx = self._make_ctx(sid, resume)
        if self._debug:
            self._backend.append_task_log(sid, task)
        session = create_session(sid, self._db_path)
        agent = self._prepare_agent(ctx)
        hooks = self._make_hooks()
        stream: RunResultStreaming = Runner.run_streamed(
            agent,
            input=task,
            context=ctx,
            session=session,
            hooks=hooks,
            max_turns=self._max_turns,
        )
        async for event in stream.stream_events():
            yield event


class DeepRunResult:
    def __init__(self, output: str, session_id: str, plan: Plan) -> None:
        self.output = output
        self.session_id = session_id
        self.plan = plan

    def __repr__(self) -> str:
        return (
            f"DeepRunResult(session_id={self.session_id!r}, "
            f"output={str(self.output)[:80]!r})"
        )


def _build_base_tools() -> list:
    return [*FILESYSTEM_TOOLS, *PLANNING_TOOLS]


def _load_memory(memory: list[str] | None, backend: BackendProtocol) -> str:
    if not memory:
        return ""
    parts: list[str] = []
    for path in memory:
        p = Path(path)
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
        else:
            rel = path.lstrip("/")
            raw = backend.read_store(rel)
            if raw is not None:
                parts.append(raw)
    return "\n\n".join(parts)


DeepAgent = create_deep_agent
```
