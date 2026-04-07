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
| 23 | Rewrite `agent.py` test | `agent.py` |