# deepx — agent harness built on openai-agents SDK

## what this is

deepx is a minimal Python package that wraps the openai-agents SDK into a batteries-included agent harness. it is inspired by langchain/deepagents but uses no langchain code whatsoever. the goal is to give any agent planning, a virtual filesystem (VFS), context-aware memory, subagent delegation, automatic tool output interception, observability, and server-side context compaction — all with a single function call.

## strict rules for this codebase

- no comments anywhere in the code — not a single line
- no docstrings
- no type: ignore
- no noqa
- no unnecessary abstractions
- keep every file as short as possible
- use plain python — no metaclasses, no descriptors, no magic
- all async — every public function that touches IO is async
- use dataclasses for all data structures, pydantic only where openai-agents SDK requires it
- do not add dependencies beyond what pyproject.toml lists
- do not import from langchain, langgraph, or any langchain ecosystem package

## dependencies (what is available)

- `openai-agents` — the SDK, provides Agent, Runner, RunHooks, RunContextWrapper, SQLiteSession, function_tool
- `openai-agents[redis]` — optional, for RedisSession
- `aiosqlite` — async sqlite, used by SQLiteSession and our VFSStore
- `pyyaml` — for parsing SKILL.md frontmatter

## key architectural decisions

### context object
every run gets a single `AgentContext` dataclass passed as `context=` to `Runner.run()`. all agents, all subagents, and all tools in a run share the same context object via `RunContextWrapper.context`. this is the shared memory bus. no global state.

### tool interception
openai-agents SDK `on_tool_end` hook receives the result AFTER it was returned to the agent loop — it cannot modify it. so we intercept at the tool wrapper level. `ToolInterceptor.wrap(tool_fn)` returns a new async function that: runs the original function, checks result size, if large — saves full result to `ctx.vfs` at `/outputs/{tool_name}_{step}.md`, always saves result to `/obs/{step}.json` for observability, then returns a truncated result string to the agent that includes the file path and an 8-line preview. the agent always knows where to find the full result.

### VFS
`ctx.context.vfs` is a `dict[str, str]` — path → content. it is ephemeral during the run. `DeepRunner` loads it from SQLite at run start (if resuming) and saves it back at run end. subagents share the parent's VFS dict — they see all files the parent wrote.

### session + compaction
`session_factory()` returns `OpenAIResponsesCompactionSession(SQLiteSession(session_id, db_path))`. this handles conversation history persistence and automatic compaction. the agent never knows this exists.

### instructions callable
`Agent(instructions=build_instructions)` — openai-agents calls this function before every LLM turn, passing `(RunContextWrapper, Agent)`. `build_instructions` reads from `ctx.context` and assembles the full system prompt including: base prompt, custom user prompt, current todos, workspace file list, memory, skills. this is the middleware equivalent for system prompt injection.

### subagents as tools
`as_tool(agent, description)` returns a `@function_tool` decorated async function that calls `Runner.run(agent, input=task, context=ctx.context)` and returns `result.final_output` as a string. the subagent shares the context, so it reads/writes to the same VFS. the parent agent sees only the final output string.

### storage
two SQLite tables in the same `agent.db` file:
- `agent_sessions` + `agent_messages` — managed by openai-agents SQLiteSession
- `vfs_files(session_id, path, content, modified_at)` — managed by our VFSStore
- `step_log(session_id, step, data_json)` — managed by our VFSStore

### the public API
```python
from deepx import create_agent

runner = create_agent(
    model="gpt-4o",
    tools=[my_tool],
    subagents=[
        (sql_agent, "query the database"),
        (web_agent, "search and fetch web pages"),
    ],
    skills_path="./skills/",
    memory_path="./memory/AGENTS.md",
    system_prompt="You are a research assistant.",
)

result = await runner.run(task="...", session_id="user_123")
print(result.output)
print(result.vfs)         # all files written during run
print(result.step_log)    # every tool call input/output
```

## file responsibilities (implement exactly this, nothing more)

### deepx/context.py
define `AgentContext` dataclass with these fields only:
- session_id: str
- vfs: dict[str, str] = field(default_factory=dict)
- todos: list[str] = field(default_factory=list)
- visited_urls: set[str] = field(default_factory=set)
- memory: str = ""
- skills_info: str = ""
- step_log: list[dict] = field(default_factory=list)
- token_usage: int = 0
- _step_counter: int = field(default=0, repr=False)

### deepx/instructions.py
define `build_instructions(ctx: RunContextWrapper[AgentContext], agent: Agent) -> str`
assemble system prompt from sections. each section only appears if the relevant data exists. sections in order:
1. BASE_PROMPT (constant string at top of file — the persona + behavior rules + file organization convention)
2. custom user prompt (if any — stored in a module-level var set by factory)
3. current todos (if ctx.context.todos is non-empty)
4. workspace files (if ctx.context.vfs is non-empty — list sorted paths, max 40)
5. memory (if ctx.context.memory is non-empty)
6. skills (if ctx.context.skills_info is non-empty)
7. visited urls (if ctx.context.visited_urls is non-empty — max 20)

the base prompt MUST include the file organization convention:
```
/plan.md              current task plan
/memory/notes.md      persistent memory
/research/*.md        web / external data
/db/*.md              sql results
/subagents/*.md       subagent outputs
/outputs/*.md         auto-saved large tool results
/obs/*.json           observability log (auto-written, do not edit)
```

### deepx/middleware/interceptor.py
define `ToolInterceptor` class with:
- `EVICTION_THRESHOLD: int = 40_000` (chars, ~10k tokens)
- `wrap(fn, tool_name: str) -> callable` classmethod
  - returns an async wrapper that:
    1. calls the original fn
    2. increments ctx.context._step_counter
    3. saves result to `/obs/{step:04d}_{tool_name}.json` always (json with tool_name, step, input summary, result preview, full_path)
    4. if len(result) > EVICTION_THRESHOLD: saves full result to `/outputs/{step:04d}_{tool_name}.md`, returns truncated string with path + 8-line preview
    5. else: returns result unchanged
- `apply(tools: list, ctx_ref) -> list` classmethod — wraps a list of function_tool objects

NOTE: the wrapper needs access to ctx but function_tool functions receive ctx as their first arg. the wrapper must be transparent — it must preserve the function signature so openai-agents can still inject ctx. the trick: since all our tools take `ctx: RunContextWrapper[AgentContext]` as their first parameter, the wrapper just passes through all args and reads ctx from args[0].

### deepx/middleware/hooks.py
define `DeepRunHooks(RunHooks[AgentContext])` with:
- `on_agent_end(ctx, agent, output)`: set ctx.context.token_usage = ctx.usage.total_tokens
that is all. observability is handled by ToolInterceptor, not hooks.

### deepx/middleware/hitl.py
define `HITLHooks(RunHooks[AgentContext])`:
- `__init__(self, sensitive_tools: set[str], approval_fn=None)`
  - if approval_fn is None, default to CLI input
- `on_tool_start(ctx, agent, tool)`: if tool.name in sensitive_tools — call approval_fn(tool.name, str(ctx)) — if approval_fn raises or returns False, raise Exception("rejected")

### deepx/tools/planning.py
three function_tool functions: `write_todos`, `mark_done`, `read_todos`
write_todos: takes list[str], sets ctx.context.todos, writes to ctx.context.vfs["/plan.md"], returns formatted string
mark_done: takes index: int, marks item as done (prefix "✓ "), updates /plan.md, returns confirmation
read_todos: no args, returns current plan from ctx.context.vfs.get("/plan.md", "(no plan)")

### deepx/tools/vfs.py
function_tool functions: `write_file`, `read_file`, `edit_file`, `ls`, `append_to_file`
- write_file(ctx, path, content): error if path exists, writes to ctx.context.vfs[path], returns "written: {path}"
- read_file(ctx, path, offset=0, limit=100): reads from ctx.context.vfs, paginates by line, formats with line numbers "  {n}\t{line}", returns error string if not found
- edit_file(ctx, path, old_string, new_string): exact string replacement, error if not found or old_string not in content, returns "edited: {path}"
- ls(ctx, path="/"): lists files in ctx.context.vfs whose paths start with path, returns sorted newline-joined list
- append_to_file(ctx, path, content): creates if not exists, appends \n + content, returns "appended to: {path}"

### deepx/tools/memory.py
two function_tool functions: `update_memory`, `read_memory`
- update_memory(ctx, note): appends "- {note}" to ctx.context.memory, syncs to ctx.context.vfs["/memory/notes.md"], returns "remembered"
- read_memory(ctx): returns ctx.context.memory or "(empty)"

### deepx/tools/shell.py
one function_tool: `execute_command(ctx, command, timeout=30)`
uses subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
combines stdout + stderr (stderr lines prefixed "[stderr] ")
if returncode != 0: appends "\n[exit code: {returncode}]"
result goes through ToolInterceptor logic (large → save to VFS) — but since this is a tool fn itself, the interceptor wraps it at the tool list level in the factory. just return the raw combined output from this function.

### deepx/tools/__init__.py
import all tools from each module, define `CORE_TOOLS: list` containing all of them in this order: write_todos, mark_done, read_todos, write_file, read_file, edit_file, ls, append_to_file, update_memory, read_memory, execute_command

### deepx/agents/subagent.py
define `as_tool(agent: Agent, description: str) -> function_tool`
returns a function_tool that:
- takes ctx: RunContextWrapper[AgentContext], task: str
- runs `await Runner.run(agent, input=task, context=ctx.context)`
- saves result to ctx.context.vfs[f"/subagents/{agent.name}_{ctx.context._step_counter:04d}.md"]
- returns result.final_output

### deepx/skills.py
define `SkillsLoader`:
- `discover(path: str) -> list[dict]` — walks directory, finds SKILL.md files, parses YAML frontmatter (between --- delimiters), returns list of {name, description, path, allowed_tools}
- `format_for_prompt(skills: list[dict]) -> str` — formats as markdown list with name, description, path for each skill

### deepx/session.py
define `session_factory(session_id: str, db_path: str) -> Session`
returns `OpenAIResponsesCompactionSession(SQLiteSession(session_id, db_path))`
import from `agents` and `agents.memory`

### deepx/storage/vfs_store.py
define `VFSStore(db_path: str)`:
- `__init__`: store db_path, create table `vfs_files(session_id TEXT, path TEXT, content TEXT, modified_at TEXT, PRIMARY KEY(session_id, path))` if not exists
- `async save(vfs: dict, session_id: str)`: upsert all entries
- `async load(session_id: str) -> dict[str, str]`: return all entries for session_id as dict
- `async save_step(session_id: str, step: int, data: dict)`: insert into step_log table

### deepx/storage/memory_store.py
define `MemoryStore`:
- `load(path: str) -> str`: read file at path, return content or ""
- `save(content: str, path: str)`: write content to path, create parent dirs

### deepx/runner.py
define `DeepRunResult` dataclass: output, session_id, vfs, step_log, token_usage

define `DeepRunner`:
- `__init__(self, agent, db_path, max_turns, skills_path, memory_path, hitl_tools, hitl_approval_fn)` — store all
- `async run(self, task: str, session_id: str, resume: bool = False) -> DeepRunResult`:
  1. create AgentContext(session_id=session_id)
  2. if resume: ctx.vfs = await VFSStore(db_path).load(session_id)
  3. if memory_path: ctx.memory = MemoryStore().load(memory_path)
  4. if skills_path: load skills, ctx.skills_info = SkillsLoader.format_for_prompt(skills)
  5. session = session_factory(session_id, db_path)
  6. build hooks list: [DeepRunHooks()] + optional HITLHooks if hitl_tools
  7. result = await Runner.run(agent, input=task, context=ctx, session=session, hooks=hooks[0], max_turns=max_turns)
     NOTE: openai-agents Runner.run takes a single hooks object. if we need HITL too, subclass DeepRunHooks and merge both. keep it simple.
  8. await VFSStore(db_path).save(ctx.vfs, session_id)
  9. if memory_path: MemoryStore().save(ctx.memory, memory_path)
  10. return DeepRunResult(output=result.final_output, session_id=session_id, vfs=ctx.vfs, step_log=ctx.step_log, token_usage=ctx.token_usage)

### deepx/agents/factory.py
define `create_agent(*, model, tools=None, subagents=None, skills_path=None, memory_path=None, system_prompt="", db_path="deepx.db", max_turns=200, hitl_tools=None, hitl_approval_fn=None) -> DeepRunner`

- resolve subagent tools: [as_tool(a, d) for a, d in (subagents or [])]
- all_tools = CORE_TOOLS + subagent_tools + (tools or [])
- wrap all_tools with ToolInterceptor.apply(all_tools)
- store system_prompt in instructions module so build_instructions can access it (use a module-level dict keyed by agent name or just a single global _user_prompt — keep simple)
- build_instr = lambda ctx, agent: build_instructions(ctx, agent, user_prompt=system_prompt)
- agent = Agent(name="orchestrator", instructions=build_instr, model=model, tools=wrapped_tools)
- return DeepRunner(agent, db_path, max_turns, skills_path, memory_path, hitl_tools, hitl_approval_fn)

### deepx/__init__.py
expose only: `create_agent`, `AgentContext`, `DeepRunResult`, `as_tool`

## testing the package works

a working usage example (DO NOT put this in the package, it is just for reference):
```python
import asyncio
from agents import Agent, function_tool
from deepx import create_agent, as_tool

@function_tool
async def get_price(ctx, item: str) -> str:
    return f"price of {item} is $42"

sql_agent = Agent(
    name="sql_agent",
    instructions="You write SQL. Read schema from /db/schema.md first.",
    model="gpt-4o-mini",
)

runner = create_agent(
    model="gpt-4o",
    tools=[get_price],
    subagents=[(sql_agent, "query the database for structured records")],
    system_prompt="You are a research assistant.",
    db_path="test.db",
)

async def main():
    result = await runner.run("what is the price of apples", session_id="test_001")
    print(result.output)

asyncio.run(main())
```