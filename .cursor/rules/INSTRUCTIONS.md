# cursor agent instructions — build deepx

read AGENTS.md first. that file contains the full architectural spec. this file contains the implementation order and concrete rules. follow both.

## strict coding rules

- no comments in code, no docstrings, no noqa, no type: ignore
- all functions that touch IO must be async
- use dataclasses everywhere, never pydantic models except where openai-agents SDK requires it
- keep files short — if a file exceeds 120 lines, something is wrong
- no unused imports
- 4-space indentation
- import order: stdlib → third-party → local (ruff handles this)

## implementation order — do these in sequence

### step 1 — create package skeleton

create these empty files (with just `pass` or the minimum import):
```
deepx/__init__.py
deepx/context.py
deepx/instructions.py
deepx/runner.py
deepx/session.py
deepx/skills.py
deepx/middleware/__init__.py
deepx/middleware/interceptor.py
deepx/middleware/hooks.py
deepx/middleware/hitl.py
deepx/tools/__init__.py
deepx/tools/planning.py
deepx/tools/vfs.py
deepx/tools/memory.py
deepx/tools/shell.py
deepx/agents/__init__.py
deepx/agents/factory.py
deepx/agents/subagent.py
deepx/storage/__init__.py
deepx/storage/vfs_store.py
deepx/storage/memory_store.py
```

### step 2 — implement context.py

```python
from dataclasses import dataclass, field

@dataclass
class AgentContext:
    session_id: str
    vfs: dict[str, str] = field(default_factory=dict)
    todos: list[str] = field(default_factory=list)
    visited_urls: set[str] = field(default_factory=set)
    memory: str = ""
    skills_info: str = ""
    step_log: list[dict] = field(default_factory=list)
    token_usage: int = 0
    _step_counter: int = field(default=0, repr=False)
```

### step 3 — implement tools (planning, vfs, memory, shell)

implement each tool file. every tool function:
- is decorated with `@function_tool` from `agents`
- takes `ctx: RunContextWrapper[AgentContext]` as the first parameter (after self if any)
- is async
- accesses state via `ctx.context`

planning.py tools: write_todos, mark_done, read_todos
vfs.py tools: write_file, read_file, edit_file, ls, append_to_file
memory.py tools: update_memory, read_memory
shell.py tools: execute_command

see AGENTS.md for exact behavior of each.

tools/__init__.py must define:
```python
from deepx.tools.planning import write_todos, mark_done, read_todos
from deepx.tools.vfs import write_file, read_file, edit_file, ls, append_to_file
from deepx.tools.memory import update_memory, read_memory
from deepx.tools.shell import execute_command

CORE_TOOLS = [
    write_todos, mark_done, read_todos,
    write_file, read_file, edit_file, ls, append_to_file,
    update_memory, read_memory,
    execute_command,
]
```

### step 4 — implement middleware/interceptor.py

this is the most important file. implement `ToolInterceptor` exactly as described in AGENTS.md.

the wrapper function pattern:
```python
import functools
import json
from datetime import datetime, UTC

EVICTION_THRESHOLD = 40_000

class ToolInterceptor:
    @classmethod
    def wrap(cls, fn, tool_name: str):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            ctx = args[0]  # RunContextWrapper is always first arg
            result = await fn(*args, **kwargs)
            
            ctx.context._step_counter += 1
            step = ctx.context._step_counter
            
            obs_entry = {
                "step": step,
                "tool": tool_name,
                "result_chars": len(str(result)),
                "ts": datetime.now(UTC).isoformat(),
                "preview": str(result)[:300],
            }
            
            obs_path = f"/obs/{step:04d}_{tool_name}.json"
            ctx.context.vfs[obs_path] = json.dumps(obs_entry, indent=2)
            ctx.context.step_log.append(obs_entry)
            
            if len(str(result)) > EVICTION_THRESHOLD:
                out_path = f"/outputs/{step:04d}_{tool_name}.md"
                ctx.context.vfs[out_path] = str(result)
                lines = str(result).splitlines()
                preview = "\n".join(lines[:8])
                return (
                    f"Result saved to {out_path} ({len(lines)} lines).\n"
                    f"Preview:\n{preview}\n"
                    f"...\nUse read_file to access the full result."
                )
            
            return result
        return wrapper

    @classmethod
    def apply(cls, tools: list) -> list:
        wrapped = []
        for tool in tools:
            if hasattr(tool, "fn"):
                tool.fn = cls.wrap(tool.fn, tool.name)
            wrapped.append(tool)
        return wrapped
```

NOTE: check how openai-agents function_tool stores the underlying function. it may be `tool.fn` or accessed differently. inspect the ToolFunction or FunctionTool object from the agents SDK and access the callable correctly. the key is to replace the internal callable with our wrapper.

### step 5 — implement middleware/hooks.py

```python
from agents import RunHooks, RunContextWrapper
from deepx.context import AgentContext

class DeepRunHooks(RunHooks[AgentContext]):
    async def on_agent_end(self, ctx: RunContextWrapper[AgentContext], agent, output):
        ctx.context.token_usage = ctx.usage.total_tokens
```

### step 6 — implement middleware/hitl.py

```python
from agents import RunHooks, RunContextWrapper
from deepx.context import AgentContext

class HITLHooks(RunHooks[AgentContext]):
    def __init__(self, sensitive_tools: set[str], approval_fn=None):
        self.sensitive_tools = sensitive_tools
        self.approval_fn = approval_fn or self._cli_approval

    async def on_tool_start(self, ctx: RunContextWrapper[AgentContext], agent, tool):
        if tool.name not in self.sensitive_tools:
            return
        approved = await self.approval_fn(tool.name)
        if not approved:
            raise Exception(f"human rejected tool call: {tool.name}")

    async def _cli_approval(self, tool_name: str) -> bool:
        answer = input(f"approve {tool_name}? [y/n]: ").strip().lower()
        return answer == "y"
```

### step 7 — implement session.py

```python
from agents import SQLiteSession
from agents.memory import OpenAIResponsesCompactionSession

def session_factory(session_id: str, db_path: str):
    raw = SQLiteSession(session_id, db_path)
    return OpenAIResponsesCompactionSession(
        session_id=session_id,
        underlying_session=raw,
    )
```

NOTE: verify the exact import path for OpenAIResponsesCompactionSession from the openai-agents SDK. it may be `agents.memory` or `agents.extensions.memory`. check the SDK source.

### step 8 — implement storage/vfs_store.py

use aiosqlite directly. two tables:
- vfs_files(session_id TEXT, path TEXT, content TEXT, modified_at TEXT, PRIMARY KEY(session_id, path))
- step_log(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, step INTEGER, data_json TEXT)

```python
import aiosqlite
from datetime import datetime, UTC

class VFSStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def _ensure_tables(self, db):
        await db.execute("""
            CREATE TABLE IF NOT EXISTS vfs_files (
                session_id TEXT,
                path TEXT,
                content TEXT,
                modified_at TEXT,
                PRIMARY KEY (session_id, path)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS step_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                step INTEGER,
                data_json TEXT
            )
        """)
        await db.commit()

    async def save(self, vfs: dict, session_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_tables(db)
            now = datetime.now(UTC).isoformat()
            for path, content in vfs.items():
                await db.execute(
                    "INSERT OR REPLACE INTO vfs_files VALUES (?,?,?,?)",
                    (session_id, path, content, now)
                )
            await db.commit()

    async def load(self, session_id: str) -> dict[str, str]:
        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_tables(db)
            cursor = await db.execute(
                "SELECT path, content FROM vfs_files WHERE session_id=?",
                (session_id,)
            )
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}
```

### step 9 — implement storage/memory_store.py

```python
from pathlib import Path

class MemoryStore:
    def load(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    def save(self, content: str, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
```

### step 10 — implement skills.py

```python
import re
from pathlib import Path
import yaml

class SkillsLoader:
    @staticmethod
    def discover(path: str) -> list[dict]:
        skills = []
        base = Path(path)
        for skill_md in base.rglob("SKILL.md"):
            content = skill_md.read_text(encoding="utf-8")
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if not match:
                continue
            try:
                meta = yaml.safe_load(match.group(1))
            except yaml.YAMLError:
                continue
            if not isinstance(meta, dict):
                continue
            name = str(meta.get("name", "")).strip()
            description = str(meta.get("description", "")).strip()
            if not name or not description:
                continue
            skills.append({
                "name": name,
                "description": description,
                "path": str(skill_md),
                "allowed_tools": meta.get("allowed-tools", ""),
            })
        return skills

    @staticmethod
    def format_for_prompt(skills: list[dict]) -> str:
        if not skills:
            return "(no skills available)"
        lines = []
        for s in skills:
            lines.append(f"- **{s['name']}**: {s['description']}")
            lines.append(f"  → read `{s['path']}` for full instructions")
        return "\n".join(lines)
```

### step 11 — implement instructions.py

the BASE_PROMPT constant — include all of this verbatim:

```
You are a deep autonomous agent capable of long-running, multi-step tasks.

## behavior rules
- call write_todos FIRST before any multi-step task — commit to a plan before acting
- write findings to files using write_file — never hold large content only in conversation
- pass file paths to subagents, not raw content — subagents can read from the shared workspace
- call update_memory for any fact you will need across sessions or restarts
- when tasks are independent, call multiple subagent tools in the same response — they run in parallel
- after major steps, verify output before reporting done
- if a tool output was evicted to /outputs/, use read_file to access it

## workspace file convention
/plan.md              current task plan (written by write_todos)
/memory/notes.md      persistent memory (written by update_memory)
/research/*.md        web scraping and external data
/db/*.md              sql query results
/subagents/*.md       subagent outputs (auto-written when subagents complete)
/outputs/*.md         auto-saved large tool results (too large to display inline)
/obs/*.json           observability log — every tool call recorded here, do not edit
```

then `build_instructions(ctx, agent, user_prompt="") -> str` assembles sections.

### step 12 — implement agents/subagent.py

```python
from agents import Agent, Runner, RunContextWrapper, function_tool
from deepx.context import AgentContext

def as_tool(agent: Agent, description: str):
    async def _run(ctx: RunContextWrapper[AgentContext], task: str) -> str:
        result = await Runner.run(agent, input=task, context=ctx.context)
        step = ctx.context._step_counter
        path = f"/subagents/{agent.name}_{step:04d}.md"
        ctx.context.vfs[path] = result.final_output
        return result.final_output
    
    _run.__name__ = f"run_{agent.name}"
    _run.__doc__ = description
    return function_tool(_run)
```

### step 13 — implement agents/factory.py

see AGENTS.md for the full spec. keep it simple:
1. build subagent tools from the subagents list using as_tool
2. combine CORE_TOOLS + subagent_tools + user tools
3. wrap all with ToolInterceptor.apply()
4. build instructions callable that closes over user_prompt
5. create Agent(name="orchestrator", instructions=build_instr, model=model, tools=wrapped_tools)
6. return DeepRunner(...)

### step 14 — implement runner.py

see AGENTS.md for the full spec.
implement DeepRunResult and DeepRunner as described.

### step 15 — implement __init__.py

```python
from deepx.agents.factory import create_agent
from deepx.agents.subagent import as_tool
from deepx.context import AgentContext
from deepx.runner import DeepRunResult

__all__ = ["create_agent", "as_tool", "AgentContext", "DeepRunResult"]
```

## verification checklist after implementation

1. `from deepx import create_agent` works
2. `runner = create_agent(model="gpt-4o-mini")` returns a DeepRunner
3. ToolInterceptor.apply() actually wraps the tools (check the underlying callable is replaced)
4. VFSStore creates tables on first use and loads/saves correctly
5. session_factory returns a valid session object (check the import path for OpenAIResponsesCompactionSession)
6. as_tool(agent, description) returns a function_tool, not a plain coroutine

## common mistakes to avoid

- do not put blocking IO (sqlite3, open()) in async functions — use aiosqlite for all DB, pathlib for file ops
- do not import from langchain anywhere
- do not add __all__ to every module — only __init__.py needs it
- do not wrap CORE_TOOLS a second time in the factory if they are already wrapped — ToolInterceptor.apply should be called once on the final combined list
- the function_tool decorator from openai-agents inspects the function signature to generate the JSON schema. our wrapper must preserve the original signature with functools.wraps so the schema is generated correctly for the original function, not the wrapper. verify this.
- HITLHooks.on_tool_start is async — the CLI input call must be wrapped in asyncio.to_thread() to avoid blocking the event loop