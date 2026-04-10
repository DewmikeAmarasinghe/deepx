# deepx — Agent Harness Built on OpenAI Agents SDK

`deepx` is a generic Python agent harness, inspired by `langchain/deepagents`, built on top of the
`openai-agents` SDK. It gives any agent built with this framework:

- Planning tools (`write_todos`, `update_todos`) plus `think_tool`
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
    checkpointer="agent.db",
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
        ├── plans/
        │   └── {agent_name}.json    ← current Plan model (todos + metadata)
        │
        ├── files/                   ← files written by agents (write_file, outputs, artifacts)
        │   ├── research/            ← example: structured outputs (e.g., markdown reports)
        │   ├── large_tool_results/  ← large outputs stored as blobs (referenced elsewhere)
        │   └── ...                  ← any other agent-created files
        │
        ├── logs/                    ← observability + execution traces
        │   ├── plans.json           ← aggregated plan updates/events across the session
        │   │
        │   └── tools/               ← auto-logged tool call I/O
        │       └── {tool_category}/ ← generic grouping (NOT specific tool names)
        │           └── 1.json, 2.json, …   ← sequential logs (include `call_id`, input, output)
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
- `write_todos(todos)` — replace the full plan (ids assigned `1`…`n`)
- `update_todos(patches)` — patch existing todos by id (`title` / `status`)

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
