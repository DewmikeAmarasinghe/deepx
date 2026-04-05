# deepx

a minimal agent harness built on [openai-agents SDK](https://github.com/openai/openai-agents-python). inspired by [deepagents](https://github.com/langchain-ai/deepagents), no langchain dependency.

gives any agent:
- planning (`write_todos`)
- virtual filesystem (`write_file`, `read_file`, `ls`, ...)
- persistent memory across sessions
- automatic tool output interception — large results saved to VFS, agent gets a path + preview
- observability — every tool call logged to `/obs/*.json` automatically
- subagent delegation — any `Agent` becomes a callable tool
- skills loading from SKILL.md files
- context compaction via OpenAI Responses API

## install

```bash
uv add deepx
# or
pip install deepx
```

## usage

```python
import asyncio
from agents import Agent, function_tool
from deepx import create_agent, as_tool

@function_tool
async def search_web(ctx, query: str) -> str:
    # your implementation
    return f"results for: {query}"

sql_agent = Agent(
    name="sql_agent",
    instructions="You write SQL queries. Read /db/schema.md for the schema first.",
    model="gpt-4o-mini",
)

runner = create_agent(
    model="gpt-4o",
    tools=[search_web],
    subagents=[
        (sql_agent, "query the database for structured records"),
    ],
    skills_path="./skills/",
    memory_path="./memory/AGENTS.md",
    system_prompt="You are a research assistant.",
    db_path="agent.db",
)

async def main():
    result = await runner.run(
        task="research competitor pricing and compare with our database records",
        session_id="session_001",
        resume=False,
    )
    print(result.output)
    print(f"tokens used: {result.token_usage}")
    print(f"files created: {list(result.vfs.keys())}")

asyncio.run(main())
```

## resuming a session

```python
result = await runner.run(
    task="continue the research",
    session_id="session_001",
    resume=True,
)
```

## human in the loop

```python
runner = create_agent(
    model="gpt-4o",
    tools=[...],
    hitl_tools={"write_file", "execute_command"},
)
```

this pauses before any call to `write_file` or `execute_command` and prompts for CLI approval. pass `hitl_approval_fn=your_async_fn` to use a custom approval flow (webhook, queue, etc).

## skills

create a directory structure:
```
skills/
└── my-skill/
    └── SKILL.md
```

SKILL.md format:
```markdown
---
name: my-skill
description: what this skill does and when to use it
---
# instructions
...
```

the agent sees skill names and descriptions in its system prompt. it reads the full SKILL.md on demand using `read_file`.

## what gets stored where

| data | where |
|---|---|
| conversation history | SQLite `agent_messages` table (managed by openai-agents) |
| VFS files | SQLite `vfs_files` table (managed by deepx) |
| tool call log | SQLite `step_log` table + in-memory `result.step_log` |
| persistent memory | disk file at `memory_path` |
| skills | disk files, read-only |