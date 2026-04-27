# Deepx framework (`deepx`)

This package implements the **Deepx** agent framework: a **`DeepAgentRunner`** built from **`create_deep_agent`**, running on the OpenAI Agents SDK with a pluggable **`BackendProtocol`**, composed **run hooks**, a **tool pipeline** (large-result eviction + HITL wrapping), and a **dynamic system prompt** (`deepx.system_prompt.build_system_prompt`).

Higher-level docs: **[repository `README.md`](../../README.md)** (links to this file, CLI, and `test_demo`).

---

## What the framework does

1. **Workspace scope** — All file tools operate through a **backend**. Paths are **agent paths** rooted at the backend’s **`root_dir`** (project root), plus a metadata tree at **`/.deepx/...`** mapped to `<root_dir>/.deepx/...` on disk for filesystem backends.
2. **Built-in tools** — `ls`, `read_file`, `write_file`, `edit_file`, `grep`, `glob`, **`save_memory`**, **`write_todos`**, **`think_tool`**, and optionally **`execute`** when the backend is a **`LocalShellBackend`** (`deepx.tools.builtin_tools_for_backend`).
3. **Subagents as tools** — Each **`DeepAgentRunner`** in **`subagents=`** becomes an SDK **`function_tool`** that runs a nested **`Runner.run`** with that runner’s own backend, memory, and `debug`, while **reusing the parent `session_id` and `Hitl`** (`deepx.factory._subagent_tool_from_runner`).
4. **Sessions** — **`checkpointer`** is a SQLite path (or `":memory:"`) used by **`deepx.sessions.create_session`**, which wraps **`SQLiteSession`** in **`OpenAIResponsesCompactionSession`** with a compaction rule at ~90% of the model context window.
5. **Human-in-the-loop** — Tool names listed in **`interrupt_on`** are wrapped so each invocation can consult **`deepx.middleware.hitl.Hitl`** before running. Approvals keyed by **`(agent_name, tool_name)`** can be persisted under **`/.deepx/sessions/<session_id>/approvals.json`** via the **tool runner’s** `backend` (important when the orchestrator and specialists use different backends).

---

## Backends define scope

Backends implement **`deepx.backends.protocol.BackendProtocol`**: `ls`, `read`, `grep`, `glob`, `write`, `edit`, and **`execute`** (optional / may return a fixed “not available” string).

| Backend | Module | Role |
|---------|--------|------|
| **`FilesystemBackend`** | `deepx.backends.filesystem` | Maps agent paths under **`root_dir`**; **`.deepx`** content is **not** visible to normal file tools (only via `/.deepx/...` agent paths). **`execute`** returns a message that shell is unavailable—use **`LocalShellBackend`** for `execute`. |
| **`LocalShellBackend`** | `deepx.backends.local_shell` | Subclass of **`FilesystemBackend`**; **`execute`** runs **`subprocess.run(..., shell=True, cwd=root_dir)`** with timeout capped at 600s. |
| **`InMemoryBackend`** | `deepx.backends.memory` | In-process dict storage with the **same path rules** as the filesystem backend (including **`/.deepx/...`**). Useful for tests or a virtual workspace; pair with a real **`root_dir`** if you want **`data_root`** aligned to a repo (e.g. `<repo>/.deepx` on disk is not automatic unless you mirror with a filesystem backend or custom logic). |

Shared helpers and constants live in **`deepx.backends.utils`** (e.g. **`data_root_for_host`**, **`data_root_as_agent_path`**, **`OUTPUTS_PREFIX`** `/_outputs`, **`OUTPUTS_LARGE_TOOL_RESULTS_PREFIX`**, **`MAX_READ_FILE_LINES`**, tool-result eviction messages, **`resolve_root_dir`**, **`resolve_data_root`**, **`resolve_backend_paths`**, and path normalization helpers used by backends).

**`BackendProtocol.resolve_path`** — Optional host path for an agent path; used when tooling needs a real filesystem path.

---

## Agent-visible output locations

- **`/_outputs/`** — Conventional tree for **user-facing deliverables** the model can write (replace rules are relaxed under this prefix per backend implementation). In a demo with **`FilesystemBackend(REPO_ROOT)`**, this is `<repo>/_outputs/`.
- **`/_outputs/large_tool_results/`** — Large tool outputs may be spilled here by **`deepx.middleware.tool_pipeline`** with a preview in the tool result.
- **`/.deepx/...`** — Session metadata: e.g. **`sessions/<id>/approvals.json`** (HITL), **`sessions/<id>/logs/...`** (when logging is enabled—see **Debug** below), **`AGENTS.md`** (via **`save_memory`**), plan snapshots under **`logs/plans/`** when **`debug=True`**.

---

## Middleware

Middleware is mostly **run hooks** (SDK **`RunHooksBase`**) and **tool wrappers** applied when the agent is prepared.

| Piece | Module | Purpose |
|-------|--------|---------|
| **`FilesystemHooks`** | `deepx.middleware.filesystem` | On agent start: sets **`context.agent_name`**, syncs **`plan.agent_name`**, and if **`context.resume`**, loads a saved plan JSON via **`run_log_load_plan`**. |
| **`SessionToolLogHooks`** | `deepx.middleware.logs` | On tool end: writes one JSON file per call under **`/.deepx/sessions/<id>/logs/tools/<tool_name>/<n>.json`**. **Only registered when `debug=True`** on the runner (`DeepAgentRunner._make_hooks`). |
| **`apply_tool_pipeline`** | `deepx.middleware.tool_pipeline` | Wraps **`FunctionTool`** instances: eviction of huge results to **`/_outputs/large_tool_results/`**, then **`wrap_tools_for_hitl`** for **`interrupt_on`** tools. |
| **`Hitl` / `wrap_tools_for_hitl`** | `deepx.middleware.hitl` | Prompt policy (async callback); **`consult(..., tool_backend=ac.backend)`** so persistence uses the **agent that owns the tool**. |
| **`compose_run_hooks` / `ChainedRunHooks`** | `deepx.middleware.run_hooks` | Runs multiple hook objects in sequence. |
| **`setup_observability`** | `deepx.middleware.observability` | Called from **`create_deep_agent`**; registers LangSmith tracing for the Agents SDK when **`LANGSMITH_API_KEY`** is set. |

**Plan persistence vs debug:** **`write_todos`** always updates in-memory **`Plan`**. Calls to **`run_log_save_plan`** and **`run_log_append_plan_event`** run **only when `ctx.context.debug` is True** (`deepx.tools.planning`). So **`sessions/.../logs/plans/`** and plan event JSON are **debug-gated**; **`SessionToolLogHooks`** is also **debug-gated**.

---

## Core types and entrypoints

- **`create_deep_agent`** — `deepx.factory.create_deep_agent` (alias **`DeepAgent`**).
- **`DeepAgentRunner`** — Holds the SDK **`Agent`**, backend, checkpointer path, `max_turns`, skill roots, memory string, `debug`, `interrupt_on`, etc. **`bind(session_id, resume=..., hitl=...)`** returns **`DeepRunBinding`**.
- **`DeepRunBinding`** — Prepared agent, **`create_session(session_id, checkpointer)`**, **`AgentContext`** (`ctx`), optional **`hitl.attach_session(runner._backend, session_id)`** for loading approvals from the **bound** runner’s backend.
- **`AgentContext`** — `deepx.context.AgentContext`: `session_id`, `backend`, `agent_name`, `plan`, `memory`, `skills`, `debug`, `resume`, `hitl`, `interrupt_on`.
- **`Hitl`** — Exported from **`deepx.factory`** (and **`deepx.middleware.hitl`**).

---

## `create_deep_agent` parameters

Below matches **`deepx.factory.create_deep_agent`** as implemented. Several optional fields are passed through to the OpenAI Agents **`Agent`** constructor (same general ideas as in the upstream SDK: model tuning, guardrails, hooks, tool-use behavior, dynamic prompts).

| Parameter | Default / notes |
|-----------|-----------------|
| **`model`** | `DEFAULT_MODEL` (`"gpt-5-mini"`). |
| **`tools`** | Extra user **`Tool`** / **`FunctionTool`** instances (in addition to built-ins and subagent tools). |
| **`name`** | Agent name (default `"agent"`). |
| **`description`** | Used for subagent tool description when this runner is exposed as a tool. |
| **`system_prompt`** | **Role / task** text only; merged into the full prompt by **`build_system_prompt`**. |
| **`subagents`** | Sequence of **`DeepAgentRunner`**; each becomes a **`function_tool`**. |
| **`skills`** | List of **directory paths** (skill roots); resolved and cataloged for the prompt. |
| **`memory`** | List of **file paths** (not inline prose); loaded in order, joined with `\n\n`, passed as **`AgentContext.memory`**. Relative paths: backend **`resolve_root_dir`** first, then **cwd** (`_load_memory`). |
| **`response_format`** | Passed to **`Agent`** as **`output_type`** (structured output / Pydantic model type). |
| **`backend`** | **`BackendProtocol`**; default **`FilesystemBackend(Path.cwd())`** if omitted. |
| **`checkpointer`** | SQLite DB path or `":memory:"`; stripped; used by **`create_session`**. |
| **`debug`** | **`True`** by default. When **`True`**, **`SessionToolLogHooks`** is attached and plan files/events under **`logs/`** are written when todos change (see planning module). |
| **`max_turns`** | Passed to **`Runner.run`** / nested runs (default `1000`). |
| **`run_hooks`** | Extra **`RunHooksBase`** instances, **composed after** **`FilesystemHooks`** (and after **`SessionToolLogHooks`** if `debug`). |
| **`include_general_purpose`** | If **`True`** and no subagent named `general_purpose`, appends an auto-created general-purpose **`DeepAgentRunner`** sharing tools/skills/memory/backend/checkpointer. |
| **`model_settings`** | Forwarded to **`Agent`** as **`model_settings`**. |
| **`input_guardrails`** / **`output_guardrails`** | Forwarded to **`Agent`**. |
| **`agent_hooks`** | Forwarded as **`Agent.hooks`** (per-agent lifecycle hooks in the SDK). |
| **`tool_use_behavior`** | Forwarded to **`Agent`**. |
| **`reset_tool_choice`** | Forwarded to **`Agent`**. |
| **`prompt`** | Forwarded to **`Agent`** as **`prompt`** (SDK dynamic prompt / override). |
| **`interrupt_on`** | List of tool **name strings**; must exist on the assembled tool list or **`ValueError`**. Wrapped for HITL via **`apply_tool_pipeline`**. |

---

## Package file map (`src/deepx/`)

| Path | Role |
|------|------|
| **`__init__.py`** | Public exports: **`create_deep_agent`**, **`DeepAgent`**, **`DeepAgentRunner`**, **`DeepRunBinding`**, **`DeepRunResult`**, backends, **`BackendProtocol`**. |
| **`_version.py`** | Package version (Hatch reads this). |
| **`factory.py`** | **`create_deep_agent`**, **`DeepAgentRunner`**, **`DeepRunBinding`**, **`DeepRunResult`**, **`_load_memory`**, subagent tool wiring, **`Hitl`** export in module **`__all__`**. |
| **`context.py`** | **`AgentContext`** dataclass. |
| **`sessions.py`** | **`create_session`** — SQLite + compaction session. |
| **`system_prompt.py`** | **`build_system_prompt`**, prompt sections, skills discovery helpers. |
| **`backends/protocol.py`** | **`BackendProtocol`**, result dataclasses, eviction constants. |
| **`backends/filesystem.py`** | **`FilesystemBackend`**. |
| **`backends/local_shell.py`** | **`LocalShellBackend`**. |
| **`backends/memory.py`** | **`InMemoryBackend`**. |
| **`backends/utils.py`** | Paths, eviction helpers, **`MAX_READ_FILE_LINES`**, backend path resolution. |
| **`middleware/__init__.py`** | Lazy exports for middleware symbols. |
| **`middleware/filesystem.py`** | **`FilesystemHooks`**. |
| **`middleware/logs.py`** | Plan + tool JSON logging helpers; **`SessionToolLogHooks`**. |
| **`middleware/hitl.py`** | **`Hitl`**, **`HitlRequest`**, **`HitlDecision`**, **`wrap_tools_for_hitl`**. |
| **`middleware/tool_pipeline.py`** | Large-result eviction + HITL wrapping. |
| **`middleware/run_hooks.py`** | **`compose_run_hooks`**, **`ChainedRunHooks`**. |
| **`middleware/observability.py`** | **`setup_observability`**. |
| **`tools/__init__.py`** | **`builtin_tools_for_backend`**, tool group constants. |
| **`tools/filesystem.py`** | File tools (`ls`, `read_file`, …). |
| **`tools/execute.py`** | **`execute`** tool (uses backend **`execute`**). |
| **`tools/agent_memory.py`** | **`save_memory`**. |
| **`tools/planning.py`** | **`write_todos`**, **`think_tool`**, **`Plan`** model. |

---

## Nested specialist sessions

Subagent tool calls use a **derived** session id:

`f"{parent_session_id}:{subagent_agent.name}:{tool_call_id}"`

with the **child runner’s** **`checkpointer`** (`deepx.factory._subagent_tool_from_runner`). The **CLI session id** you pass to **`bind`** remains the parent’s; nested SQLite sessions are separate rows/DBs as configured per runner.

---

## Imports

```python
from deepx import (
    create_deep_agent,
    DeepAgentRunner,
    DeepRunBinding,
    FilesystemBackend,
    LocalShellBackend,
    InMemoryBackend,
    BackendProtocol,
)
from deepx.factory import Hitl  # HITL coordinator
```

For middleware symbols, prefer **`deepx.middleware`** lazy attributes or direct submodule imports (see **`middleware/__init__.py`**).
