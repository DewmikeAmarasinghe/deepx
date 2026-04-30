# Test demo (`test_demo`)

This directory contains an **example multi-agent setup** for the Deepx repository: an **orchestrator** that delegates to **specialist** runners (**web**, **SQL**, **PDF**, optional **Hugging Face**). It is **not** required to use the **`deepx`** library in your own project; it documents one way to wire several **`DeepAgentRunner`** instances together.

- **Root readme:** [`README.md`](../README.md)
- **Framework:** [`src/deepx/README.md`](../src/deepx/README.md)
- **Terminal chat:** [`src/deepx_cli/README.md`](../src/deepx_cli/README.md) (minimal REPL only).
- **Removed experiments** (Temporal, Chainlit, old MCP wiring): [`removed_features.md`](../removed_features.md).

**Setup:** from the repository root, `uv sync --extra demo`, ensure **`OPENAI_API_KEY`** (and any provider keys your agents need) are available, e.g. via `.env` loaded in the demo entrypoints.

---

## `test_demo/` tree (overview)

```text
test_demo/
├── orchestrator.py        # CLI: --chat, --chat_sync, --session (main demo entry)
├── web_agent.py
├── sql_agent.py
├── pdf_agent.py
├── hf_agent.py
├── sql_tools.py
├── sample_tasks.py
├── __init__.py
├── dbs/
│   ├── agent_dbs/*.db     # per-agent SQLite checkpointers
│   └── test_dbs/*.db      # chinook, northwind sample DBs
├── pdfs/                  # sample PDFs for pdf_agent
├── skills/                # tavily, write-report, pdf, sql, …
└── scripts/               # optional standalone helpers
```

---

## Deliverable outputs: `/_outputs/`

In agent path terms, specialists are instructed to write **human-facing artifacts** under **`/_outputs/`** (see orchestrator system prompt in **`orchestrator.py`**). On a **`FilesystemBackend`** whose **`root_dir`** is the **repository root**, that corresponds to:

**`<repo>/_outputs/`**

Use that tree for **demo deliverables** (reports, extracts, etc.). Large or incidental blobs may also appear under **`/_outputs/large_tool_results/`** when the tool pipeline evicts an oversized tool result (see **`src/deepx/README.md`**).

Session metadata (logs, HITL approvals, `AGENTS.md`, etc.) lives under **`/.deepx/...`** → **`<repo>/.deepx/...`** when the backend’s data root is the repo.

---

## Entrypoint: `orchestrator.py`

- **`REPO_ROOT`** — `Path(__file__).resolve().parents[1]`; used for `sys.path`, DB paths, and **`DEMO_BACKEND = FilesystemBackend(REPO_ROOT)`** (see **`orchestrator.py`**) so agent paths map to real files under the repo.
- **`create_deep_agent`** — Builds the **`orchestrator`** runner with **`subagents`** (web, sql, pdf, optional hf), **`tools`** (e.g. **`render_files`**), **`memory`**, **`checkpointer`** SQLite paths under **`test_demo/dbs/agent_dbs/`**, **`interrupt_on`** as configured per agent, and **`debug`** as set in code.
- **CLI** — **`--chat`** / **`--chat_sync`**, **`--session`** for resume; uses **`deepx_cli`** streaming or sync REPL.

Run examples (from repo root):

```bash
python -m test_demo.orchestrator --chat
python -m test_demo.orchestrator --chat --session <id>
```

---

## Specialist agents (same directory)

| Module             | Role                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------ |
| **`web_agent.py`** | Web / Tavily-oriented runner; often paired with **`LocalShellBackend`** for **`execute`**. |
| **`sql_agent.py`** | Read-only SQL over sample DBs under **`test_demo/dbs/test_dbs/`**.                         |
| **`pdf_agent.py`** | PDF workflows + skills under **`test_demo/skills/pdf/`**.                                  |
| **`hf_agent.py`**  | Optional HF Hub runner when **`HF_TOKEN`** / config allows.                                |

Each file constructs a **`DeepAgentRunner`** via **`create_deep_agent`** with its own **`checkpointer`** path, **`backend`**, **`tools`**, and **`interrupt_on`** list.

---

## Other files

| Path                                      | Role                                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------- |
| **`sample_tasks.py`**                     | Example programmatic tasks / smoke flows against the demo agents.               |
| **`sql_tools.py`**                        | SQL-related tools used by the SQL agent.                                        |
| **`scripts/extract_and_analyze_pdfs.py`** | Standalone script (PDF pipeline helper), not the main agent entrypoint.         |
| **`skills/pdf/scripts/*.py`**             | Skill-bundled utilities referenced from **`test_demo/skills/pdf`**.             |
| **`dbs/agent_dbs/*.db`**                  | Per-agent SQLite **conversation** stores (OpenAI Agents session checkpointers). |
| **`dbs/test_dbs/*.db`**                   | Sample **business** databases for SQL demos.                                    |

---

## Debugging and logs in the demo

When a runner is created with **`debug=True`** (as in the current orchestrator configuration), the framework attaches **`SessionToolLogHooks`** and writes per-tool JSON under:

**`/.deepx/sessions/<session_id>/logs/tools/<tool_name>/<n>.json`**

Plan snapshots under **`logs/plans/`** and plan event append are also **debug-gated** in **`deepx.tools.planning`**. Setting **`debug=False`** disables those writes; **HITL `approvals.json`** persistence is separate and follows **`deepx.middleware.hitl`** rules.

---

## Note on backends in the demo

Specialists may use **`LocalShellBackend`**, **`FilesystemBackend`**, or **`InMemoryBackend`** depending on the module. **HITL persistence** for a gated tool uses the **`AgentContext.backend`** of the agent executing that tool, so approvals for **`web_agent`’s `execute`** are stored via the **web** runner’s backend—not necessarily the orchestrator’s. See **`src/deepx/README.md`** for details.
