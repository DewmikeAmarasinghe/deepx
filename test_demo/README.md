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
├── orchestrator.py        # multi-agent REPL entry (same flags as other agents)
├── web_agent.py
├── sql_agent.py
├── pdf_agent.py
├── hf_agent.py
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

The **built-in system prompt** (`deepx.system_prompt`) tells agents to use **`/_outputs/`** as the **default workspace** when no other path is specified. On a backend rooted at the repo, that is **`<repo>/_outputs/`**.

Large tool returns may also be spilled under **`/_outputs/large_tool_results/`** (see **`src/deepx/README.md`**).

Session metadata (logs, HITL approvals, `AGENTS.md`, etc.) lives under **`/.deepx/...`** → **`<repo>/.deepx/...`** when the backend’s data root is the repo.

---

## Interactive CLI (all demo agents)

**`deepx_cli.cli.run_interactive_cli`** wires **`--chat`** (streaming; default), **`--chat_sync`**, and **`--session`**. **`orchestrator.py`**, **`sql_agent.py`**, **`web_agent.py`**, **`pdf_agent.py`**, and **`hf_agent.py`** call it from their **`main()`**.

- **`REPO_ROOT`** — `Path(__file__).resolve().parents[1]`; used for `sys.path`, DB paths, and backends so agent paths map to real files under the repo.
- The **orchestrator** is built with **`subagents`**, **`render_files`**, and its own checkpointer; specialists are separate **`create_deep_agent`** runners with their own SQLite session files under **`test_demo/dbs/agent_dbs/`**.

Run examples **from the repository root**:

```bash
python test_demo/orchestrator.py --chat
python test_demo/sql_agent.py --chat
python test_demo/web_agent.py --chat --session <id>
python test_demo/hf_agent.py --chat   # requires HF_TOKEN
```

---

## Specialist agents (same directory)

| Module | Role |
|--------|------|
| **`web_agent.py`** | Web / Tavily-oriented runner; often paired with **`LocalShellBackend`** for **`execute`**. |
| **`sql_agent.py`** | **Host `sqlite3`** via **`execute`** over **`test_demo/dbs/test_dbs/`**, guided by **`sql-assistant`** / **`sql-query-generator`** / **`sql-toolkit`**. |
| **`pdf_agent.py`** | PDF workflows + skills under **`test_demo/skills/pdf/`**. |
| **`hf_agent.py`** | Optional HF Hub runner when **`HF_TOKEN`** / config allows. |

Each file constructs a **`DeepAgentRunner`** via **`create_deep_agent`** and exposes **`main()`** for the same interactive CLI as the orchestrator.

---

## Other files

| Path | Role |
|------|------|
| **`sample_tasks.py`** | Example programmatic tasks / smoke flows against the demo agents. |
| **`scripts/extract_and_analyze_pdfs.py`** | Standalone script (PDF pipeline helper), not the main agent entrypoint. |
| **`skills/pdf/scripts/*.py`** | Skill-bundled utilities referenced from **`test_demo/skills/pdf`**. |
| **`dbs/agent_dbs/*.db`** | Per-agent SQLite **conversation** stores (OpenAI Agents session checkpointers). |
| **`dbs/test_dbs/*.db`** | Sample **business** databases for SQL demos. |

---

## Debugging and logs in the demo

When a runner is created with **`debug=True`** (as in the current orchestrator configuration), the framework attaches **`SessionToolLogHooks`** and writes per-tool JSON under:

**`/.deepx/sessions/<session_id>/logs/tools/<tool_name>/<n>.json`**

Plan snapshots under **`logs/plans/`** and plan event append are also **debug-gated** in **`deepx.tools.planning`**. Setting **`debug=False`** disables those writes; **HITL `approvals.json`** persistence is separate and follows **`deepx.middleware.hitl`** rules.

---

## Note on backends in the demo

Specialists may use **`LocalShellBackend`**, **`FilesystemBackend`**, or **`InMemoryBackend`** depending on the module. **HITL persistence** for a gated tool uses the **`AgentContext.backend`** of the agent executing that tool, so approvals for **`web_agent`’s `execute`** are stored via the **web** runner’s backend—not necessarily the orchestrator’s. See **`src/deepx/README.md`** for details.
