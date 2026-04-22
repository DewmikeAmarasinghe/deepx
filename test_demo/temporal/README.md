# Temporal orchestrator demo

Runs `DeepxOrchestratorWorkflow`, which awaits `Runner.run` inside workflow code with the Temporal OpenAI Agents plugin (plugin-managed activities for model/SDK steps).

See the official cookbook: [Durable Agent with Tools - OpenAI Agents SDK](https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python).

## Sessions (`checkpointer`)

The OpenAI Agents `SQLiteSession` uses `asyncio.to_thread` for storage I/O. Event loops used by durable workflow code often **do not** support a default thread-pool executor, so SQLite-backed sessions can fail there.

Workflow runs use `orchestrator_runner_workflow` from [`test_demo/orchestrator.py`](../orchestrator.py) with **`checkpointer="memory"`** on the orchestrator and specialists, which uses an in-process async list store (`AsyncListSession` under `OpenAIResponsesCompactionSession`) instead of SQLite.

Local runs (`USE_TEMPORAL` unset/false) use `orchestrator_runner` with **file-backed** SQLite paths under `test_demo/dbs/agent_dbs/`.

Legacy `":memory:"` is treated as the same in-memory list session as `"memory"` (see `deepx.sessions.create_session`).

## Human-in-the-loop (HITL)

Workflow code must not block on `input()`. Pending tool approvals are exposed via the **`hitl_pending`** query; the orchestrator CLI runs a small background task that prompts on **this** terminal and sends **`hitl_approval`** signals to the workflow.

## Multi-turn chat (`USE_TEMPORAL=1`)

The CLI starts (or reattaches to) a long-lived workflow id **`deepx-chat-{SESSION_ID}`** and sends each line with the **`user_message`** signal so in-memory session history stays in one workflow execution. Use **`/bye`** to signal **`end_session`** and end the loop.

## What shows up in Temporal history

- **`OpenAIAgentsPlugin`** turns **LLM / SDK-managed steps** into plugin activities (e.g. `invoke_model_activity` labeled with agent names such as `orchestrator` or `web_agent`).
- **Ordinary `function_tool` tools** (e.g. `read_file`, `web_search`) are **not** separate Temporal activities unless you wrap them with **`activity_as_tool`** and register the activity callables on **`Worker(..., activities=[...])`** (see the cookbook). This demo does **not** wrap every Deepx tool that way; tools run inside those model/tool batches.
- **Retries** apply at **activity** boundaries the plugin defines (e.g. a model step), not per arbitrary Python function tool, unless that tool is promoted to its own activity.

## Tools and `asyncio.to_thread`

The OpenAI Agents SDK runs **synchronous** `@function_tool` handlers via `asyncio.to_thread`. That also breaks in the same restricted loops. Deepx built-in tools are **`async def`** so the handler runs without `to_thread`.

## Streaming

**Temporal mode** uses `run_binding_until_settled` from the workflow (no token streaming). **Local CLI** uses `run_stream_until_settled` with `stream_text=True` when `USE_TEMPORAL` is off.

## LangSmith in the worker

The worker calls `load_dotenv()` then sets **`LANGSMITH_TRACING=false`** only when that variable is **unset**, which avoids noisy “No active trace” / invalid parent span errors while still allowing you to enable tracing by setting `LANGSMITH_TRACING=true` (and `LANGSMITH_API_KEY`) in the environment or `.env`.

## Running

Start Temporal, then:

```bash
uv run --extra demo python -m test_demo.temporal.worker
```

In another shell:

```bash
USE_TEMPORAL=1 uv run --extra demo python -m test_demo.orchestrator --chat
```

Tool approvals are prompted in the **CLI** process; the worker receives choices via **signals**.
