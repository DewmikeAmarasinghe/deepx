# Temporal orchestrator demo

Runs `DeepxOrchestratorWorkflow`, which awaits `Runner.run` inside workflow code with the Temporal OpenAI Agents plugin (fine-grained activities for model/tool steps).

## Workflow vs local sessions

Temporal’s durable asyncio loop does not support `asyncio.to_thread` / a default executor. The OpenAI Agents **`SQLiteSession`** implementation uses a thread pool for `get_items` / `add_items` even when the DSN is `:memory:`, so **SQLite-backed sessions cannot run inside workflow code**.

Workflow runs use `orchestrator_runner_workflow` from [`test_demo/orchestrator.py`](../orchestrator.py) with **`temporal_workflow=True`**, which builds an **async in-memory list session** (`AsyncListSession` under `OpenAIResponsesCompactionSession`) instead of SQLite.

Local runs (`USE_TEMPORAL` unset/false) use `orchestrator_runner` with **`temporal_workflow=False`**: normal file-backed SQLite checkpointers under `test_demo/dbs/agent_dbs/`.

## Streaming

**Temporal mode** uses `run_binding_until_settled` from the workflow (no token streaming). **Local CLI** uses `run_stream_until_settled` with `stream_text=True` when `USE_TEMPORAL` is off.

## Running

Start Temporal, then:

```bash
uv run --extra demo python -m test_demo.temporal.worker
```

In another shell:

```bash
USE_TEMPORAL=1 uv run --extra demo python -m test_demo.orchestrator --chat
```

Tool approvals are read from the **worker** process stdin when running interactively.
