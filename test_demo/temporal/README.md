# Temporal orchestrator demo

Runs `DeepxOrchestratorWorkflow`, which awaits `Runner.run` inside workflow code with the Temporal OpenAI Agents plugin (fine-grained activities for model/tool steps).

## Workflow vs local SQLite

The workflow uses `orchestrator_runner_workflow` from [`test_demo/orchestrator.py`](../orchestrator.py) with **`checkpointer=":memory:"`**. Temporal’s durable asyncio loop does not support `asyncio.to_thread` / a default executor, which the OpenAI Agents **file-backed** `SQLiteSession` relies on—so on-disk session DBs cannot be used inside the workflow sandbox.

Local runs (`USE_TEMPORAL` unset/false) use `orchestrator_runner` with `orchestrator.db` and normal subagent SQLite files.

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
