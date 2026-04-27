# Removed or replaced features

This document records design decisions and features that were **removed** or **replaced** during Deepx development. It is historical context for contributors; the current behavior is described in [`README.md`](README.md) and [`src/deepx/README.md`](src/deepx/README.md).

---

## Subagent “role” prompt branch

A dedicated **subagent role prompt** (treating subagents differently in `build_system_prompt`) was **removed**.

**Why:** Specialists often use **different backends** than the parent orchestrator. They do **not** share one physical filesystem just because the model sees similar path strings—`/report.md` on agent A’s backend is not the same bytes as `/report.md` on agent B’s backend. A prompt that implied “one shared tree” was misleading. Routing and behavior are now expressed through **per-runner** `system_prompt`, **skills**, and **delegation briefs**, not a blanket subagent role string.

```
SUBAGENT_ROLE_PROMPT = """\
You are working as a **specialist subagent**. Your audience is an **orchestrator**, not the end user.

**What you do:** own the specialist work end-to-end — plan with `write_todos`, use skills and
tools, and write **complete, final artifacts** under the project tree (especially `/_outputs/`).

**What you return to the parent:** only
- **Absolute-style paths** to every deliverable you created or updated, and
- a **very short** summary (what you did, what to open, blockers if any).

**Hard limits for the parent message:** do **not** paste full reports, long markdown, raw CLI
dumps, or huge JSON. The orchestrator shows files to the user with **`render_files`** and only
needs your paths plus a brief recap. If a tool result was evicted to `/_outputs/large_tool_results/`,
use **`read_file`** and read them.

Finish the work, then return to the parent only **paths** and a **short summary** as above.

**Quality:** when the user will see output via `render_files`, files must be **complete** (no “see
other file” stubs). Delete scratch files when done; keep final outputs only.

**Planning:** same `write_todos` / `think_tool` rules as the main agent — multi-step work requires
an up-to-date todo list after each major step.\
"""
```

---

## `tool_namespaces` + hosted “tool search” patterns

Experiments with **`tool_namespaces`** and a **tool-search tool** (dynamically surfacing tools) were **removed**.

**Why:** Tools registered that way **did not show up in the LangSmith UI** for this project’s tracing setup. The team optimized for **observability** (explicit tool list, normal `FunctionTool` wiring) over dynamic namespace indirection.

```
import asyncio
from agents import Agent, Runner, function_tool, tool_namespace, ToolSearchTool

@function_tool(defer_loading=True)
def get_order_status(order_id: str) -> str:
    return f"Order {order_id} is shipped."

@function_tool(defer_loading=True)
def get_customer_balance(customer_id: str) -> str:
    return f"Balance for {customer_id} is $50.00."

crm_tools = tool_namespace(
    name="crm",
    description="Tools for customer relationship management.",
    tools=[get_order_status, get_customer_balance]
)

agent = Agent(
    name="Support Assistant",
    instructions="Use CRM tools to help customers.",
    tools=[crm_tools, ToolSearchTool()]
)

async def main():
    result = await Runner.run(agent, "Check status for order 12345.")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```


---

## `Agent.as_tool` for subagents → `function_tool` + `Runner.run`

Subagents are exposed with **`function_tool`** and an inner **`Runner.run`** (see `deepx.factory._subagent_tool_from_runner`), not **`Agent.as_tool`**.

**Why:** With **`.as_tool`**, **checkpointer / intermediate steps** for nested runs were hard to observe and control the way this codebase expects. Even after trying **lower-level OpenAI Agents SDK** APIs to recover nested checkpoints, **LangSmith traces broke after HITL-related flows**. Using an explicit **`function_tool`** + **`Runner.run`** keeps **per-subagent SQLite sessions** (`checkpointer` on each `DeepAgentRunner`) and a **single trace story** that works with the manual HITL layer.

---

## SDK `interruptions` for HITL → manual `Hitl`

The framework does **not** rely on the SDK’s built-in interruption / approval pipeline for gated tools.

**Why:** With **`function_tool`** and the SDK’s **`result.interruptions`**, **subagent HITL did not bubble up** to the host the way we needed (parent vs child approval, sticky “allow for session”, etc.). **HITL is implemented manually** via `deepx.middleware.hitl` (`wrap_tools_for_hitl`, `Hitl.consult`, optional terminal policy in `deepx_cli`). A **persistence layer** writes **`/.deepx/sessions/<session_id>/approvals.json`** through the **tool runner’s** `backend` so orchestrator and specialists can use different storage backends.

---

## `mcp_servers` / `mcp_config` on `create_deep_agent`

Parameters to attach **MCP servers** directly on `create_deep_agent` were **removed**.

**Why:** That path used SDK mechanisms tied to **`.interruptions`** for MCP-backed tools. Tools appended at runtime **did not automatically** go through the same **HITL wrapping** as normal `FunctionTool`s. The replacement approach is to use **FastMCP** (or similar) to turn MCP capabilities into ordinary **`Tool` / `FunctionTool`** instances and pass them via **`tools=`**, so **`apply_tool_pipeline`** (eviction + `interrupt_on` + HITL) applies uniformly.

---

## Composite backend

A **composite backend** abstraction was **removed**.

**Why:** It was **unused** in the active codebase and added surface area without a clear, maintained use case.

---

## `test_demo/temporal`

The **Temporal**-based demo workflow was **removed**.

**Why:** It did not work well with **token streaming**, surfaced errors involving the **OpenAI Agents SDK**, **SQLiteSession** checkpointing, and **`NotImplementedError`**-class issues in the integration. The maintained demo is the **in-process** orchestrator + specialists (`test_demo/orchestrator.py`).

```
commit 62d392ecc0b493f8f79938c3ddc36b3f15cfc5c1
Author: dewmike <dewmikela@gmail.com>
Date:   Wed Apr 22 09:18:16 2026 +0530

    update

 delete mode 100644 test_demo/temporal/README.md
 delete mode 100644 test_demo/temporal/__init__.py
 delete mode 100644 test_demo/temporal/client.py
 delete mode 100644 test_demo/temporal/worker.py
 delete mode 100644 test_demo/temporal/workflows.py

commit ba01e4a0f8c6a02deeb9d52d85b80934f697f1c3
Author: dewmike <dewmikela@gmail.com>
Date:   Tue Apr 21 13:07:12 2026 +0530

    update

 delete mode 100644 test_demo/temporal/activities.py
```

---

## Chainlit UI

The **Chainlit**-based UI was **removed**.

**Why:** The goal was a **multi-tab** experience per session (e.g. switch between agents, see tool calls and reasoning in near real time). Chainlit’s **opinionated** app model (`chainlit.md`, session model) made it **difficult to wire five separate SQLite checkpointers** (one per demo agent) in a clean way. Complexity grew quickly, while **LangSmith** already provided **stronger end-to-end observability** for traces than the chainlit UI would have.

```
commit 122e6335bd1cf2f2560dba0f73a8472ca18809c8
Author: dewmike <dewmikela@gmail.com>
Date:   Sun Apr 19 19:37:13 2026 +0530

    update

D       test_demo/ui/__init__.py
D       test_demo/ui/app.py
D       test_demo/ui/auth/__init__.py
D       test_demo/ui/auth/password_auth.py
D       test_demo/ui/bootstrap/__init__.py
D       test_demo/ui/bootstrap/paths.py
D       test_demo/ui/chat/__init__.py
D       test_demo/ui/chat/profiles.py
D       test_demo/ui/chat/session_hooks.py
D       test_demo/ui/chat/settings_widgets.py
D       test_demo/ui/persistence/__init__.py
D       test_demo/ui/persistence/chainlit_schema.py
D       test_demo/ui/persistence/data_layer.py
D       test_demo/ui/runs/__init__.py
D       test_demo/ui/runs/run_modes.py
D       test_demo/ui/runs/session_prefs.py
D       test_demo/ui/runs/workflow_event_stream.py
```

---

## See also

- [`src/deepx/README.md`](src/deepx/README.md) — current architecture, backends, middleware.
- [`src/deepx_cli/README.md`](src/deepx_cli/README.md) — minimal terminal REPL helpers.
