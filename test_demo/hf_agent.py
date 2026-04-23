"""Hugging Face Hub specialist via **hosted** MCP (Streamable HTTP).

Uses ``https://huggingface.co/mcp`` with ``Authorization: Bearer <HF_TOKEN>``. Token:
https://huggingface.co/settings/tokens — browser / OAuth: https://huggingface.co/mcp?login

Set ``HF_TOKEN`` in the repo-root ``.env`` (next to ``pyproject.toml``) or in the process environment.

MCP lifecycle is handled in :mod:`deepx.factory` via :class:`agents.mcp.MCPServerManager`.

**HITL:** Deepx ``interrupt_on`` only wraps tools that go through :func:`deepx.middleware.tool_pipeline.apply_tool_pipeline`
(the static ``agent.tools`` list). Hub tools are MCP-backed ``FunctionTool`` instances added at runtime; gate them with
``require_approval`` on :class:`agents.mcp.MCPServerStreamableHttp` (per-tool map or ``"always"``), not ``interrupt_on``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

from agents.mcp import MCPServerStreamableHttp  # noqa: E402

from deepx.backends.filesystem import FilesystemBackend  # noqa: E402
from deepx.factory import DeepAgentRunner, create_deep_agent  # noqa: E402

_DEMO_BACKEND = FilesystemBackend(_REPO_ROOT)
_AGENT_DBS = _REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
_AGENT_DBS.mkdir(parents=True, exist_ok=True)
_HF_DB = str(_AGENT_DBS / "hf_agent.db")

_hf_token = (os.environ.get("HF_TOKEN") or "").strip()

_HF_MCP_URL = "https://huggingface.co/mcp"

_HF_SYSTEM = """\
You use Hugging Face MCP tools to search the Hub and fetch docs. \
Save substantive results under **/_outputs/** (markdown or JSON); keep the chat reply short \
with **file paths** and a brief summary — do not paste long tool dumps in the reply.
"""

if not _hf_token:
    hf_agent_runner: DeepAgentRunner = create_deep_agent(
        name="hf_agent",
        description=(
            "Hugging Face Hub via hosted MCP — **not configured**: set HF_TOKEN in .env or the environment."
        ),
        tools=[],
        system_prompt=(
            "You are **hf_agent** without HF_TOKEN. Say briefly that Hugging Face MCP is not configured; "
            "do not pretend to call Hub tools."
        ),
        backend=_DEMO_BACKEND,
        checkpointer=_HF_DB,
        debug=True,
        subagents=None,
    )
else:
    hf_agent_runner = create_deep_agent(
        name="hf_agent",
        description=(
            "This agent is connected to Hugging Face Hub via hosted MCP (https://huggingface.co/mcp). "
            "Can search models, datasets, spaces, and docs; write digests under **/_outputs/** and return "
            "paths plus a short summary."
        ),
        tools=[],
        system_prompt=_HF_SYSTEM,
        backend=_DEMO_BACKEND,
        checkpointer=_HF_DB,
        debug=True,
        subagents=None,
        mcp_servers=[
            MCPServerStreamableHttp(
                params={
                    "url": _HF_MCP_URL,
                    "headers": {"Authorization": f"Bearer {_hf_token}"},
                    "timeout": 120.0,
                    "sse_read_timeout": 300.0,
                },
                name="huggingface",
                cache_tools_list=True,
                max_retry_attempts=2,
            )
        ],
    )

