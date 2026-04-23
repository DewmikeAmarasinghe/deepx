"""Hugging Face Hub specialist via **hosted** MCP (Streamable HTTP).

Uses ``https://huggingface.co/mcp`` with ``Authorization: Bearer <HF_TOKEN>``. Token:
https://huggingface.co/settings/tokens — browser / OAuth: https://huggingface.co/mcp?login

Set ``HF_TOKEN`` in the repo-root ``.env`` (next to ``pyproject.toml``) or in the process environment.

When ``HF_TOKEN`` is unset, :data:`hf_agent_runner` is ``None`` and the orchestrator omits this subagent.

MCP lifecycle: :class:`agents.mcp.MCPServerManager` in :mod:`deepx.factory`.

**Gating:** Deepx ``interrupt_on`` only applies to tools wrapped in :func:`deepx.middleware.tool_pipeline.apply_tool_pipeline`.
MCP Hub tools are not in that list, so ``interrupt_on`` cannot gate them. Use SDK ``require_approval`` on the MCP server
only if your runner resumes :class:`~agents.items.ToolApprovalItem` (the demo CLI does not yet).
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
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token

_HF_MCP_URL = "https://huggingface.co/mcp"

_HF_SYSTEM = """\
You use Hugging Face MCP tools to search the Hub and fetch docs. \
When the orchestrator specifies paths, write substantive results under **/_outputs/** (markdown or JSON). \
Keep the chat reply short: **file paths** and a brief summary — do not paste long tool dumps in the reply.
"""

hf_agent_runner: DeepAgentRunner | None
if not _hf_token:
    hf_agent_runner = None
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
                require_approval="never",
                cache_tools_list=True,
                max_retry_attempts=2,
            )
        ],
    )
