"""Hugging Face Hub MCP subagent (demo-only).

Requires **Node.js** with ``npx`` on ``PATH``. The package is fetched on demand via
``npx -y @llmindset/hf-mcp-server``.

**Authentication:** set ``HF_TOKEN`` in the **repository root** ``.env`` (next to ``pyproject.toml``)
or in the process environment. Values are passed to the MCP subprocess via ``env``.

The OpenAI Agents SDK requires ``await server.connect()`` before MCP tools run; :mod:`deepx.factory`
does that around each :class:`~agents.Runner` invocation.

If tools still fail, some server builds write non–JSON-RPC lines to stdout, which breaks stdio MCP.
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

from agents.mcp import MCPServerStdio, MCPServerStdioParams  # noqa: E402

from deepx.backends.filesystem import FilesystemBackend  # noqa: E402
from deepx.factory import DeepAgentRunner, create_deep_agent  # noqa: E402

_DEMO_BACKEND = FilesystemBackend(_REPO_ROOT)
_AGENT_DBS = _REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
_AGENT_DBS.mkdir(parents=True, exist_ok=True)
_HF_DB = str(_AGENT_DBS / "hf_agent.db")

_hf_token = (os.environ.get("HF_TOKEN") or "").strip()
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token

_hf_disabled_prompt = """\
You are **hf_agent**, but Hugging Face MCP is **not** configured: set **HF_TOKEN** in the repo-root
``.env`` or in the process environment, and ensure **Node/npx** is available. Reply briefly;
do not pretend to call Hub tools.
"""

if not _hf_token:
    hf_agent_runner: DeepAgentRunner = create_deep_agent(
        name="hf_agent",
        description=(
            "Hugging Face Hub via MCP (requires HF_TOKEN and npx). Not configured in this environment."
        ),
        tools=[],
        system_prompt=_hf_disabled_prompt,
        backend=_DEMO_BACKEND,
        checkpointer=_HF_DB,
        debug=True,
        subagents=None,
    )
else:
    _hf_env = os.environ.copy()
    _hf_env["HF_TOKEN"] = _hf_token
    _hf_params: MCPServerStdioParams = {
        "command": "npx",
        "args": ["-y", "@llmindset/hf-mcp-server"],
        "env": _hf_env,
    }
    hf_agent_runner = create_deep_agent(
        name="hf_agent",
        description=(
            "Hugging Face Hub via MCP: search models, datasets, spaces, and documentation."
        ),
        tools=[],
        system_prompt=(
            "You use Hugging Face MCP tools to search the Hub and fetch docs. "
            "Save substantive results under **/_outputs/** (markdown or JSON); keep the chat reply short "
            "with **file paths** and a brief summary — do not paste long tool dumps in the reply."
        ),
        backend=_DEMO_BACKEND,
        checkpointer=_HF_DB,
        debug=True,
        subagents=None,
        mcp_servers=[
            MCPServerStdio(
                _hf_params,
                name="huggingface",
                require_approval="never",
            )
        ],
    )
