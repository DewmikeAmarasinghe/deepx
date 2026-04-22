"""Hugging Face Hub MCP subagent (demo-only).

Requires ``HF_TOKEN`` or ``HUGGINGFACE_HUB_TOKEN``, Node/npx on PATH for
``npx @llmindset/hf-mcp-server`` (stdio MCP). If no token is set,
:attr:`hf_agent_runner` is ``None``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agents.mcp import MCPServerStdio, MCPServerStdioParams  # noqa: E402

from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.factory import DeepAgentRunner, create_deep_agent  # noqa: E402

_DEMO_BACKEND = LocalShellBackend(_REPO_ROOT)
_AGENT_DBS = _REPO_ROOT / "test_demo" / "dbs" / "agent_dbs"
_AGENT_DBS.mkdir(parents=True, exist_ok=True)
_HF_DB = str(_AGENT_DBS / "hf_agent.db")


def build_hf_agent_runner(*, checkpointer: str | None = None) -> DeepAgentRunner | None:
    token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    ).strip()
    if not token:
        return None
    cp = _HF_DB if checkpointer is None else checkpointer
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    params: MCPServerStdioParams = {
        "command": "npx",
        "args": ["-y", "@llmindset/hf-mcp-server"],
        "env": env,
    }
    return create_deep_agent(
        name="hf_agent",
        description=(
            "Hugging Face Hub via MCP: search models, datasets, spaces, and documentation."
        ),
        tools=[],
        system_prompt=(
            "You use Hugging Face MCP tools to search the Hub and fetch docs. Keep answers concise; "
            "write long tool dumps under **/_outputs/** and return paths."
        ),
        backend=_DEMO_BACKEND,
        checkpointer=cp,
        debug=True,
        include_general_purpose=False,
        subagents=None,
        mcp_servers=[MCPServerStdio(params, name="huggingface")],
    )


hf_agent_runner = build_hf_agent_runner()
