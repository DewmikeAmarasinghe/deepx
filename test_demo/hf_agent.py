"""Hugging Face Hub specialist via FastMCP (Streamable HTTP).

Uses ``https://huggingface.co/mcp`` with ``Authorization: Bearer <HF_TOKEN>``. Token:
https://huggingface.co/settings/tokens — browser / OAuth: https://huggingface.co/mcp?login

Set ``HF_TOKEN`` in the repo-root ``.env`` (next to ``pyproject.toml``) or in the process environment.

Hub tools are exposed as :class:`~agents.tool.FunctionTool` instances (not ``Agent.mcp_servers``) so
they go through Deepx :func:`~deepx.middleware.tool_pipeline.apply_tool_pipeline` and
``interrupt_on`` / :class:`~deepx.middleware.hitl.Hitl` like other tools.

When ``HF_TOKEN`` is unset, :data:`hf_agent_runner` is ``None`` and the orchestrator omits this subagent.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import mcp.types as mt
from agents.tool import FunctionTool
from agents.tool_context import ToolContext

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

from fastmcp import Client  # noqa: E402
from fastmcp.client.client import CallToolResult  # noqa: E402
from fastmcp.client.transports import StreamableHttpTransport  # noqa: E402

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
_CLIENT_TIMEOUT_S = 300.0

_HF_SYSTEM = """\
You use Hugging Face Hub tools to search the Hub and fetch docs. \
When the orchestrator specifies paths, write substantive results under **/_outputs/** (markdown or JSON). \
Keep the chat reply short: **file paths** and a brief summary — do not paste long tool dumps in the reply.
"""


def _make_hf_transport() -> StreamableHttpTransport:
    return StreamableHttpTransport(
        _HF_MCP_URL,
        headers={"Authorization": f"Bearer {_hf_token}"},
    )


def _input_schema_as_dict(tool: mt.Tool) -> dict[str, Any]:
    raw = tool.inputSchema
    if raw is None:
        return {"type": "object", "additionalProperties": True}
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "model_dump"):
        return raw.model_dump(mode="json", exclude_none=True)
    return dict(raw)


def _format_hub_tool_result(result: CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, mt.TextContent):
            parts.append(block.text)
        elif isinstance(block, mt.ImageContent):
            parts.append(f"[image {block.mimeType}]")
        else:
            parts.append(str(block))
    if result.structured_content:
        parts.append(json.dumps(result.structured_content, indent=2, default=str))
    out = "\n".join(p for p in parts if p)
    return out if out else "(no output)"


def _make_hf_function_tool(
    tool_name: str,
    description: str,
    params_json_schema: dict[str, Any],
) -> FunctionTool:
    """One Hub MCP tool as a :class:`FunctionTool` (new FastMCP client session per call)."""

    async def on_invoke_tool(ctx: ToolContext, args_json: str) -> str:
        try:
            arguments = json.loads(args_json) if (args_json or "").strip() else {}
        except json.JSONDecodeError as e:
            return f"Invalid JSON arguments: {e}"
        if not isinstance(arguments, dict):
            return "Tool arguments must be a JSON object."

        transport = _make_hf_transport()
        async with Client(transport, timeout=_CLIENT_TIMEOUT_S) as client:
            result = await client.call_tool(
                tool_name,
                arguments,
                raise_on_error=False,
            )
        if result.is_error:
            return (
                _format_hub_tool_result(result)
                or f"Tool {tool_name!r} returned an error."
            )
        return _format_hub_tool_result(result)

    desc = (description or "").strip() or f"Hugging Face Hub MCP tool `{tool_name}`."
    return FunctionTool(
        name=tool_name,
        description=desc,
        params_json_schema=params_json_schema,
        on_invoke_tool=on_invoke_tool,
        strict_json_schema=False,
    )


async def _fetch_hf_tools_async() -> list[FunctionTool]:
    transport = _make_hf_transport()
    async with Client(transport, timeout=_CLIENT_TIMEOUT_S) as client:
        discovered = await client.list_tools()
    return [
        _make_hf_function_tool(
            t.name,
            t.description or "",
            _input_schema_as_dict(t),
        )
        for t in discovered
    ]


def _load_hf_tools_sync() -> list[FunctionTool]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_fetch_hf_tools_async())

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(asyncio.run, _fetch_hf_tools_async())
        return cast(list[FunctionTool], fut.result())


hf_agent_runner: DeepAgentRunner | None
if not _hf_token:
    hf_agent_runner = None
else:
    _hf_tools = _load_hf_tools_sync()
    hf_agent_runner = create_deep_agent(
        name="hf_agent",
        memory=[".deepx/AGENTS.md"],
        description=(
            "Hugging Face Hub specialist: model/dataset/space search, docs, and Hub tools exposed "
            "as normal function tools. Use for anything on hf.co / the Hub API surface. "
            "Ask it to write digests under **/_outputs/** and return paths plus a short summary."
        ),
        tools=_hf_tools,
        interrupt_on=["hub_repo_search"],
        system_prompt=_HF_SYSTEM,
        backend=_DEMO_BACKEND,
        checkpointer=_HF_DB,
        debug=True,
        subagents=None,
    )
