"""Hugging Face Hub specialist via FastMCP (Streamable HTTP).

Uses ``https://huggingface.co/mcp`` with ``Authorization: Bearer <HF_TOKEN>``.
Get a token at https://huggingface.co/settings/tokens.

Set ``HF_TOKEN`` in the repo-root ``.env`` or in the process environment.
When ``HF_TOKEN`` is unset, :data:`hf_agent_runner` is ``None`` and the orchestrator
omits this subagent.

Run standalone::

    python -m test_demo.hf_agent --chat
    python -m test_demo.hf_agent --chat_sync
    python -m test_demo.hf_agent --chat --session <id>
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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

load_dotenv()

from fastmcp import Client  # noqa: E402
from fastmcp.client.client import CallToolResult  # noqa: E402
from fastmcp.client.transports import StreamableHttpTransport  # noqa: E402

from deepx.backends.filesystem import FilesystemBackend  # noqa: E402
from deepx.factory import DeepAgentRunner, create_deep_agent  # noqa: E402

_HF_TOKEN = (os.environ.get("HF_TOKEN") or "").strip()
if _HF_TOKEN:
    os.environ["HF_TOKEN"] = _HF_TOKEN

_HF_MCP_URL = "https://huggingface.co/mcp"
_CLIENT_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------


def _input_schema_as_dict(tool: mt.Tool) -> dict[str, Any]:
    raw = tool.inputSchema
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "model_dump"):
        return raw.model_dump(mode="json", exclude_none=True)  # type: ignore[union-attr]
    if raw is not None:
        return dict(raw)  # type: ignore[arg-type]
    return {"type": "object", "additionalProperties": True}


# ---------------------------------------------------------------------------
# Hub MCP transport and tool factory
# ---------------------------------------------------------------------------


def _make_transport() -> StreamableHttpTransport:
    return StreamableHttpTransport(
        _HF_MCP_URL,
        headers={"Authorization": f"Bearer {_HF_TOKEN}"},
    )


def _format_result(result: CallToolResult) -> str:
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
    return "\n".join(p for p in parts if p) or "(no output)"


def _make_hub_tool(
    tool_name: str,
    description: str,
    params_schema: dict[str, Any],
) -> FunctionTool:
    """Wrap one Hub MCP tool as a FunctionTool (new FastMCP session per call)."""

    async def on_invoke_tool(ctx: ToolContext, args_json: str) -> str:
        try:
            arguments = json.loads(args_json) if (args_json or "").strip() else {}
        except json.JSONDecodeError as exc:
            return f"Invalid JSON arguments: {exc}"
        if not isinstance(arguments, dict):
            return "Tool arguments must be a JSON object."

        async with Client(_make_transport(), timeout=_CLIENT_TIMEOUT) as client:
            result = await client.call_tool(tool_name, arguments, raise_on_error=False)

        if result.is_error:
            return _format_result(result) or f"Tool {tool_name!r} returned an error."
        return _format_result(result)

    return FunctionTool(
        name=tool_name,
        description=description.strip() or f"Hugging Face Hub MCP tool `{tool_name}`.",
        params_json_schema=params_schema,
        on_invoke_tool=on_invoke_tool,
        strict_json_schema=False,
    )


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


async def _fetch_hub_tools() -> list[FunctionTool]:
    async with Client(_make_transport(), timeout=_CLIENT_TIMEOUT) as client:
        discovered = await client.list_tools()
    return [
        _make_hub_tool(t.name, t.description or "", _input_schema_as_dict(t))
        for t in discovered
    ]


def _load_hub_tools_sync() -> list[FunctionTool]:
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return cast(list[FunctionTool], pool.submit(asyncio.run, _fetch_hub_tools()).result())
    except RuntimeError:
        return asyncio.run(_fetch_hub_tools())


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

hf_sys = """\
You are the Hugging Face Hub specialist. Use your Hub MCP tools to search models,
datasets, and spaces, and to fetch documentation and metadata from hf.co.

## Output

**Primary:** answer in **clear markdown** in your reply (`#` / `##` headings, bullets, `---`
between sections, fenced blocks for JSON or short excerpts). That is what the parent agent
and the user read first.

**Optional files:** for very large or repeatable digests, you may also write to
`/_outputs/<name>.md` or `.json` and mention the path in a short line — do not rely on files
instead of a readable answer unless the caller asked for artifacts on disk.

## Tool use

- Use Hub tools directly — do not attempt curl, requests, or CLI commands.
- If a tool call returns an error, retry once with adjusted parameters before reporting failure.
- For searches, prefer specific filters (task, library, dataset type) over broad queries.

## Format guidelines

- Markdown: title, short takeaway up top, then sections (e.g. Top Models, Key Datasets,
  Notable Spaces) with tables or bullets.
- JSON in chat or in a file: pretty-print with two-space indentation; prefer a top-level `"meta"`
  with query and timestamp when saving structured data.
"""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


hf_agent_runner: DeepAgentRunner | None

if not _HF_TOKEN:
    hf_agent_runner = None
else:
    hf_agent_runner = create_deep_agent(
        name="hf_agent",
        memory=[".deepx/AGENTS.md"],
        description=(
            "Hugging Face Hub specialist: searches models, datasets, and spaces; fetches "
            "documentation and Hub API data. Use for anything on hf.co or the Hub surface. "
            "Returns findings as markdown in its reply"
        ),
        tools=_load_hub_tools_sync(),
        interrupt_on=["hub_repo_search"],
        system_prompt=hf_sys,
        backend=FilesystemBackend(_REPO_ROOT),
        checkpointer=str(_REPO_ROOT / "test_demo" / "dbs" / "agent_dbs" / "hf_agent.db"),
        debug=True,
        subagents=None,
    )