"""Web research subagent: Tavily-backed tools (demo)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402
from agents import RunContextWrapper, function_tool  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _REPO_ROOT
SKILLS_DIR = DEMO_DIR / "skills"


def _strip_images(obj: Any) -> Any:
    if isinstance(obj, dict):
        obj = dict(obj)
        obj.pop("images", None)
        for k, v in list(obj.items()):
            obj[k] = _strip_images(v)
        return obj
    if isinstance(obj, list):
        return [_strip_images(x) for x in obj]
    return obj


@function_tool
async def web_search(ctx: RunContextWrapper, queries: list[str]) -> str:
    """
    Search the public web using the **Tavily** API (hosted index).

    Many parallel queries return one large JSON string; oversized returns may be written under
    ``/_outputs/large_tool_results/...`` (see framework prompt) with a preview in the tool message.
    """
    _ = ctx
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot perform web search."
    qs = [q.strip() for q in queries if q and str(q).strip()]
    if not qs:
        return "No queries provided."
    async with httpx.AsyncClient() as client:

        async def one(q: str) -> dict:
            r = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_KEY,
                    "query": q,
                    "max_results": 5,
                },
                timeout=25,
            )
            r.raise_for_status()
            data = r.json()
            row = {
                "query": q,
                "answer": data.get("answer"),
                "results": data.get("results", []),
            }
            return _strip_images(row)

        rows = await asyncio.gather(*[one(q) for q in qs])
    return json.dumps(rows, indent=2)


@function_tool
async def web_extract(
    ctx: RunContextWrapper,
    urls: list[str],
    query: str | None = None,
    chunks_per_source: int | None = None,
) -> str:
    """Extract readable text from specific URLs using the **Tavily** extract API."""
    _ = ctx
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot extract content."
    if not urls:
        return "[]"
    body: dict = {"api_key": TAVILY_KEY, "urls": urls}
    if query:
        body["query"] = query
    if chunks_per_source is not None:
        body["chunks_per_source"] = max(1, min(5, int(chunks_per_source)))
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/extract",
            json=body,
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
    out = _strip_images(
        {
            "results": data.get("results", []),
            "failed_results": data.get("failed_results", []),
        }
    )
    return json.dumps(out, indent=2)


@function_tool
async def web_map(
    ctx: RunContextWrapper,
    url: str,
    instructions: str | None = None,
    max_depth: int = 1,
    max_breadth: int = 12,
    limit: int = 40,
) -> str:
    """Map outbound links from a root URL using the **Tavily** map API (single-site crawl)."""
    _ = ctx
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot map site."
    body: dict = {
        "api_key": TAVILY_KEY,
        "url": url,
        "max_depth": max(1, min(5, int(max_depth))),
        "max_breadth": max(1, min(500, int(max_breadth))),
        "limit": max(1, int(limit)),
    }
    if instructions:
        body["instructions"] = instructions
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/map",
            json=body,
            timeout=160,
        )
        r.raise_for_status()
        return json.dumps(_strip_images(r.json()), indent=2)


WEB_TOOLS = [web_search, web_extract, web_map]


def web_agent_spec(*, checkpointer: str, interrupt_on: list[str]) -> dict:
    return {
        "name": "web_agent",
        "description": (
            "Web research specialist: Tavily search/extract/map; saves structured notes and can "
            "author final markdown reports under the project tree. Returns artifact paths."
        ),
        "system_prompt": (
            "You are the **web_agent** internal service. Tools: `web_search`, `web_extract`, `web_map`.\n"
            "For any multi-step brief, call **`write_todos` first** after skimming the relevant "
            "**deep-research** / **arxiv-search** skill files (`read_file` on the paths listed in "
            "your prompt), then **`update_todos`** after each major step.\n"
            "Persist research in a small number of well-named files (one topical area per file when "
            "possible). Use clear headings, inline citations, and a Sources section on written "
            "deliverables.\n"
            "When the brief includes a **written deliverable**, produce the full final markdown "
            "yourself with `write_file`—executive summary, findings, and sources—so the parent agent "
            "does not need a separate writer.\n"
            "Return every artifact path you created plus a tight summary. Do not dump large raw "
            "JSON into chat; keep bulky tool output in files and point to them briefly."
        ),
        "tools": WEB_TOOLS,
        "skills": [
            str(SKILLS_DIR / "deep-research"),
            str(SKILLS_DIR / "arxiv-search"),
        ],
        "interrupt_on": interrupt_on,
        "checkpointer": checkpointer,
    }


if __name__ == "__main__":
    from deepx import create_deep_agent
    from deepx.backends.local_shell import LocalShellBackend

    DEMO_DIR.joinpath("dbs", "agent_dbs").mkdir(parents=True, exist_ok=True)
    cp = str(DEMO_DIR / "dbs" / "agent_dbs" / "web_agent_standalone.db")
    runner = create_deep_agent(
        name="web_agent",
        description="Standalone web research + report agent.",
        tools=list(WEB_TOOLS),
        skills=[
            str(SKILLS_DIR / "deep-research"),
            str(SKILLS_DIR / "arxiv-search"),
        ],
        system_prompt=web_agent_spec(checkpointer=cp, interrupt_on=[])["system_prompt"],
        checkpointer=cp,
        backend=LocalShellBackend(REPO_ROOT),
        interrupt_on=["web_search"],
        debug=True,
    )
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    _script = Path(__file__).resolve()
    _resume = f"{sys.executable} {_script}"
    task = """
You are helping a product committee decide whether to standardise on **Postgres** or **DuckDB**
for an analytics lakehouse MVP (10–50 TB, mostly Parquet on object storage, daily batch + a few
interactive dashboards). I need:

1) A tight comparison on strengths/weaknesses for our shape of work (join-heavy SQL, windowing,
   semi-structured JSON, operational cost, ecosystem, hosted vs self-managed realities).

2) A **risk register** (top 8 risks) with mitigations—not generic bullets; tie each risk to how we
said we work.

3) A **decision memo** as a project markdown file: recommendation, **three** acceptance tests we should run
in week one on our own data, and a one-page experiment plan (datasets, queries, success metrics).

4) Paths to every file you created plus a 5-line summary in your final message.
"""
    result = runner.run_sync(task, session_id=sid)
    print(result.output)
    print(
        f"\nSession: {result.session_id}\n"
        "To continue with the same session, use your runner’s resume flow; this script was:\n"
        f"  {_resume}\n"
    )
