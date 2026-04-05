import asyncio
import os
from typing import Optional

import httpx
from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv

from deepx import AgentContext, create_agent

load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")


@function_tool
async def tavily_search(
    ctx: RunContextWrapper[AgentContext],
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
    include_domains: str = "",
    exclude_domains: str = "",
) -> str:
    payload: dict = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": False,
    }
    if include_domains.strip():
        payload["include_domains"] = [d.strip() for d in include_domains.split(",") if d.strip()]
    if exclude_domains.strip():
        payload["exclude_domains"] = [d.strip() for d in exclude_domains.split(",") if d.strip()]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post("https://api.tavily.com/search", json=payload)
        resp.raise_for_status()
        data = resp.json()

    lines = []
    if data.get("answer"):
        lines.append(f"## AI Answer\n{data['answer']}\n")

    lines.append("## Search Results")
    for i, r in enumerate(data.get("results", []), 1):
        lines.append(f"\n### {i}. {r.get('title', 'Untitled')}")
        lines.append(f"URL: {r['url']}")
        if r.get("published_date"):
            lines.append(f"Date: {r['published_date']}")
        if r.get("content"):
            lines.append(f"\n{r['content']}")

    ctx.context.visited_urls.update(r["url"] for r in data.get("results", []))
    return "\n".join(lines)


@function_tool
async def tavily_extract(
    ctx: RunContextWrapper[AgentContext],
    urls: str,
) -> str:
    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    payload = {
        "api_key": TAVILY_API_KEY,
        "urls": url_list,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post("https://api.tavily.com/extract", json=payload)
        resp.raise_for_status()
        data = resp.json()

    lines = []
    for item in data.get("results", []):
        lines.append(f"\n## {item.get('url', '')}")
        if item.get("raw_content"):
            lines.append(item["raw_content"])
        elif item.get("content"):
            lines.append(item["content"])

    for item in data.get("failed_results", []):
        lines.append(f"\n## FAILED: {item.get('url', '')}")
        lines.append(f"Error: {item.get('error', 'unknown')}")

    ctx.context.visited_urls.update(url_list)
    return "\n".join(lines) if lines else "(no content extracted)"


@function_tool
async def scrape_url(
    ctx: RunContextWrapper[AgentContext],
    url: str,
    mode: str = "reader",
) -> str:
    if mode == "reader":
        target = f"https://r.jina.ai/{url}"
        headers = {"Accept": "text/plain"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    elif mode == "search_grounded":
        target = f"https://s.jina.ai/{url}"
        headers = {"Accept": "text/plain"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    else:
        target = url
        headers = {}

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(target, headers=headers)
        resp.raise_for_status()
        content = resp.text

    ctx.context.visited_urls.add(url)
    return content


@function_tool
async def batch_search(
    ctx: RunContextWrapper[AgentContext],
    queries: str,
    max_results_each: int = 5,
) -> str:
    query_list = [q.strip() for q in queries.split("|") if q.strip()]
    results = []

    async def _one(q: str) -> str:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": q,
            "max_results": max_results_each,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            return q, resp.json()

    tasks = [_one(q) for q in query_list]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for item in responses:
        if isinstance(item, Exception):
            results.append(f"## ERROR\n{item}\n")
            continue
        q, data = item
        lines = [f"## Query: {q}"]
        if data.get("answer"):
            lines.append(f"Answer: {data['answer']}")
        for r in data.get("results", []):
            lines.append(f"- [{r.get('title', '')}]({r['url']})")
            if r.get("content"):
                lines.append(f"  {r['content'][:300]}")
        results.append("\n".join(lines))
        ctx.context.visited_urls.update(r["url"] for r in data.get("results", []))

    return "\n\n---\n\n".join(results)


RESEARCH_PROMPT = """You are an expert research analyst. You produce structured, citation-rich reports.

## research workflow
1. write_todos with a clear research plan before starting any task
2. use batch_search for broad topic discovery (pipe-separate multiple queries)
3. use tavily_search for deep dives on specific subtopics
4. use tavily_extract or scrape_url to get full content from the most relevant URLs
5. write findings incrementally to /research/*.md files as you gather them
6. synthesize into a final report at /report.md using write_file
7. call update_memory for any persistent facts about sources or methodology

## report structure
Every final report must include:
- Executive Summary
- Key Findings (numbered, with inline citations [Source: URL])
- Detailed Analysis sections
- Data / Statistics (if available)
- Sources & References

## file naming convention
/research/01_overview.md      initial landscape scan
/research/02_deep_dive_*.md   per-subtopic deep research
/research/03_data.md          statistics and numbers gathered
/report.md                    final synthesized report
"""


def create_research_runner(
    model: str = "gpt-4o",
    db_path: str = "research.db",
    memory_path: Optional[str] = "memory/research_notes.md",
    skills_path: Optional[str] = None,
    max_turns: int = 80,
):
    return create_agent(
        model=model,
        tools=[
            tavily_search,
            tavily_extract,
            scrape_url,
            batch_search,
        ],
        skills_path=skills_path,
        memory_path=memory_path,
        system_prompt=RESEARCH_PROMPT,
        db_path=db_path,
        max_turns=max_turns,
    )


async def main():
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY environment variable not set")

    runner = create_research_runner(
        model="gpt-4o",
        db_path="research.db",
        memory_path="memory/research_notes.md",
    )

    task = (
        "Research the current state of AI coding assistants in 2025. "
        "Cover: major players (Cursor, GitHub Copilot, Windsurf, Claude Code, etc.), "
        "market share estimates, pricing models, key differentiators, "
        "recent funding/acquisitions, and where the market is heading. "
        "Produce a comprehensive report."
    )

    print("Starting research...\n")
    result = await runner.run(task=task, session_id="research_001")

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(result.output)
    print(f"\nTokens used: {result.token_usage}")
    print(f"Files written: {sorted(result.vfs.keys())}")
    print(f"Tool calls: {len(result.step_log)}")

    if "/report.md" in result.vfs:
        with open("final_report.md", "w", encoding="utf-8") as f:
            f.write(result.vfs["/report.md"])
        print("\nReport saved to final_report.md")


if __name__ == "__main__":
    asyncio.run(main())