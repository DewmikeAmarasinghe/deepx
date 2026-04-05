from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import httpx
from pathlib import Path
from agents import RunContextWrapper, function_tool
from deepx import create_deep_agent


TAVILY_KEY = os.environ["TAVILY_API_KEY"]


@function_tool
async def web_search(ctx: RunContextWrapper, query: str) -> str:
    """Search the web for current information. Returns titles, URLs, and text snippets."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_KEY, "query": query, "max_results": 5},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "No results found."
    return "\n\n---\n\n".join(
        f"[{i+1}] {res['title']}\nURL: {res['url']}\n{res['content']}"
        for i, res in enumerate(results)
    )


@function_tool
async def web_extract(ctx: RunContextWrapper, url: str) -> str:
    """Extract the full text content from a URL. Use after web_search to read a full page."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/extract",
            json={"api_key": TAVILY_KEY, "urls": [url]},
            timeout=25,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "Could not extract content from that URL."
    return results[0].get("raw_content", "No content extracted.")


task = Path("task.md").read_text()

agent = create_deep_agent(
    model="gpt-4o-mini",
    tools=[web_search, web_extract],
    agents_path="agents/",
    workspace_path=".deepx",
    require_approval=["web_search"],
    log_tools=True,
)

result = agent.run_sync(task, session_id="research_001")

print("\n" + "=" * 60)
print(result.output)
print("=" * 60)
print("\nPlan:")
for i, todo in enumerate(result.plan.todos, 1):
    print(f"  [{i}] ({todo.status.value}) {todo.title}")