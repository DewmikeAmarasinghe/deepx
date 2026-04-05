from __future__ import annotations
import os
import httpx
from dotenv import load_dotenv
from agents import RunContextWrapper, function_tool
from deepx import create_deep_agent, HumanInTheLoopHooks
from deepx.backends.filesystem import FilesystemBackend

load_dotenv()

TAVILY_KEY = os.environ["TAVILY_API_KEY"]


@function_tool
async def web_search(ctx: RunContextWrapper, query: str) -> str:
    """Search the web for current information. Returns titles, URLs, and text snippets for up to 5 results."""
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
    lines = [
        f"[{i+1}] {res['title']}\nURL: {res['url']}\n{res['content']}"
        for i, res in enumerate(results)
    ]
    return "\n\n---\n\n".join(lines)


@function_tool
async def web_extract(ctx: RunContextWrapper, url: str) -> str:
    """Extract full text content from a URL. Use this after web_search to read a full page.
    Returns the raw text of the page. Prefer Wikipedia or official pages for factual content."""
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


agent = create_deep_agent(
    model="gpt-4o-mini",
    tools=[web_search, web_extract],
    workspace_path=".deepx",
    system_prompt=(
        "You are a meticulous research agent. "
        "Always plan with write_todos before acting. "
        "Expand and rewrite your plan with write_todos whenever you discover new sub-tasks mid-execution. "
        "Save all research to files. Never hold large content in the conversation."
    ),
)

TASK = """
You are on a web research treasure hunt across four stages. Execute them in order.

STAGE 1 — Draw the map:
Search for the 5 most-visited tourist attractions in Kyoto, Japan.
Write the list (name + one-sentence description each) to research/kyoto_attractions.md.
Mark stage 1 done.

STAGE 2 — Dig for treasure:
For each of the 5 attractions, add a dedicated sub-task to your plan (rewrite the plan
with write_todos to include them). Then for each attraction, search for its Wikipedia page
URL, extract that page, and pull out one surprising or little-known historical fact not
commonly mentioned in travel guides. Append each fact to research/kyoto_facts.md as you
go, one section per attraction. Mark each sub-task done as you complete it.

STAGE 3 — Follow the hidden trail:
Search for "underrated Kyoto attractions" and "hidden gems Kyoto Japan". Identify which
of the 5 attractions (if any) appear in these hidden-gem articles and note why travellers
consider them underrated. Save findings to research/hidden_gems.md. Mark stage 3 done.

STAGE 4 — Claim the treasure:
Read all three research files. Write a compelling 350-word travel guide to
output/kyoto_guide.md that weaves together: the top attractions, the surprising historical
facts, and the hidden-gem angle, in a narrative style aimed at curious travellers.
Then call update_memory with the single top recommendation from your research.
Mark stage 4 done.
"""

result = agent.run_sync(TASK, session_id="kyoto_hunt_001")

print("\n" + "=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(result.output)

print("\n" + "=" * 60)
print("PLAN STATUS")
print("=" * 60)
for i, todo in enumerate(result.plan.todos, 1):
    print(f"[{i}] ({todo.status.value:12}) {todo.title}")

print("\n" + "=" * 60)
print("WORKSPACE FILES")
print("=" * 60)
backend = FilesystemBackend(".deepx")
for f in backend.list_files("kyoto_hunt_001"):
    print(f)