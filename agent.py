from __future__ import annotations

from dotenv import load_dotenv

import os
import httpx
from agents import RunContextWrapper, function_tool
from deepx import create_deep_agent

load_dotenv()

TAVILY_KEY = os.environ["TAVILY_API_KEY"]


@function_tool
async def web_search(ctx: RunContextWrapper, query: str) -> str:
    """Search the web for current information. Returns titles, URLs, and snippets."""
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


tools = [web_search, web_extract]

researcher = {
    "name": "researcher",
    "description": (
        "Delegate deep research to this agent. It searches the web, extracts full "
        "page content, and saves findings to research/ files. Give it one focused "
        "topic at a time."
    ),
    "system_prompt": (
        "You are a thorough research specialist. "
        "Always save findings to research/ files before returning. "
        "Extract full page content for important sources — not just snippets. "
        "Return a short summary of what you found and where you saved it."
    ),
    "tools": tools,
}

writer = {
    "name": "writer",
    "description": (
        "Delegate report writing to this agent. It reads research/ files and "
        "produces a polished output/ document. Use after research is complete."
    ),
    "system_prompt": (
        "You are a professional writer. "
        "Always read the relevant research/ files before writing. "
        "Write output to output/ files. "
        "Be direct and opinionated — no hedging. "
        "Return the path to the file you wrote."
    ),
    "tools": tools,
}

agent = create_deep_agent(
    model="gpt-4o-mini",
    tools=tools,
    subagents=[researcher, writer],
    system_prompt=(
        "You are an orchestrator. Delegate research to the researcher subagent "
        "and report writing to the writer subagent. "
        "Keep your own context clean — pass file paths, not raw content. "
        "Use the task tool with subagent_type researcher or writer."
    ),
    interrupt_on=["web_search"],
    debug=True,
    db_path="agent.db",
)

TASK = (
    "Compare Ollama, vLLM, and llama.cpp for self-hosting a 7B model on a single "
    "consumer GPU (RTX 3090). For each: current version, GPU support, one recent "
    "benchmark, one known limitation. End with a clear winner and why. "
    "Save notes to research/ as you go, final report to output/comparison.md."
)

result = agent.run_sync(TASK, session_id="inference_001")

print("\n" + "=" * 60)
print(result.output)
print("=" * 60)
print(f"\nSession: {result.session_id}")
print("Plan:")
for i, todo in enumerate(result.plan.todos, 1):
    print(f"  [{i}] ({todo.status.value}) {todo.title}")