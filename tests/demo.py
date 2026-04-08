"""Demo: Multi-agent research and architecture decision record (ADR) generation.

Demonstrates:
- write_todos for structured planning
- Parallel task delegation via the task tool
- Researcher subagent doing deep web research (only agent with web tools)
- Writer subagent producing a polished ADR from research files
- HITL session approval (approve once, not again)
- Debug tool logging

Usage:
    OPENAI_API_KEY=... TAVILY_API_KEY=... python tests/demo.py
"""
from __future__ import annotations

import os
import uuid

import httpx
from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv

from deepx import create_deep_agent

load_dotenv()

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")


@function_tool
async def web_search(ctx: RunContextWrapper, query: str) -> str:
    """Search the web for current information. Returns titles, URLs, and snippets."""
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot perform web search."
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
        f"[{i + 1}] {res['title']}\nURL: {res['url']}\n{res['content']}"
        for i, res in enumerate(results)
    )


@function_tool
async def web_extract(ctx: RunContextWrapper, url: str) -> str:
    """Extract the full text content from a URL."""
    if not TAVILY_KEY:
        return "TAVILY_API_KEY not set. Cannot extract content."
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


researcher = {
    "name": "researcher",
    "description": (
        "Deep research specialist with web search access. "
        "Give it a complete list of topics to research. "
        "It will search the web, extract key sources, and save all findings to "
        "research/ files in the session. Returns the paths to the saved research files."
    ),
    "system_prompt": (
        "You are a thorough research specialist. "
        "For each topic you are given: search for the latest information, "
        "extract full page content for the most relevant sources, "
        "and save structured findings to research/<topic-slug>.md. "
        "Once all topics are researched, return a message listing the file paths "
        "where you saved your findings, e.g.: 'Saved research to: research/ollama.md, research/vllm.md'"
    ),
    "tools": [web_search, web_extract],
}

writer = {
    "name": "writer",
    "description": (
        "Technical writer. Reads research files specified by the caller, then produces "
        "a polished document and returns the full content as its final response."
    ),
    "system_prompt": (
        "You are a professional technical writer. "
        "You will be given a task description and one or more research file paths to read. "
        "Read all specified research files using read_file, then write the requested document. "
        "Return the complete document content in your final response — do not save it to a file."
    ),
}

agent = create_deep_agent(
    model="gpt-4o-mini",
    name="orchestrator",
    subagents=[researcher, writer],
    system_prompt=(
        "You are a research orchestrator. Follow this workflow:\n"
        "1. Use write_todos to plan your steps before starting.\n"
        "2. Delegate ALL research topics to the researcher subagent in a single task call "
        "(not one call per topic). The researcher will save its findings to research/ files "
        "and tell you which files it saved.\n"
        "3. Once you have the research file paths, pass them to the writer subagent along "
        "with the user's request and exact output format required.\n"
        "4. Return the writer's output directly to the user — do not save to a file."
    ),
    interrupt_on=["web_search"],
    debug=True,
    db_path="tests/demo.db",
)

TASK = """
One compelling task for a research agent is to investigate the long-term viability of sodium-ion batteries as a sustainable alternative to lithium-ion technology in the electric vehicle market. The research should analyze current energy density limitations, the geopolitical stability of the raw material supply chain, and the existing manufacturing hurdles for large-scale adoption. Additionally, the agent should identify key startups leading the sector and provide a cost-benefit analysis comparing the lifecycle environmental impact of sodium versus lithium extraction.
"""

if __name__ == "__main__":
    session_id = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    result = agent.run_sync(TASK, session_id=session_id)

    print("\n" + "=" * 70)
    print("FINAL OUTPUT:")
    print("=" * 70)
    print(result.output)
    print("=" * 70)
    print(f"\nSession: {result.session_id}  (resume with SESSION_ID={result.session_id})")
    print("\nPlan:")
    for i, todo in enumerate(result.plan.todos, 1):
        print(f"  [{i}] ({todo.status.value}) {todo.title}")
