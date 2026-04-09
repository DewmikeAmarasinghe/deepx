"""Demo: Multi-agent research and writing pipeline.

Demonstrates:
- write_todos / think_tool for structured planning
- web_agent_subagent for web research (web_search, web_extract)
- writer_subagent for polished document generation
- HITL session approval (approve once; declines return a tool message)
- Skills loaded from tests/orchestrator/

Usage:
    # One-shot task:
    OPENAI_API_KEY=... TAVILY_API_KEY=... python tests/orchestrator/orchestrator.py

    # Interactive chat session:
    OPENAI_API_KEY=... TAVILY_API_KEY=... python tests/orchestrator/orchestrator.py --chat

    # Resume a session:
    SESSION_ID=<id> python tests/orchestrator/orchestrator.py --chat
"""

from __future__ import annotations

import os
import sys
import uuid

import httpx
from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv

from deepx import create_deep_agent
from deepx_cli import run_interactive

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


web_agent = {
    "name": "web_agent",
    "description": (
        "Web research specialist. Give it a complete list of topics or questions to investigate. "
        "It searches the web, synthesises findings, writes distilled notes to research/<topic>.md, "
        "and returns those file paths."
    ),
    "system_prompt": (
        "You are a web research specialist with web_search, web_extract, and think_tool.\n"
        "Use think_tool after each web_search or web_extract to reflect on results and plan next steps.\n"
        "For each topic: search, extract the most relevant URLs as needed, then write a synthesised "
        "markdown file at research/<topic-slug>.md — structured prose, clear headings, short quoted "
        "excerpts, inline citations [1][2] mapping to source URLs. Do NOT paste raw tool output.\n"
        "When done, return the list of file paths you wrote, e.g.: "
        "'Saved research to: research/foo.md, research/bar.md'."
    ),
    "tools": [web_search, web_extract],
}

writer = {
    "name": "writer",
    "description": (
        "Technical writer. Reads the research files you specify, then produces a polished document "
        "and returns the full content as its final response. Never asks the user to read a file."
    ),
    "system_prompt": (
        "You are a professional technical writer. "
        "You will be given a task description and one or more research file paths. "
        "Read all specified files with read_file, then write the requested document. "
        "Return the complete document inline in your final response — do not save it to a file "
        "and do not tell the user to open any file."
    ),
}

agent = create_deep_agent(
    model="gpt-5-nano",
    name="deep_researcher",
    description="Deep research orchestrator: delegates web research and writing to specialised subagents.",
    subagents=[web_agent, writer],
    skills=["tests/orchestrator"],
    system_prompt=(
        "You are the deep_researcher: orchestrate multi-step web research and document writing.\n"
        "Before planning, check your skill list for the 'deep_researcher' skill and read its file.\n"
        "Delegate research to web_agent_subagent and writing to writer_subagent. "
        "Use think_tool after major delegations or when you need to reason about next steps."
    ),
    interrupt_on=["web_search"],
    debug=True,
    db_path="",
)

TASK = """
Investigate the long-term viability of sodium-ion batteries as a sustainable alternative to
lithium-ion technology in the electric vehicle market. Analyse current energy density limitations,
the geopolitical stability of the raw material supply chain, and existing manufacturing hurdles
for large-scale adoption. Identify key startups leading the sector and compare the lifecycle
environmental impact of sodium versus lithium extraction.
After investigating, write a comprehensive report.
"""

if __name__ == "__main__":
    if "--chat" in sys.argv:
        run_interactive(agent)
    else:
        session_id = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
        result = agent.run_sync(TASK, session_id=session_id)

        print("\n" + "=" * 70)
        print(result.output)
        print("=" * 70)
        print(f"\nSession: {result.session_id}  (resume with SESSION_ID={result.session_id})")
