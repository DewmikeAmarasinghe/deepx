├── deep_research
│   ├── agents
│   │   ├── clarifying_agent.py
│   │   ├── __init__.py
│   │   ├── planner_agent.py
│   │   ├── search_agent.py
│   │   ├── triage_agent.py
│   │   └── writer_agent.py
│   ├── __init__.py
│   ├── models.py
│   └── workflows
│       ├── research_manager.py
│       └── research_workflow.py
├── .env
├── .env-sample
├── .github
│   └── CODEOWNERS
├── .gitignore
├── pyproject.toml
├── README.md
├── run_server.py
├── run_worker.py
├── temporal.md
├── ui
│   ├── index.html
│   ├── src
│   │   ├── css
│   │   │   └── styles.css
│   │   └── js
│   │       └── api-client.js
│   └── success.html
└── uv.lock

9 directories, 24 files

~/Documents/Projects/temporal.io/edu-deep-research-tutorial-template (edu-deep-research-tutorial-template) main* 
❯ 

```search_agent.py
"""
Search Agent - Performs web searches and summarizes results.

This agent uses the built-in WebSearchTool to execute real web searches
and produces concise summaries for the research report.
"""

from agents import Agent, Runner, WebSearchTool

from deep_research.models import SearchResult

SEARCH_PROMPT = """You are a research assistant that searches the web and summarizes results.

Given a search query and the reason for the search, use the web search tool
to find relevant information, then produce a concise summary.

Your summary should:
- Be 1-2 paragraphs, under 250 words
- Capture the main points and key facts
- Include specific details, numbers, or examples where appropriate
- Be written in a note-taking style (concise, not full sentences if needed)
- Focus on information relevant to the research reason provided
"""


def new_search_agent() -> Agent:
    """Create a new search agent with web search capabilities."""
    return Agent(
        name="Search Agent",
        instructions=SEARCH_PROMPT,
        model="gpt-4o-mini",
        tools=[WebSearchTool()], 
    )


async def perform_web_search(query: str, reason: str) -> str:
    """
    Perform a web search and return a summary of results.

    Args:
        query: The search query
        reason: Why this search is being performed

    Returns:
        A summary of the search results
    """
    agent = new_search_agent()
    prompt = f"Search for: {query}\nReason for search: {reason}\n\nSearch the web and provide a summary of the results."
    result = await Runner.run(agent, prompt)
    search_result = result.final_output_as(SearchResult)
    return search_result.summary
```


```research_workflow.py
from dataclasses import dataclass, field

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from deep_research.workflows.research_manager import InteractiveResearchManager
    from deep_research.models import ReportData

@dataclass
class UserQueryInput:
    """Input for starting research."""
    query: str


@dataclass
class SingleClarificationInput:
    """Input for providing a single clarification answer."""
    answer: str


@dataclass
class ResearchStatus:
    """Current status of the research workflow."""
    original_query: str | None
    clarification_questions: list[str]
    clarification_responses: list[str]
    status: str


@dataclass
class InteractiveResearchResult:
    """Final result from the research workflow."""
    short_summary: str
    markdown_report: str
    follow_up_questions: list[str]


@workflow.defn
class InteractiveResearchWorkflow:
    """
    Long-running workflow for interactive research with clarifying questions.

    The workflow:
    1. Waits for research to be started via update
    2. If clarifications needed, waits for each answer via updates
    3. Once all answers collected, completes research
    4. Returns the final report
    """

    def __init__(self) -> None:
        self.research_manager = InteractiveResearchManager()

        # State that persists across crashes
        self.original_query: str | None = None
        self.clarification_questions: list[str] = []
        self.clarification_responses: list[str] = []
        self.report_data: ReportData | None = None
        self.research_initialized: bool = False


    def _build_result(
        self,
        summary: str,
        report: str,
        questions: list[str],
    ) -> InteractiveResearchResult:
        """Helper to build the result object."""
        return InteractiveResearchResult(
            short_summary=summary,
            markdown_report=report,
            follow_up_questions=questions,
        )
    
    @workflow.query
    def get_status(self) -> ResearchStatus:
        """Get current research status."""
        if self.report_data:
            status = "completed"
        elif self.clarification_questions and len(self.clarification_responses) < len(self.clarification_questions):
            status = "awaiting_clarification"
        elif self.original_query:
            status = "researching"
        else:
            status = "pending"

        return ResearchStatus(
            original_query=self.original_query,
            clarification_questions=self.clarification_questions,
            clarification_responses=self.clarification_responses,
            status=status,
        )

    @workflow.update
    async def start_research(self, input: UserQueryInput) -> ResearchStatus:
        """Start a new research session."""
        workflow.logger.info(f"Starting research for: '{input.query}'")
        self.original_query = input.query

        # Check if clarifications are needed (calls the manager)
        result = await self.research_manager.run_with_clarifications_start(self.original_query)

        if result.needs_clarifications:
            self.clarification_questions = result.questions
        else:
            self.report_data = result.report_data

        self.research_initialized = True
        return self.get_status()
    

    @workflow.update
    async def provide_clarification(self, input: SingleClarificationInput) -> ResearchStatus:
        """Provide an answer to the current clarification question."""
        self.clarification_responses.append(input.answer)
        return self.get_status()
    

    @workflow.run
    async def run(self) -> InteractiveResearchResult:
        """Waits for research to start and complete."""
        # Wait for research to be initialized via the start_research Update
        await workflow.wait_condition(lambda: self.research_initialized)

        # If clarifications needed, wait for all answers
        if self.clarification_questions:
            await workflow.wait_condition(
                lambda: len(self.clarification_responses) >= len(self.clarification_questions)
            )
            # Complete research with the enriched query
            self.report_data = await self.research_manager.run_with_clarifications_complete(
                self.original_query,
                self.clarification_questions,
                self.clarification_responses,
            )

        # Return the final report
        return self._build_result(
            self.report_data.short_summary,
            self.report_data.markdown_report,
            self.report_data.follow_up_questions,
        )
```

```run_server.py
"""
FastAPI Backend for Deep Research Agent.

Run with: uv run run_server.py
"""

import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import timedelta

from temporalio.client import Client
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin, ModelActivityParameters

from deep_research.workflows.research_workflow import (
    InteractiveResearchWorkflow,
    UserQueryInput,
    SingleClarificationInput,
)

load_dotenv()

TASK_QUEUE = "deep-research-queue"

temporal_client: Client = None

app = FastAPI(title="Deep Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://0.0.0.0:8234", "http://localhost:8234"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class StartResearchRequest(BaseModel):
    query: str


class AnswerRequest(BaseModel):
    answer: str


# Static File Serving
@app.get("/")
async def serve_index():
    """Serve the main chat interface."""
    index_path = Path(__file__).parent / "ui" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    raise HTTPException(status_code=404, detail="Index page not found")


@app.get("/success")
async def serve_success():
    """Serve the success/results page."""
    success_path = Path(__file__).parent / "ui" / "success.html"
    if success_path.exists():
        return HTMLResponse(content=success_path.read_text())
    raise HTTPException(status_code=404, detail="Success page not found")


static_path = Path(__file__).parent / "ui"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# API Endpoints
@app.on_event("startup")
async def startup():
    """Connect to Temporal on server startup."""
    global temporal_client

    openai_plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=120),
        )
    )

    temporal_client = await Client.connect(
        "localhost:7233",
        namespace="default",
        plugins=[openai_plugin],
    )
    print("Connected to Temporal!")


@app.post("/api/start-research")
async def start_research(request: StartResearchRequest):
    workflow_id = f"research-{uuid.uuid4()}"

    handle = await temporal_client.start_workflow(
        InteractiveResearchWorkflow.run,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    status = await handle.execute_update(
        InteractiveResearchWorkflow.start_research,
        UserQueryInput(query=request.query.strip()),
    )

    return {
        "session_id": workflow_id,
        "status": status.status,
        "clarification_questions": status.clarification_questions,
    }

@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    handle = temporal_client.get_workflow_handle(session_id)
    status = await handle.query(InteractiveResearchWorkflow.get_status)

    # Compute current question for the UI
    current_question = None
    current_question_index = 0
    if status.clarification_questions:
        current_question_index = len(status.clarification_responses)
        if current_question_index < len(status.clarification_questions):
            current_question = status.clarification_questions[current_question_index]

    return {
        "session_id": session_id,
        "status": status.status,
        "clarification_questions": status.clarification_questions,
        "clarification_responses": status.clarification_responses,
        "current_question": current_question,
        "current_question_index": current_question_index,
    }

@app.post("/api/answer/{session_id}/{question_index}")
async def submit_answer(session_id: str, question_index: int, request: AnswerRequest):
    handle = temporal_client.get_workflow_handle(session_id)

    status = await handle.execute_update(
        InteractiveResearchWorkflow.provide_clarification,
        SingleClarificationInput(answer=request.answer.strip()),
    )

    return {"status": "accepted", "session_status": status.status}

@app.get("/api/result/{session_id}")
async def get_result(session_id: str):
    handle = temporal_client.get_workflow_handle(session_id)
    result = await handle.result()

    return {
        "session_id": session_id,
        "short_summary": result["short_summary"],
        "markdown_report": result["markdown_report"],
        "follow_up_questions": result["follow_up_questions"],
    }

if __name__ == "__main__":
    import uvicorn

    print("Starting Deep Research Agent on http://localhost:8234")
    print("Open your browser to http://localhost:8234 to use the chat interface")
    print("\nWARNING: This version has NO durability. If the server crashes,")
    print("all research sessions are lost. Complete the tutorial to add Temporal!")

    uvicorn.run(app, host="0.0.0.0", port=8234)

```


```run_worker.py
import asyncio
from datetime import timedelta

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import Worker
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin, ModelActivityParameters

from deep_research.workflows.research_workflow import InteractiveResearchWorkflow

load_dotenv()  # Load OPENAI_API_KEY from .env

TASK_QUEUE = "deep-research-queue"


async def main():
    """Start the Worker with OpenAI Agents integration."""

    # Configure OpenAI Agents plugin for automatic LLM durability
    openai_plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                backoff_coefficient=2.0,
                maximum_attempts=5,
            ),
        )
    )

    # Connect to Temporal
    client = await Client.connect(
        "localhost:7233",
        namespace="default",
        plugins=[openai_plugin],
    )

    # Create worker
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[InteractiveResearchWorkflow],
    )

    print("Worker started on task queue: deep-research-queue")
    print("Press Ctrl+C to stop")

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
```