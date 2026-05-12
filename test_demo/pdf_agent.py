"""PDF specialist subagent: read, extract, create, and manipulate PDF files.

Run standalone::

    python test_demo/pdf_agent.py --chat
    python test_demo/pdf_agent.py --chat_sync
    python test_demo/pdf_agent.py --chat --session <id>
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from deepx.backends.local_shell import LocalShellBackend  # noqa: E402
from deepx.factory import create_deep_agent  # noqa: E402

pdf_sys = """\
You are the PDF specialist. You handle all PDF work: text and table extraction, metadata
inspection, form filling, merging, splitting, and creating new PDFs from markdown or
structured content.

## Before starting any task

Read the bundled PDF skills under `/test_demo/skills/pdf/` for helper scripts, libraries, and shell patterns.

## Input files

When the orchestrator provides input paths (e.g. a markdown report from web_agent), read
them with `read_file` and use their content as source material. Do not ask the user to
paste content that is already in a file.

## Output

Choose the output path as follows:
- If the orchestrator specified a path in its brief, write there.
- Otherwise write to `/_outputs/<descriptive_name>.pdf` (or the appropriate extension).

Return only: the output file path(s) and a two-to-three sentence summary of what was
produced. Do not paste file contents in the reply.
"""

pdf_agent_runner = create_deep_agent(
    name="pdf_agent",
    memory=[".deepx/AGENTS.md"],
    description=(
        "PDF specialist: reads, extracts, creates, merges, splits, and manipulates PDF files. "
        "Handles text and table extraction, metadata, form filling, and generating new PDFs from "
        "provided content or file paths. Writes output files to `/_outputs/` and returns paths "
        "plus a short summary."
    ),
    tools=[],
    skills=["./test_demo/skills/pdf"],
    system_prompt=pdf_sys,
    backend=LocalShellBackend(_REPO_ROOT),
    checkpointer=str(_REPO_ROOT / "test_demo" / "dbs" / "agent_dbs" / "pdf_agent.db"),
    debug=True,
    subagents=None,
    interrupt_on=["execute"],
)