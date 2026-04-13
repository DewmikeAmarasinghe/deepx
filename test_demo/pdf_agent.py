"""PDF / forms subagent (demo)."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
PDF_SKILLS_DIR = DEMO_DIR / "skills" / "pdf"


def pdf_agent_spec(*, checkpointer: str) -> dict:
    return {
        "name": "pdf_agent",
        "description": (
            "PDF and form workflows: reading, merging, splitting, extracting tables/text, fillable "
            "forms; may use `execute` for bundled scripts when appropriate."
        ),
        "system_prompt": (
            "You specialise in PDF tasks. For multi-step work, call **`write_todos` first** (after "
            "`read_file` on **pdf** / **forms** skill entry paths), then **`update_todos`** as you go.\n"
            "Follow the skill workflows; use `execute` for repo scripts when the environment has the "
            "needed Python packages. Keep outputs under the session workspace and return paths.\n"
            "**OCR / scanned PDFs:** if the user needs OCR, say clearly that **Tesseract** (or similar) "
            "must be installed on the host system—pip packages alone are not enough."
        ),
        "skills": [str(PDF_SKILLS_DIR)],
        "checkpointer": checkpointer,
    }

TASK = """
I have two research PDFs at `/test_demo/pdfs/attention.pdf` and `/test_demo/pdfs/gpt4.pdf` and I
want to study them together.

Can you summarize both in a structured way and highlight the key ideas, architectures, and
limitations?

Then compare them and explain how the ideas evolved over time.

If there are any important tables or quantitative results, extract them.

At the end, create a clean report I can keep under the session workspace, and also combine the two
PDFs into a single file. Return all paths.
"""

if __name__ == "__main__":
    from deepx import create_deep_agent
    from deepx.backends.local_shell import LocalShellBackend

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    DEMO_DIR.joinpath("dbs", "agent_dbs").mkdir(parents=True, exist_ok=True)
    cp = str(DEMO_DIR / "dbs" / "agent_dbs" / "pdf_agent_standalone.db")
    runner = create_deep_agent(
        name="pdf_agent",
        description="Standalone PDF workflows.",
        tools=[],
        skills=[str(PDF_SKILLS_DIR)],
        system_prompt=pdf_agent_spec(checkpointer=cp)["system_prompt"],
        checkpointer=cp,
        backend=LocalShellBackend(REPO_ROOT),
        debug=True,
    )
    sid = os.environ.get("SESSION_ID") or uuid.uuid4().hex[:12]
    task = TASK
    print(runner.run_sync(task, session_id=sid).output)
