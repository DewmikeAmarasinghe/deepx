---
name: orchestration
description: Multi-agent research-to-document pipeline. Coordinates a researcher and writer via shared VFS files.
---

## Pipeline

This orchestrator coordinates two specialist agents in a fixed sequence:

**Step 1 — Research phase**

Delegate to the `researcher` agent. Give it a precise list of all topics to investigate. The researcher will:
- Search and extract information from the web
- Save its findings as structured markdown files under `research/` in the shared session filesystem
- Return the exact file paths it saved (e.g. `research/ollama.md, research/vllm.md`)

The researcher MUST save its findings to files and return the file paths. Do not continue until it has done so.

**Step 2 — Writing phase**

Once the researcher has returned the file paths, delegate to the `writer` agent. Give it:
- The user's original task (what document to produce, what format, what decision to make)
- The exact file paths returned by the researcher

The writer will read those files from the shared filesystem and produce the final document. It returns the complete document content as its response — it does NOT save to a file.

**Step 3 — Return to user**

Return the writer's output directly to the user. Do not save it.

## Rules

- The researcher must finish before the writer starts — the writer depends on the researcher's output files.
- Always pass explicit file paths to the writer. Do not tell it to "find" or "discover" files.
- Do not summarize or paraphrase the writer's output — return it verbatim.
