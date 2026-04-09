---
name: deep_researcher
description: Domain knowledge and quality standards for multi-source web research with a written deliverable (report, analysis, comparison, ADR, etc.).
---

# Deep Research — Domain Knowledge & Quality Standards

## What this skill covers

This skill applies when the task requires:
- Investigating a topic across multiple web sources and producing a synthesised written deliverable.
- The deliverable should be a research-backed document — not a summary of what was searched.

## Research quality

Good research in this domain means:

- **Synthesise, don't summarise.** Draw conclusions across multiple sources. A file that just
  repeats what one source said is not useful.
- **Evidence-based claims.** Every factual assertion must be traceable to a source URL.
- **Structured notes.** Research files should use clear headings, key findings in bullet form,
  and short quoted excerpts (not full copy-paste of source text).
- **Inline citations.** Use `[1]`, `[2]`, … throughout. Map each number to a URL at the bottom
  of the file.
- **Cover the full scope.** If the task lists multiple topics or questions, all of them must be
  addressed before moving to the writing stage.

## Deliverable quality

The final document must:

- Have a clear structure: **Executive Summary → Findings → Analysis → Recommendations → Sources**.
- Cite every factual claim with an inline reference.
- End with a `### Sources` section: one line per source — `[n] Title: URL`.
- Use a professional, neutral tone. No meta-narration ("I searched for…", "As an AI, I…").
- Be complete and standalone — the user should not need to read any other file.

## Subagent capabilities (context)

- `web_agent_subagent` specialises in web research. It can search, extract, and write distilled
  markdown notes across many topics in a single call. Give it the complete list of topics.
  It returns the file paths it created in `research/`.
- `writer_subagent` specialises in turning research notes into polished prose. It reads the files
  you provide and returns the full document text. It does not save to files — it returns inline.

## Constraints

- Do not pass research file paths to `writer_subagent` until all research files are written.
- Pass the exact file paths — do not ask `writer_subagent` to discover files on its own.
- The final deliverable must be returned inline in your response — never as a file the user
  must open.
- Consolidate citation numbers across all research files before passing to the writer so that
  each unique URL has exactly one number throughout the final document.
