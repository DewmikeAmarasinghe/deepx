---
name: deep_researcher
description: Domain knowledge and quality standards for multi-source web research with a written deliverable (report, analysis, comparison, ADR, etc.).
---

# Deep research (this skill is for deep-research tasks)

Use this skill when the user wants **investigative web research** synthesized into a **written deliverable** (report, memo, comparison, ADR, etc.).

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
- Use a professional, neutral tone. No meta-narration ("I searched for…", "As an AI,…").
- Be complete and standalone — the user should not need to read any other file.

## Tool workflow (web stack)

Use external web tools in this order of preference:

1. **`web_search` with multiple queries in one call** when you need several **independent**
   angles (for example different entities or unrelated questions). Parallel search is efficient
   only when each query is necessary; do not duplicate near-identical queries.
2. **`web_map`** once you already know the **host** and need a **structured list of URLs** on
   that site (navigation, sitemap-style discovery). Keep `limit`, `max_depth`, and `max_breadth`
   small unless the task truly needs broad coverage — each call can spend multiple index credits.
3. **`web_extract`** on a **small, deliberate set of URLs** when you need full page text. Prefer
   a tight URL list. Optional `query` plus `chunks_per_source` can focus extraction when the API
   supports it.

**Saving work:** write structured notes under `/_workspace_/research/` (or similar) with
`write_file`. When the brief includes a **final written deliverable**, produce the polished
markdown yourself with `write_file` (for example under `/_workspace_/reports/`)—clear structure,
citations, and a **Sources** section. Large raw tool JSON belongs in files, not in chat; use
`read_file` / `grep` / `glob` to re-use what you already saved.

**Cost and breadth:** external search and map calls are billed per vendor rules. Prefer the
smallest number of calls that still meets the brief. Reserve wide maps and many parallel searches
for tasks that explicitly require exhaustive site coverage; otherwise stop when you have enough
evidence to answer.

## Parent agent / handoff

- The **calling agent** (often a coordinator) may invoke you as **`web_agent`** with the full topic
  list. You search, map, extract as needed, save structured notes, and—when required—**author the
  final markdown** for the brief. Return **paths** and short summaries for every artifact the parent
  should follow up on. Whether the parent **previews files in the terminal** is their concern, not
  yours.

## Constraints

- Do not start the final report until research notes exist unless the brief is trivially small.
- Pass exact paths to any follow-up step—do not ask the orchestrator to guess filenames.
- Keep citation numbering consistent: one number per unique URL across notes and final report.
