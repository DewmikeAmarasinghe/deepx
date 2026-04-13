from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import yaml
from agents import Agent, RunContextWrapper

from deepx.context import AgentContext

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

BASE_AGENT_PROMPT = """\
You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools.
You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- Don't say "I'll now do X" — just do it.
- If the request is ambiguous, ask questions before acting.
- If asked how to approach something, explain first, then act.

## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs.
- Disagree respectfully when the user is incorrect.
- Avoid unnecessary superlatives, praise, or emotional validation.

## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing patterns. Quick but thorough.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not against your own output.
   Your first attempt is rarely correct — iterate.

Keep working until the task is fully complete. Don't stop partway and explain what you would do
— just do it. Only yield back to the user when the task is done or you're genuinely blocked.

**When things go wrong:** stop and analyse *why* — don't keep retrying the same approach.

## Progress and final messages

During long work you may send short status lines if needed. When you deliver the **final**
answer to a human user, keep it clean: no section titles like "Progress update", no editor or
terminal path references, no `@`-style file pointers to local tooling, and no meta-recaps of
your own process unless the user asked for a post-mortem.\
"""

SUBAGENT_ROLE_PROMPT = """\
You are a subagent. Your response goes to an orchestrating agent, not to a human user.

Return structured results: include the exact file paths of every file you created, key
findings, and any data the orchestrator needs to proceed. Be direct and concise — the
orchestrator reads your output programmatically.\
"""

PLANNING_PROMPT = """\
## Working Rules

1. **Understand your capabilities before planning.** Review your tool list, identify every
   tool that delegates to a **subagent** (full nested agents) from its description, and check
   your skills. If a skill matches the task, call `read_file` on its path before writing your plan.
   Then ask yourself: given these specific tools, subagents, and skills — what is the best strategy for this task?

2. **Planning tools:** Use `write_todos` once to create the initial plan (ids will be `1`, `2`, …).
   After that, prefer `update_todos` to change status or titles — it is cheaper than replacing
   the full list. Use `write_todos` again only when you need to reset the whole plan.

3. **Execution loop (typical step):**
   ```
   tool / subagent call → then update_todos to mark progress (completed / next in_progress)
   ```
   Call `think_tool` only when something is unclear, surprising, or you have been autonomous
   for many steps and want to sanity-check the plan — not after every tool result.
   If the plan structure must change a lot, use `write_todos` to replace the todo list, or combine
   `update_todos` patches as needed.

4. **Deliver final results inline.** When responding to a human user, write all content
   directly in your response — do not reference internal file paths in the user-visible text.
   (Not applicable when you are a subagent returning results to an orchestrator.)

## Session filesystem conventions (suggested layout)

These are conventions only — the backend does not enforce folder names. Prefer paths under
`/_workspace_/` for everything the agent creates.

- **`/_workspace_/scratch/`** — optional notes, drafts, and scratch work.
- **`/_workspace_/research/`** — web research notes, scraped summaries, topic digests (if you split them out).
- **`/_workspace_/large_tool_results/`** — where oversized tool output is spilled automatically; use `read_file` there.

Keep the number of files **reasonable** (prefer fewer, well-structured files when it helps), but
you may use multiple files when the task warrants it.

---

## Phase 1 — Capability Assessment (before writing any plan)

1. **Inventory your tools.** Tools whose descriptions say they run a **subagent** invoke full
   autonomous agents that can handle complex, multi-step tasks. Understand what each specialises in before planning.
2. **Evaluate your skills.** If any skill matches the task, call `read_file` on its path now —
   before writing a single todo. The skill may define quality standards that change your approach.
3. **Design a strategy-driven plan.** Write todos that leverage your real capabilities:

   Bad — many small calls to the same subagent that could do the work in one go:
   ```
   [1] Call research agent: subtask A
   [2] Call research agent: subtask B
   [3] Call research agent: subtask C
   ```
   Good — one delegation per subagent scope, high-level prompts, paths passed between steps:
   ```
   [1] Assess capabilities and read matching skill files   (in_progress)
   [2] Call the appropriate subagent tool ONCE with the full scope of its work; ask it to
       consolidate outputs into as few session files as practical and return paths.
   [3] Pass those paths to the next subagent or finish the task yourself
   [4] Return the final result inline to the user
   ```

## Phase 2 — Execution Loop (repeat for every step)

```
EXECUTE         → perform the step (tool or subagent delegation)
THEN:
  expected result  → update_todos: mark step completed, next in_progress
  plan structure change → write_todos (full replace) or simple update_todos patches for todo status updates or todo title updates after selecting a todo by its id
  blocked          → update_todos: describe blocker; keep current in_progress
  confused / surprised / have_been_runing_autonomously_for_a_long_time → think_tool, then update_todos or write_todos or continue
```

## Phase 3 — Delegation to subagents

- Give each subagent tool a **complete, self-contained prompt** — it cannot ask follow-up
  questions and has no memory of your prior work. Include all context it needs.
- **Call each subagent once with the full scope** of what that agent is responsible for. Do not
  split the same agent's work across many tiny calls if one call can carry the whole brief.
- Ask subagents to **use as few files as practical** (fewer files usually means fewer follow-up
  reads). They may still use multiple files when the task requires it.
- **Pass file paths between agents** — do not read large file bodies only to paste them back
  into the next prompt. Subagents share the same session filesystem.
- Parallelise subagent calls only when there is **zero data dependency** between them.

## Rules for `write_todos` and `update_todos`

- `write_todos`: full replace; pass every step in order; never omit entries you want to keep.
- `update_todos`: patch by numeric id (`"1"`, `"2"`, …).
- Never call `write_todos` in parallel with other planning tools.\
"""

def _build_filesystem_prompt(session_id: str, checkpointer: str) -> str:
    """Build the filesystem section with the actual session scope injected."""
    scope_note: str
    if checkpointer == ":memory:":
        scope_note = (
            "Your workspace is **ephemeral** (in-memory). Paths are scoped to this run and "
            f"session `{session_id}`."
        )
    else:
        scope_note = (
            f"Persistent session data for this conversation is tied to session `{session_id}`. "
            "Paths under `/_workspace_/` persist for this session across runs when a checkpointer "
            "is configured."
        )

    return f"""\
## Filesystem

{scope_note}

### Scope (on disk vs agent paths)

On the host, session data lives under a canonical `.deepx` data directory: `data_root` is the
host workspace directory plus `/.deepx` (unless the host path already *is* a `.deepx` folder).
Under that: `sessions/{{id}}/_workspace_/` is exposed as **`/_workspace_/`**; cross-session files
live in `data_root/memory/` and appear as **`/_memory_/`** (for example `memory/AGENTS.md` ↔
`/_memory_/AGENTS.md`).

All paths start with `/`. **What you can access is enforced by the backend**, not by the tools.

- **`/_workspace_/`** — your private working tree for this session. Prefer this for agent-created files.
- **`/_memory_/`** — persistent memory shared across sessions (for example `/_memory_/AGENTS.md`).
- **Paths like `/src/...` or `/skills/...`** — files under the host project root the backend
  attached; use for repository sources, bundled skills, and anything else that lives on disk
  under that root. Skill entries in your prompt list concrete paths here — read them with
  `read_file` / `glob` like any other file (respect repository layout; do not invent mount namespaces).

### `ls` — list directory contents
Use before `read_file` or `edit_file` to confirm a file exists and explore the structure.

### `read_file` — read a file
- Reads up to `limit` lines starting at `offset` (0-indexed, default limit=100).
- Paginate large files: `read_file(path, limit=100)`, then `read_file(path, offset=100, ...)`.
- Always read a file before editing it.
- File contents are returned as plain text (binary files are decoded with replacement for invalid UTF-8).

### `grep` — search file contents
Literal pattern search under an optional directory or glob filter.

### `glob` — match paths by pattern
Find files relative to a directory using glob syntax.

### `write_file` — create a new file
Fails if the file already exists. Prefer editing over recreating.

### `edit_file` — replace an exact string in a file
Read the file first. Provide enough context in `old_string` to uniquely identify the location.
Use `replace_all=True` to replace every occurrence.

## Large tool results

When a tool result is too large, it is saved under `/_workspace_/large_tool_results/<call_id>.txt`.
Use `read_file` with pagination to inspect it.\
"""


MEMORY_PROMPT = """\
<agent_memory>
{agent_memory}
</agent_memory>

The above memory was loaded from your persistent store (`/_memory_/` paths). Treat it as prior knowledge.

**When to update memory** (use `read_file` then `edit_file` on the same `/_memory_/...` path):
- User explicitly asks you to remember something
- User corrects your behaviour or describes how you should work
- You discover a pattern or convention worth retaining for future sessions

**When NOT to update memory:**
- Temporary or one-time information (task requests, transient state, small talk)
- Credentials or API keys — never store these\
"""

SKILLS_PROMPT = """\
You have access to a skills library. Each skill provides domain knowledge, quality standards,
and proven workflows for a specialised task.

**Progressive disclosure:** each skill row points at one `SKILL.md` path under the host tree
(for example `/skills/pdf/SKILL.md`). Read that file first. Deeper material (`reference.md`,
`scripts/`, etc. next to it) is **on demand** — use `read_file` or `glob` from that directory when
you need it; do not assume those files are loaded into context automatically.

**Before writing your plan**, check every skill below. If any matches the task, call
`read_file` on its path now — this may fundamentally change your approach.

**Available skills:**

{skills_list}\
"""

HITL_PROMPT = """\
The following tools require human approval in this session: {tools}

**Approval is enforced at the code level — do not ask the user whether you may use these tools.**
Simply call the tool when your plan requires it. The system handles the approval prompt automatically.

If a tool returns a message starting with `[Human-in-the-loop]` saying approval was **declined**:
the tool did NOT run. Call `think_tool` if needed, then `update_todos` or `write_todos`, and proceed differently.\
"""

_SEP = "\n\n" + "=" * 80 + "\n\n"


# ---------------------------------------------------------------------------
# Skill discovery
# ---------------------------------------------------------------------------

class SkillMetadata(TypedDict, total=False):
    name: str
    description: str
    path: str
    preview: str
    license: str | None
    compatibility: str | None
    allowed_tools: list[str]


def _skill_md_in_dir(d: Path) -> Path | None:
    p = d / "SKILL.md"
    return p if p.is_file() else None


def _iter_skill_md_files(paths: list[str]) -> list[Path]:
    """Return SKILL.md paths for all bundles found under the configured skill roots."""
    md_files: list[Path] = []
    for raw in paths:
        root = Path(raw)
        if not root.exists():
            continue
        candidates: list[Path] = []
        direct = _skill_md_in_dir(root)
        if direct:
            candidates.append(direct)
        elif root.is_dir():
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                sm = _skill_md_in_dir(child)
                if sm:
                    candidates.append(sm)
                else:
                    for sub in sorted(child.iterdir()):
                        if sub.is_dir():
                            sm2 = _skill_md_in_dir(sub)
                            if sm2:
                                candidates.append(sm2)
        md_files.extend(candidates)
    return md_files


def _skill_roots_ordered(skill_root_dirs: list[str]) -> list[str]:
    """User-level skills first (~/.deepx/skills), then configured roots (dedup)."""
    out: list[str] = []
    seen: set[str] = set()
    u = Path.home() / ".deepx" / "skills"
    if u.is_dir():
        s = str(u.resolve())
        if s not in seen:
            seen.add(s)
            out.append(s)
    for raw in skill_root_dirs:
        s = str(Path(raw).expanduser().resolve())
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def skills_catalog_for_host(host_root: Path, skill_root_dirs: list[str]) -> list[SkillMetadata]:
    """Skill entries for the prompt using paths under the backend host root (agent paths `/...`)."""
    host_resolved = host_root.expanduser().resolve()
    roots = _skill_roots_ordered(skill_root_dirs)
    by_name: dict[str, SkillMetadata] = {}
    for md_file in _iter_skill_md_files(roots):
        smd = md_file.resolve()
        if not smd.is_file():
            continue
        try:
            rel = smd.relative_to(host_resolved)
            agent_path = "/" + rel.as_posix()
        except ValueError:
            continue
        text = smd.read_text(encoding="utf-8", errors="replace")
        meta = _parse_skill_frontmatter(text, agent_path)
        if meta and meta.get("name") and meta.get("description"):
            meta = {**meta, "path": agent_path}
            by_name.setdefault(str(meta["name"]), meta)
    return list(by_name.values())


def discover_skills(paths: list[str]) -> list[SkillMetadata]:
    """Discover skills: each skill is a folder containing SKILL.md (required).

    For each configured path:
    - If `path/SKILL.md` exists, load that skill.
    - Else if `path` is a directory, load `path/<child>/SKILL.md` for each immediate child.
    - Else if `path/<child>` is a directory without SKILL.md, load `path/<child>/<sub>/SKILL.md`
      for each immediate subfolder (one extra level, e.g. sql/query-writing).

    Paths in metadata point at resolved filesystem locations. For agent prompts under a host
    root, use `skills_catalog_for_host` so paths match `read_file` on the FilesystemBackend.
    """
    by_name: dict[str, SkillMetadata] = {}
    for md_file in _iter_skill_md_files(paths):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        meta = _parse_skill_frontmatter(text, str(md_file.resolve()))
        if meta and meta.get("name") and meta.get("description"):
            by_name.setdefault(str(meta["name"]), meta)
    return list(by_name.values())


def format_skills_for_prompt(skills: list[SkillMetadata]) -> str:
    if not skills:
        return ""
    lines: list[str] = []
    for skill in skills:
        lines.append(f"- **{skill['name']}**: {skill['description']}")
        lines.append(f"  Path: `{skill['path']}`")
        preview = skill.get("preview", "")
        if preview:
            indented = "\n".join(f"  | {line}" for line in preview.splitlines())
            lines.append(f"  Preview:\n{indented}")
    return "\n".join(lines)


def _extract_preview(body: str, max_words: int = 150) -> str:
    """Extract a plain-text preview from a markdown body, capped at max_words words."""
    text = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", lambda m: m.group(0)[1:-1], text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"^[\-\*\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def _parse_skill_frontmatter(content: str, path: str) -> SkillMetadata | None:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return None
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("name", "")).strip()
    description = str(data.get("description", "")).strip()
    if not name or not description:
        return None
    body = content[match.end():]
    preview = _extract_preview(body)
    meta: SkillMetadata = {
        "name": name,
        "description": description,
        "path": path,
        "preview": preview,
    }
    if data.get("license") is not None:
        meta["license"] = str(data["license"])
    if data.get("compatibility") is not None:
        meta["compatibility"] = str(data["compatibility"])
    at = data.get("allowed-tools") or data.get("allowed_tools")
    if isinstance(at, list):
        meta["allowed_tools"] = [str(x) for x in at]
    elif isinstance(at, str):
        meta["allowed_tools"] = [at]
    return meta


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------

def _section(title: str, content: str) -> str:
    return f"# {title}\n\n{content}"


def build_system_prompt(
    ctx: RunContextWrapper[AgentContext],
    agent: Agent,
    custom_prompt: str = "",
    checkpointer: str = ":memory:",
) -> str:
    sections: list[str] = []

    if custom_prompt:
        sections.append(_section("ROLE", custom_prompt))

    sections.append(_section("CORE BEHAVIOR", BASE_AGENT_PROMPT))

    utc_now = datetime.now(timezone.utc)
    sections.append(
        _section(
            "CONTEXT",
            f"Current date and time (UTC): {utc_now.strftime('%Y-%m-%d %H:%M:%S')} ({utc_now.date().isoformat()}).",
        )
    )

    if ctx.context.is_subagent:
        sections.append(_section("YOUR ROLE", SUBAGENT_ROLE_PROMPT))

    sections.append(_section("PLANNING & DELEGATION", PLANNING_PROMPT))

    all_skills = ctx.context.skills.strip()

    if all_skills:
        sections.append(_section("SKILLS", SKILLS_PROMPT.format(skills_list=all_skills)))

    if ctx.context.memory:
        sections.append(
            _section("MEMORY", MEMORY_PROMPT.format(agent_memory=ctx.context.memory))
        )

    sections.append(
        _section(
            "FILESYSTEM",
            _build_filesystem_prompt(ctx.context.session_id, checkpointer),
        )
    )

    if ctx.context.hitl_tools:
        sections.append(
            _section(
                "HUMAN-IN-THE-LOOP",
                HITL_PROMPT.format(tools=", ".join(ctx.context.hitl_tools)),
            )
        )

    if ctx.context.plan.todos:
        lines = [
            f"[{t.id}] ({t.status.value}) {t.title}"
            for t in ctx.context.plan.todos
        ]
        sections.append(_section("CURRENT PLAN", "\n".join(lines)))

    gf = ctx.context.backend.glob(ctx.context.session_id, "**/*", "/_workspace_/")
    if not gf.error and gf.files:
        paths = sorted({f.path for f in gf.files if not f.is_dir})[:50]
        if paths:
            block = "\n".join(paths)
            if len(paths) >= 50:
                block += "\n... use `glob` or `ls` under `/_workspace_/` to explore further."
            sections.append(_section("SESSION WORKSPACE FILES", block))

    return _SEP.join(sections)
