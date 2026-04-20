from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NotRequired, TypedDict

import yaml
from agents import Agent, RunContextWrapper

from deepx.backends.filesystem import resolve_data_root, resolve_host_root
from deepx.context import AgentContext
from deepx.tools.planning import WRITE_TODOS_SYSTEM_APPENDIX

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

COORDINATOR_ROLE_PROMPT = """\
You coordinate specialists. Your consumer may be another agent or system—not always an end user—so
prioritize **actionable** outcomes: clear summaries, explicit file paths, and decisions that unblock
downstream work.

Use **tools** for callable delegation and **handoffs** when the SDK exposes `transfer_to_*`.
Prefer one strong delegation per specialty with **self-contained** briefs; pass **file paths**
between steps instead of pasting large bodies.

For substantial multi-step requests, call **write_todos** with a full task list, then replace the
list with **write_todos** again as steps complete. Review specialist outputs before you summarise.
"""

ORCHESTRATOR_ROLE_PROMPT = COORDINATOR_ROLE_PROMPT

FILESYSTEM_PROMPT = """\
## Filesystem and shell

### Paths (file tools)

Paths start with **`/`** and are **under the project root** (`root_dir`): `/README.md` →
`<root_dir>/README.md`, `/test_demo/foo.py` → `<root_dir>/test_demo/foo.py`. You cannot read or
write the runtime `.deepx` tree via file tools; use **`save_memory`** for durable facts.

Project root for this run: `{host_root}`.

### Tool groups (namespaces)

- **Filesystem & host:** `ls`, `read_file`, `write_file`, `edit_file`, `grep`, `glob`, `execute`
- **Planning:** `write_todos`
- **Memory:** `save_memory`

### `execute`

Prefer file tools for project files. Commands use a
**timeout** (capped); unsupported backends return a clear error instead of running a shell.

### File tools

`ls`, `read_file`, `write_file`, `edit_file`, `grep`, `glob` — paginate large reads, read before
edit, literal substring `grep`. **`grep`** supports `output_mode` (`content`, `count`,
`files_with_matches`). **`glob`** is time-bounded. **`read_file`** truncates extremely long
lines in the numbered view.

### Deliverables (project tree)

Put user-facing artifacts under **`/_outputs/`** (for example `/_outputs/report.md`). Keep the
tree tidy: remove scratch files when done.

### Large tool results

When a tool return exceeds the context budget, the framework **writes the full text** under
**`/_outputs/large_tool_results/<readable_name>.txt`** and replaces the tool message with
instructions plus a **head/tail preview**. Use **`read_file(path, offset=0, limit=100)`** (and
paginate) to pull the saved content back into context — never paste the entire file into chat.
"""


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

**Autonomy:** for standard work, do **not** ask the user where to put or other such basic questions. Use file tools under
the project tree, keep working, and return a **finished** outcome—not a partial result that waits for confirmation to continue unless you are **truly blocked** on
missing facts only the user can supply.

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
orchestrator reads your output programmatically.

**Deliverables:** write real project paths the orchestrator can `read_file`. When the parent will
**`render_files`** or ship work to a **human**, artifacts must be **complete** (not “see other file”
stubs). Remove drafts and scratch files when the task is done; keep only final outputs the user or the orchestrator
needs.

**Planning:** you have the same `write_todos` tool as the main agent.
For any multi-step task, call `write_todos` **before** heavy tool use (after any quick `read_file`
on relevant skills), then call `write_todos` again with an updated full list after each major step.
Skipping todos on multi-step work is a mistake.\
"""

HYBRID_PLANNER_EXECUTOR_PROMPT = """\
## Planner–executor loop

Use a tight **ReAct** rhythm: short reasoning when it helps, then tools, then update your view of the
world. For work that spans multiple tool batches, keep **`write_todos`** as the canonical plan:
create it early, then **replace the entire list** after each major milestone so the plan always
matches reality. Never issue parallel `write_todos` calls in the same turn.
"""

PLANNING_PROMPT = f"""\
---

{WRITE_TODOS_SYSTEM_APPENDIX}

{HYBRID_PLANNER_EXECUTOR_PROMPT}

## Step 1 — Capability Assessment (before writing any plan)

1. **Inventory your tools.** Tools whose descriptions say they run a **subagent** invoke full
   autonomous agents that can handle complex, multi-step tasks. Understand what each specialises in before planning.
2. **Evaluate your skills.** If any skill matches the task, call `read_file` on its path now —
   before writing a single todo. The skill may define quality standards that change your approach.
3. **Design a strategy-driven plan.** Use `write_todos` with items that leverage your real capabilities:

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
       consolidate outputs into as few files as practical and return paths.
   [3] Integrate results (or call the next specialist) using paths only — no huge pastes
   [4] Return the final result inline to the user (or paths to artifacts for the orchestrator)
   ```

## Step 2 — Execution Loop (repeat for every step)

```
EXECUTE         → perform the step (tool or subagent delegation)
THEN:
  progress        → write_todos again with the full list (mark completed / in_progress / add items)
  blocked         → write_todos: add or adjust items; keep honest in_progress state
  confused        → reason briefly in your assistant message, then write_todos if the list should change
```

## Step 3 — Delegation to subagents

- Give each subagent tool a **complete, self-contained prompt** — it cannot ask follow-up
  questions and has no memory of your prior work. Include all context it needs.
- **Call each subagent once with the full scope** of what that agent is responsible for. Do not
  split the same agent's work across many tiny calls if one call can carry the whole brief.
- Ask subagents to **use as few files as practical** (fewer files usually means fewer follow-up
  reads). They may still use multiple files when the task requires it.
- **Pass file paths between agents** — do not read large file bodies only to paste them back
  into the next prompt. Subagents share the same project tree.
- Parallelise subagent calls only when there is **zero data dependency** between them.

## Human approvals (host)

Some tools may **pause the run** until a human approves or rejects the tool call in the host
(terminal/UI). If a call is rejected, revise your plan with `write_todos` and try a different approach.

## Step 4 — Cleanup

- Delete intermediate scratch files when the job is finished; leave final deliverables only.
"""


MEMORY_PROMPT = """\
<agent_memory>
{agent_memory}
</agent_memory>

Loaded from persistent memory for this agent. Treat it as prior knowledge.

**When to call `save_memory`:** user asks you to remember something; stable conventions worth
keeping across sessions. **Never** store secrets.

**When not to:** one-off task state, transient chat.\
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

_SEP = "\n\n" + "=" * 80 + "\n\n"


# ---------------------------------------------------------------------------
# Skill discovery
# ---------------------------------------------------------------------------


class SkillMetadata(TypedDict):
    name: str
    description: str
    path: str
    license: NotRequired[str | None]
    compatibility: NotRequired[str | None]
    allowed_tools: NotRequired[list[str]]


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


def skills_catalog_for_host(
    host_root: Path, skill_root_dirs: list[str]
) -> list[SkillMetadata]:
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
    return "\n".join(lines)


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
    meta: SkillMetadata = {
        "name": name,
        "description": description,
        "path": path,
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
    *,
    include_coordinator_role: bool = False,
) -> str:
    sections: list[str] = []

    if custom_prompt:
        sections.append(_section("ROLE", custom_prompt))

    if include_coordinator_role:
        sections.append(_section("COORDINATION", COORDINATOR_ROLE_PROMPT))

    sections.append(_section("CORE BEHAVIOR", BASE_AGENT_PROMPT))

    utc_now = datetime.now(timezone.utc)
    host_p = resolve_host_root(ctx.context.backend)
    data_p = resolve_data_root(ctx.context.backend)
    ctx_lines = [
        f"Current date and time (UTC): {utc_now.strftime('%Y-%m-%d %H:%M:%S')} ({utc_now.date().isoformat()}).",
        f"Session id: `{ctx.context.session_id}`.",
    ]
    if host_p is not None:
        ctx_lines.append(f"Project root (`root_dir`): `{host_p}`")
    sections.append(_section("CONTEXT", "\n".join(ctx_lines)))

    if ctx.context.is_subagent:
        sections.append(_section("YOUR ROLE", SUBAGENT_ROLE_PROMPT))

    sections.append(_section("PLANNING & DELEGATION", PLANNING_PROMPT))

    all_skills = ctx.context.skills.strip()

    if all_skills:
        sections.append(
            _section("SKILLS", SKILLS_PROMPT.format(skills_list=all_skills))
        )

    if ctx.context.memory:
        sections.append(
            _section("MEMORY", MEMORY_PROMPT.format(agent_memory=ctx.context.memory))
        )

    hr = str(host_p) if host_p is not None else "the configured project root"
    _ = data_p, checkpointer
    sections.append(_section("FILESYSTEM", FILESYSTEM_PROMPT.format(host_root=hr)))

    if ctx.context.plan.todos:
        lines = [
            f"[{t.id}] ({t.status.value}) {t.content}" for t in ctx.context.plan.todos
        ]
        sections.append(_section("CURRENT PLAN", "\n".join(lines)))

    return _SEP.join(sections)
