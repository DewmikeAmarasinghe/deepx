from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import yaml


class SkillMetadata(TypedDict):
    name: str
    description: str
    path: str


def discover_skills(skills_paths: list[str]) -> list[SkillMetadata]:
    """Scan each path in *skills_paths* for ``SKILL.md`` files and return their metadata."""
    found: list[SkillMetadata] = []
    for raw in skills_paths:
        root = Path(raw)
        if not root.exists():
            continue
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            meta = _parse_skill_frontmatter(skill_md.read_text(), str(skill_md))
            if meta:
                found.append(meta)
    return found


def format_skills_for_prompt(skills: list[SkillMetadata]) -> str:
    """Format skill list for injection into the system prompt."""
    if not skills:
        return ""
    lines: list[str] = []
    for skill in skills:
        lines.append(f"- **{skill['name']}**: {skill['description']}")
        lines.append(f"  -> Read `{skill['path']}` for full instructions")
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
    return SkillMetadata(name=name, description=description, path=path)