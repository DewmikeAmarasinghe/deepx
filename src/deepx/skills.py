from __future__ import annotations
import re
from pathlib import Path
from typing import TypedDict
import yaml


class SkillMetadata(TypedDict):
    name: str
    description: str
    path: str


class SkillsLoader:
    @staticmethod
    def discover(skills_root: str | Path) -> list[SkillMetadata]:
        root = Path(skills_root)
        if not root.exists():
            return []
        skills: list[SkillMetadata] = []
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text()
            metadata = SkillsLoader._parse_frontmatter(content, str(skill_md))
            if metadata:
                skills.append(metadata)
        return skills

    @staticmethod
    def _parse_frontmatter(content: str, path: str) -> SkillMetadata | None:
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

    @staticmethod
    def format_for_prompt(skills: list[SkillMetadata]) -> str:
        if not skills:
            return ""
        lines = []
        for skill in skills:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            lines.append(f"  → Read `{skill['path']}` for full instructions")
        return "\n".join(lines)
