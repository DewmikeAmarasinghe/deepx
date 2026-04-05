import re
from pathlib import Path

import yaml


class SkillsLoader:
    @staticmethod
    def discover(path: str) -> list[dict]:
        skills = []
        base = Path(path)
        for skill_md in base.rglob("SKILL.md"):
            content = skill_md.read_text(encoding="utf-8")
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if not match:
                continue
            try:
                meta = yaml.safe_load(match.group(1))
            except yaml.YAMLError:
                continue
            if not isinstance(meta, dict):
                continue
            name = str(meta.get("name", "")).strip()
            description = str(meta.get("description", "")).strip()
            if not name or not description:
                continue
            skills.append(
                {
                    "name": name,
                    "description": description,
                    "path": str(skill_md),
                    "allowed_tools": meta.get("allowed-tools", ""),
                }
            )
        return skills

    @staticmethod
    def format_for_prompt(skills: list[dict]) -> str:
        if not skills:
            return "(no skills available)"
        lines = []
        for s in skills:
            lines.append(f"- **{s['name']}**: {s['description']}")
            lines.append(f"  → read `{s['path']}` for full instructions")
        return "\n".join(lines)
