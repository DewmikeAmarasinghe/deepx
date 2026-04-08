"""Unit tests for skill discovery and prompt formatting."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from deepx.system_prompt import (
    SkillMetadata,
    discover_skills,
    format_skills_for_prompt,
)


def _write_skill(root: Path, name: str, description: str, extra: str = "") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n# {name}\n\nInstructions here."
    (skill_dir / "SKILL.md").write_text(frontmatter)
    return skill_dir / "SKILL.md"


class TestDiscoverSkills:
    def test_discovers_single_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root, "web-research", "Web research skill")
            skills = discover_skills([str(root)])
            assert len(skills) == 1
            assert skills[0]["name"] == "web-research"
            assert skills[0]["description"] == "Web research skill"

    def test_discovers_multiple_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root, "web-research", "Research skill")
            _write_skill(root, "code-review", "Code review skill")
            skills = discover_skills([str(root)])
            names = {s["name"] for s in skills}
            assert "web-research" in names
            assert "code-review" in names

    def test_empty_path_returns_empty(self):
        skills = discover_skills(["/nonexistent/path"])
        assert skills == []

    def test_skill_without_name_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "no-name"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\ndescription: something\n---\n# x")
            skills = discover_skills([str(root)])
            assert skills == []

    def test_skill_without_description_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "no-desc"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n# x")
            skills = discover_skills([str(root)])
            assert skills == []

    def test_skill_no_frontmatter_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "plain"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Just a plain markdown file")
            skills = discover_skills([str(root)])
            assert skills == []

    def test_duplicate_names_last_wins(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            root1, root2 = Path(tmp1), Path(tmp2)
            _write_skill(root1, "shared", "First version")
            _write_skill(root2, "shared", "Second version")
            skills = discover_skills([str(root1), str(root2)])
            assert len(skills) == 1
            assert skills[0]["description"] == "Second version"

    def test_skill_path_is_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root, "my-skill", "A skill")
            skills = discover_skills([str(root)])
            assert Path(skills[0]["path"]).is_absolute()

    def test_optional_fields_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_skill(root, "licensed", "A licensed skill", "license: MIT\n")
            skills = discover_skills([str(root)])
            assert skills[0].get("license") == "MIT"


class TestFormatSkillsForPrompt:
    def test_empty_skills_returns_empty(self):
        assert format_skills_for_prompt([]) == ""

    def test_formats_single_skill(self):
        skills: list[SkillMetadata] = [
            {"name": "web-research", "description": "Research skill", "path": "/path/to/SKILL.md"}
        ]
        result = format_skills_for_prompt(skills)
        assert "web-research" in result
        assert "Research skill" in result
        assert "/path/to/SKILL.md" in result

    def test_formats_multiple_skills(self):
        skills: list[SkillMetadata] = [
            {"name": "research", "description": "Desc A", "path": "/a/SKILL.md"},
            {"name": "writing", "description": "Desc B", "path": "/b/SKILL.md"},
        ]
        result = format_skills_for_prompt(skills)
        assert "research" in result
        assert "writing" in result
