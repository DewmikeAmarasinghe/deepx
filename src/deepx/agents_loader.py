from __future__ import annotations
import re
from pathlib import Path
from typing import TypedDict
import yaml


class AgentDef(TypedDict):
    name: str
    description: str
    instructions: str


class AgentsLoader:
    @staticmethod
    def discover(agents_root: str | Path) -> list[AgentDef]:
        root = Path(agents_root)
        if not root.exists():
            return []
        result = []
        for path in sorted(root.glob("*.md")):
            defn = AgentsLoader._parse(path)
            if defn:
                result.append(defn)
        return result

    @staticmethod
    def _parse(path: Path) -> AgentDef | None:
        content = path.read_text()
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if not match:
            instructions = content.strip()
            name = path.stem
            description = f"Subagent: {name}"
        else:
            try:
                meta = yaml.safe_load(match.group(1))
            except yaml.YAMLError:
                return None
            if not isinstance(meta, dict):
                return None
            name = str(meta.get("name", path.stem)).strip()
            description = str(meta.get("description", f"Subagent: {name}")).strip()
            instructions = match.group(2).strip()
        if not name:
            return None
        return AgentDef(name=name, description=description, instructions=instructions)