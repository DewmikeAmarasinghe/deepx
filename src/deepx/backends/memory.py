from __future__ import annotations

import fnmatch
import json
import re

from deepx.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)


def _safe_agent_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name) or "agent"


def _norm_agent_path(path: str) -> str:
    p = path.replace("\\", "/").strip()
    if not p.startswith("/"):
        p = "/" + p
    return p


def _split_scope(agent_path: str) -> tuple[str, str]:
    p = _norm_agent_path(agent_path)
    body = p[1:]
    if body.startswith("_workspace_/"):
        return "ws", body[len("_workspace_/") :]
    if body == "_workspace_":
        return "ws", ""
    if body.startswith("store/"):
        return "store", body[6:]
    if body == "store":
        return "store", ""
    return "root", body


def _glob_match(rel: str, pattern: str) -> bool:
    if pattern in ("**/*", "**", "*"):
        return True
    base = rel.split("/")[-1]
    return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(base, pattern)


class InMemoryBackend(BackendProtocol):
    def __init__(self) -> None:
        self._ws: dict[tuple[str, str], str] = {}
        self._root: dict[tuple[str, str], str] = {}
        self._store: dict[str, str] = {}
        self._plans: dict[tuple[str, str], str] = {}
        self._plan_logs: dict[str, list[dict]] = {}
        self._tool_logs: dict[str, list[dict]] = {}

    def _child_names_ws(self, session_id: str, prefix: str) -> dict[str, bool]:
        pfx = prefix + "/" if prefix else ""
        out: dict[str, bool] = {}
        for sid, r in self._ws.keys():
            if sid != session_id:
                continue
            if prefix and not (r == prefix or r.startswith(pfx)):
                continue
            rest = r[len(pfx) :] if pfx else r
            if not rest:
                continue
            head, _, tail = rest.partition("/")
            is_dir = bool(tail) or any(
                k[0] == session_id and k[1].startswith(pfx + head + "/") for k in self._ws.keys()
            )
            cur = out.get(head, False)
            out[head] = cur or is_dir
        return out

    def _child_names_root(self, session_id: str, prefix: str) -> dict[str, bool]:
        pfx = prefix + "/" if prefix else ""
        out: dict[str, bool] = {}
        for sid, r in self._root.keys():
            if sid != session_id:
                continue
            if prefix and not (r == prefix or r.startswith(pfx)):
                continue
            rest = r[len(pfx) :] if pfx else r
            if not rest:
                continue
            head, _, tail = rest.partition("/")
            is_dir = bool(tail) or any(
                k[0] == session_id and k[1].startswith(pfx + head + "/") for k in self._root.keys()
            )
            cur = out.get(head, False)
            out[head] = cur or is_dir
        return out

    def _child_names_store(self, prefix: str) -> dict[str, bool]:
        pfx = prefix + "/" if prefix else ""
        out: dict[str, bool] = {}
        for r in self._store.keys():
            if prefix and not (r == prefix or r.startswith(pfx)):
                continue
            rest = r[len(pfx) :] if pfx else r
            if not rest:
                continue
            head, _, tail = rest.partition("/")
            is_dir = bool(tail) or any(sk.startswith(pfx + head + "/") for sk in self._store)
            cur = out.get(head, False)
            out[head] = cur or is_dir
        return out

    def ls(self, session_id: str, path: str) -> LsResult:
        p = _norm_agent_path(path)
        if p == "/":
            entries = [
                FileInfo(path="/_workspace_", is_dir=True),
                FileInfo(path="/store", is_dir=True),
            ]
            for name, is_dir in sorted(self._child_names_root(session_id, "").items()):
                entries.append(FileInfo(path=f"/{name}", is_dir=is_dir))
            return LsResult(entries=entries)
        scope, rel = _split_scope(p)
        if scope == "ws":
            kids = self._child_names_ws(session_id, rel)
            base = p.rstrip("/")
            return LsResult(
                entries=[
                    FileInfo(path=f"{base}/{n}", is_dir=d) for n, d in sorted(kids.items())
                ]
            )
        if scope == "store":
            kids = self._child_names_store(rel)
            base = p.rstrip("/")
            return LsResult(
                entries=[
                    FileInfo(path=f"{base}/{n}", is_dir=d) for n, d in sorted(kids.items())
                ]
            )
        kids = self._child_names_root(session_id, rel)
        base = p.rstrip("/")
        return LsResult(entries=[FileInfo(path=f"{base}/{n}", is_dir=d) for n, d in sorted(kids.items())])

    def read(
        self,
        session_id: str,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        scope, rel = _split_scope(file_path)
        if scope == "ws":
            raw = self._ws.get((session_id, rel.strip("/")))
        elif scope == "store":
            raw = self._store.get(rel.strip("/"))
        else:
            raw = self._root.get((session_id, rel.strip("/")))
        if raw is None:
            return ReadResult(error=f"Error: '{file_path}' not found.")
        lines = raw.splitlines()
        total = len(lines)
        selected = lines[offset : offset + limit]
        if not selected and lines:
            return ReadResult(
                error=f"Error: offset {offset} exceeds file length ({total} lines).",
                total_lines=total,
            )
        return ReadResult(content="\n".join(selected), total_lines=total)

    def grep(
        self,
        session_id: str,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        base = _norm_agent_path(path or "/_workspace_/")
        scope, rel = _split_scope(base)
        matches: list[GrepMatch] = []

        def scan(ap: str, content: str) -> None:
            rel_path = ap.rsplit("/", 1)[-1]
            if glob and not (
                fnmatch.fnmatch(rel_path, glob)
                or fnmatch.fnmatch(ap, glob)
            ):
                return
            for i, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=ap, line_number=i, line=line))

        if scope == "ws":
            for (sid, r), content in self._ws.items():
                if sid != session_id:
                    continue
                if rel and not (r == rel or r.startswith(rel + "/")):
                    continue
                scan("/_workspace_/" + r, content)
        elif scope == "store":
            for k, content in self._store.items():
                if rel and not (k == rel or k.startswith(rel + "/")):
                    continue
                scan("/store/" + k, content)
        else:
            for (sid, r), content in self._root.items():
                if sid != session_id:
                    continue
                if rel and not (r == rel or r.startswith(rel + "/")):
                    continue
                scan("/" + r, content)
        return GrepResult(matches=matches)

    def glob(self, session_id: str, pattern: str, path: str = "/") -> GlobResult:
        base = _norm_agent_path(path)
        scope, rel = _split_scope(base)
        files: list[FileInfo] = []
        if scope == "ws":
            for (sid, r), _ in self._ws.items():
                if sid != session_id:
                    continue
                if rel and not (r == rel or r.startswith(rel + "/")):
                    continue
                rel_part = r[len(rel) + 1 :] if rel and r.startswith(rel + "/") else r
                if rel and r == rel:
                    rel_part = ""
                check = r if not rel_part else rel_part
                if not _glob_match(check, pattern):
                    continue
                files.append(FileInfo(path="/_workspace_/" + r, is_dir=False))
        elif scope == "store":
            for k, _ in self._store.items():
                if rel and not (k == rel or k.startswith(rel + "/")):
                    continue
                if not _glob_match(k, pattern):
                    continue
                files.append(FileInfo(path="/store/" + k, is_dir=False))
        else:
            for (sid, r), _ in self._root.items():
                if sid != session_id:
                    continue
                if rel and not (r == rel or r.startswith(rel + "/")):
                    continue
                if not _glob_match(r, pattern):
                    continue
                files.append(FileInfo(path="/" + r, is_dir=False))
        files.sort(key=lambda x: x.path)
        return GlobResult(files=files)

    def write(self, session_id: str, file_path: str, content: str) -> WriteResult:
        scope, rel = _split_scope(file_path)
        rel = rel.strip("/")
        if scope == "ws":
            k = (session_id, rel)
            if k in self._ws:
                return WriteResult(error=f"Cannot write to {file_path} because it already exists.")
            self._ws[k] = content
        elif scope == "store":
            if rel in self._store:
                return WriteResult(error=f"Cannot write to {file_path} because it already exists.")
            self._store[rel] = content
        else:
            k = (session_id, rel)
            if k in self._root:
                return WriteResult(error=f"Cannot write to {file_path} because it already exists.")
            self._root[k] = content
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        session_id: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        scope, rel = _split_scope(file_path)
        rel = rel.strip("/")
        if scope == "ws":
            k = (session_id, rel)
            content = self._ws.get(k)
        elif scope == "store":
            k = rel
            content = self._store.get(rel)
        else:
            k = (session_id, rel)
            content = self._root.get(k)
        if content is None:
            return EditResult(error=f"Error: '{file_path}' not found.")
        count = content.count(old_string)
        if count == 0:
            return EditResult(error=f"Error: string not found in '{file_path}'.")
        if not replace_all and count > 1:
            return EditResult(
                error=(
                    f"Error: String appears {count} times in file. "
                    "Use replace_all=True or provide a more specific old_string."
                )
            )
        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        if scope == "ws":
            self._ws[(session_id, rel)] = new_content
        elif scope == "store":
            self._store[rel] = new_content
        else:
            self._root[(session_id, rel)] = new_content
        return EditResult(path=file_path, occurrences=count if replace_all else 1)

    def save_plan(self, session_id: str, agent_name: str, plan_json: str) -> None:
        self._plans[(session_id, agent_name)] = plan_json

    def load_plan(self, session_id: str, agent_name: str) -> str | None:
        return self._plans.get((session_id, agent_name))

    def append_plan_log(self, session_id: str, entry_json: str) -> None:
        try:
            obj = json.loads(entry_json)
        except json.JSONDecodeError:
            obj = {"raw": entry_json}
        self._plan_logs.setdefault(session_id, []).append(obj)

    def save_tool_log(self, session_id: str, log_data: dict) -> None:
        logs = self._tool_logs.setdefault(session_id, [])
        tn = str(log_data["tool_name"])
        n = 1 + sum(1 for e in logs if str(e.get("tool_name")) == tn)
        logs.append({**log_data, "call_id": str(n)})
