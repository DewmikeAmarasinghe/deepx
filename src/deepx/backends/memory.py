from __future__ import annotations

import wcmatch.glob as wcglob
from wcmatch import fnmatch as wc_fnmatch

from deepx.backends.filesystem import OUTPUTS_LARGE_TOOL_RESULTS_PREFIX
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


def _norm_agent_path(path: str) -> str:
    p = path.replace("\\", "/").strip()
    if not p.startswith("/"):
        p = "/" + p
    return p


def _rel(file_path: str) -> str:
    return _norm_agent_path(file_path).lstrip("/")


def _glob_match(rel: str, pattern: str) -> bool:
    if pattern in ("**/*", "**", "*"):
        return True
    flags = wcglob.EXTGLOB | wcglob.GLOBSTAR | wcglob.DOTGLOB
    base = rel.split("/")[-1]
    return wc_fnmatch.fnmatch(rel, pattern, flags=flags) or wc_fnmatch.fnmatch(
        base, pattern, flags=flags
    )


class InMemoryBackend(BackendProtocol):
    def __init__(self) -> None:
        self._files: dict[tuple[str, str], str] = {}

    def _child_names(self, session_id: str, prefix: str) -> dict[str, bool]:
        pfx = prefix + "/" if prefix else ""
        out: dict[str, bool] = {}
        for sid, r in self._files.keys():
            if sid != session_id:
                continue
            if prefix and not (r == prefix or r.startswith(pfx)):
                continue
            rest = r[len(pfx) :] if pfx else r
            if not rest:
                continue
            head, _, tail = rest.partition("/")
            is_dir = bool(tail) or any(
                k[0] == session_id and k[1].startswith(pfx + head + "/")
                for k in self._files.keys()
            )
            cur = out.get(head, False)
            out[head] = cur or is_dir
        return out

    def ls(self, session_id: str, path: str) -> LsResult:
        p = _norm_agent_path(path)
        rel = p.lstrip("/")
        kids = self._child_names(session_id, rel)
        base = p.rstrip("/") or "/"
        return LsResult(
            entries=[
                FileInfo(path=f"{base}/{n}" if base != "/" else f"/{n}", is_dir=d)
                for n, d in sorted(kids.items())
            ]
        )

    def read(
        self,
        session_id: str,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        r = _rel(file_path)
        raw = self._files.get((session_id, r))
        if raw is None:
            return ReadResult(error=f"Error: '{file_path}' not found.")
        lines = raw.splitlines()
        total = len(lines)
        if total > 0 and offset >= total:
            return ReadResult(
                content=(
                    f"No lines to read at offset {offset}: file has {total} line(s) "
                    f"(valid offsets are 0–{total - 1})."
                ),
                total_lines=total,
            )
        selected = lines[offset : offset + limit]
        return ReadResult(content="\n".join(selected), total_lines=total)

    def grep(
        self,
        session_id: str,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        base = _norm_agent_path(path or "/")
        rel = base.lstrip("/")
        matches: list[GrepMatch] = []
        _fg = wcglob.EXTGLOB | wcglob.GLOBSTAR | wcglob.DOTGLOB

        for (sid, r), content in self._files.items():
            if sid != session_id:
                continue
            if rel and not (r == rel or r.startswith(rel + "/")):
                continue
            ap = "/" + r
            rel_path = r.rsplit("/", 1)[-1]
            if glob and not (
                wc_fnmatch.fnmatch(rel_path, glob, flags=_fg)
                or wc_fnmatch.fnmatch(ap, glob, flags=_fg)
            ):
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=ap, line_number=i, line=line))
        return GrepResult(matches=matches)

    def glob(self, session_id: str, pattern: str, path: str = "/") -> GlobResult:
        base = _norm_agent_path(path)
        rel = base.lstrip("/")
        files: list[FileInfo] = []
        for (sid, r), _ in self._files.items():
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
            files.append(FileInfo(path="/" + r, is_dir=False))
        files.sort(key=lambda x: x.path)
        return GlobResult(files=files)

    def write(self, session_id: str, file_path: str, content: str) -> WriteResult:
        r = _rel(file_path)
        canonical = _norm_agent_path(file_path)
        allow_replace = canonical.startswith(OUTPUTS_LARGE_TOOL_RESULTS_PREFIX + "/")
        k = (session_id, r)
        if k in self._files and not allow_replace:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists."
            )
        self._files[k] = content
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        session_id: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        k = (session_id, _rel(file_path))
        content = self._files.get(k)
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
        self._files[k] = new_content
        return EditResult(path=file_path, occurrences=count if replace_all else 1)

    def execute(
        self,
        session_id: str,
        command: str,
        *,
        timeout: float = 120.0,
    ) -> str:
        _ = session_id, command, timeout
        return "Shell execution is not available on InMemoryBackend."
