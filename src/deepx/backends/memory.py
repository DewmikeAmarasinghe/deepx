from __future__ import annotations

from pathlib import Path

import wcmatch.glob as wcglob
from wcmatch import fnmatch as wc_fnmatch

from deepx.backends.protocol import (
    OUTPUTS_PREFIX,
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
from deepx.backends.utils import (
    _coerce_agent_path,
    _norm_agent_path,
    _rel_from_agent_path,
    data_root_for_host,
)

_DEEPX_BUCKET = "__deepx__"
_GLOB_FLAGS = wcglob.EXTGLOB | wcglob.GLOBSTAR | wcglob.DOTGLOB


def _glob_match(rel: str, pattern: str) -> bool:
    if pattern in ("**/*", "**", "*"):
        return True
    base = rel.split("/")[-1]
    return wc_fnmatch.fnmatch(rel, pattern, flags=_GLOB_FLAGS) or wc_fnmatch.fnmatch(
        base, pattern, flags=_GLOB_FLAGS
    )


class InMemoryBackend(BackendProtocol):
    """Virtual workspace under ``root_dir``; metadata lives under agent paths ``/.deepx/...``."""

    def __init__(self, root_dir: str | Path = "/tmp/deepx_inmem") -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._data_root = data_root_for_host(self._root_dir)
        self._files: dict[tuple[str, str], str] = {}

    @property
    def data_root(self) -> Path:
        return self._data_root

    def resolve_path(self, session_id: str, agent_path: str) -> str | None:
        _ = session_id, agent_path
        return None

    def _canon(self, raw: str) -> str:
        return _norm_agent_path(_coerce_agent_path(self._root_dir, raw))

    @staticmethod
    def _is_deepx_agent_path(canon: str) -> bool:
        return canon == "/.deepx" or canon.startswith("/.deepx/")

    def _deepx_store_prefix(self, canon: str) -> str:
        inner = canon[len("/.deepx/") :] if canon.startswith("/.deepx/") else ""
        return f".deepx/{inner}" if inner else ".deepx"

    def _deepx_immediate_children(self, store_prefix: str) -> dict[str, bool]:
        pfx = store_prefix.rstrip("/")
        pslash = pfx + "/"
        out: dict[str, bool] = {}
        for sid, r in self._files.keys():
            if sid != _DEEPX_BUCKET:
                continue
            if r == pfx:
                continue
            if not r.startswith(pslash):
                continue
            tail = r[len(pslash) :]
            head, _, rest = tail.partition("/")
            is_dir = bool(rest) or any(
                k[0] == _DEEPX_BUCKET and k[1].startswith(pslash + head + "/")
                for k in self._files.keys()
            )
            cur = out.get(head, False)
            out[head] = cur or is_dir
        return out

    def _child_names_ws(self, session_id: str, prefix: str) -> dict[str, bool]:
        pfx = prefix + "/" if prefix else ""
        children: dict[str, bool] = {}
        for sid, r in self._files.keys():
            if sid != session_id:
                continue
            if r == ".deepx" or r.startswith(".deepx/"):
                continue
            if pfx and not (r == prefix or r.startswith(pfx)):
                continue
            rest = r[len(pfx) :] if pfx else r
            if not rest:
                continue
            head, _, tail = rest.partition("/")
            is_dir = bool(tail) or any(
                k[0] == session_id
                and not (k[1] == ".deepx" or k[1].startswith(".deepx/"))
                and k[1].startswith(pfx + head + "/")
                for k in self._files.keys()
            )
            cur = children.get(head, False)
            children[head] = cur or is_dir
        return children

    def _resolve_key(
        self, session_id: str, raw: str
    ) -> tuple[tuple[str, str] | None, str | None, str]:
        canon = self._canon(raw)
        if self._is_deepx_agent_path(canon):
            store = self._deepx_store_prefix(canon)
            return (_DEEPX_BUCKET, store), None, canon
        rel = _rel_from_agent_path(canon)
        if rel == ".deepx" or rel.startswith(".deepx/"):
            return None, "Error: paths under .deepx are not accessible via file tools.", canon
        return (session_id, rel), None, canon

    def ls(self, session_id: str, path: str) -> LsResult:
        canon = self._canon(path)
        base = canon.rstrip("/") or "/"
        if self._is_deepx_agent_path(canon):
            prefix = self._deepx_store_prefix(canon)
            kids = self._deepx_immediate_children(prefix)
            if not kids:
                if (_DEEPX_BUCKET, prefix) in self._files:
                    return LsResult(error=f"Error: '{path}' is not a directory.")
            entries: list[FileInfo] = []
            for name, is_dir in sorted(kids.items()):
                ap = f"/.deepx/{name}" if base == "/.deepx" else f"{base}/{name}"
                if is_dir:
                    entries.append(FileInfo(path=ap, is_dir=True, size=0, modified_at=""))
                else:
                    sk = f"{prefix}/{name}" if prefix != ".deepx" else f".deepx/{name}"
                    content = self._files.get((_DEEPX_BUCKET, sk), "")
                    entries.append(
                        FileInfo(
                            path=ap,
                            is_dir=False,
                            size=len(content.encode("utf-8", errors="replace")),
                            modified_at="",
                        )
                    )
            return LsResult(entries=entries)

        key, err, canon_ws = self._resolve_key(session_id, path)
        if err or key is None:
            return LsResult(error=err or "Error: path not found.")
        wr = key[1]
        kids = self._child_names_ws(session_id, wr)
        if not kids:
            if (session_id, wr) in self._files:
                return LsResult(error=f"Error: '{path}' is not a directory.")
        entries: list[FileInfo] = []
        b = canon_ws.rstrip("/") or "/"
        for name, is_dir in sorted(kids.items()):
            ap = f"{b}/{name}" if b != "/" else f"/{name}"
            if is_dir:
                entries.append(FileInfo(path=ap, is_dir=True, size=0, modified_at=""))
            else:
                fk = (session_id, f"{wr}/{name}" if wr else name)
                content = self._files.get(fk, "")
                entries.append(
                    FileInfo(
                        path=ap,
                        is_dir=False,
                        size=len(content.encode("utf-8", errors="replace")),
                        modified_at="",
                    )
                )
        return LsResult(entries=entries)

    def read(
        self,
        session_id: str,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        key, err, _ = self._resolve_key(session_id, file_path)
        if err or key is None:
            return ReadResult(error=err or f"Error: '{file_path}' not found.")
        raw = self._files.get(key)
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
        base_canon = self._canon(path or "/")
        if self._is_deepx_agent_path(base_canon):
            base_store = self._deepx_store_prefix(base_canon)
            matches: list[GrepMatch] = []
            for sid, r in self._files.items():
                if sid != _DEEPX_BUCKET or not r.startswith(".deepx/"):
                    continue
                if base_store == ".deepx":
                    rel_from_base = r[len(".deepx/") :]
                else:
                    if r == base_store:
                        continue
                    if not r.startswith(base_store + "/"):
                        continue
                    rel_from_base = r[len(base_store) + 1 :]
                if glob and rel_from_base and not (
                    wc_fnmatch.fnmatch(rel_from_base, glob, flags=_GLOB_FLAGS)
                    or wc_fnmatch.fnmatch(
                        rel_from_base.rsplit("/", 1)[-1], glob, flags=_GLOB_FLAGS
                    )
                ):
                    continue
                ap = "/.deepx/" + r[len(".deepx/") :]
                content = self._files[(sid, r)]
                for i, line in enumerate(content.splitlines(), start=1):
                    if pattern in line:
                        matches.append(GrepMatch(path=ap, line_number=i, line=line))
            return GrepResult(matches=matches)

        kres = self._resolve_key(session_id, path or "/")
        key, err = kres[0], kres[1]
        if err or key is None:
            return GrepResult(error=err or f"Error: path '{path}' not found.")
        base_rel = key[1]
        matches = []
        for (sid, r), content in self._files.items():
            if sid != session_id:
                continue
            if r == ".deepx" or r.startswith(".deepx/"):
                continue
            if base_rel and not (r == base_rel or r.startswith(base_rel + "/")):
                continue
            ap = "/" + r
            if not base_rel:
                rel_from_base = r
            elif r == base_rel:
                rel_from_base = r.rsplit("/", 1)[-1]
            else:
                rel_from_base = r[len(base_rel) + 1 :]
            if glob and not (
                wc_fnmatch.fnmatch(rel_from_base, glob, flags=_GLOB_FLAGS)
                or wc_fnmatch.fnmatch(
                    rel_from_base.rsplit("/", 1)[-1], glob, flags=_GLOB_FLAGS
                )
            ):
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=ap, line_number=i, line=line))
        return GrepResult(matches=matches)

    def glob(self, session_id: str, pattern: str, path: str = "/") -> GlobResult:
        base_canon = self._canon(path)
        if self._is_deepx_agent_path(base_canon):
            base_store = self._deepx_store_prefix(base_canon)
            files: list[FileInfo] = []
            for sid, r in self._files.items():
                if sid != _DEEPX_BUCKET or not r.startswith(".deepx/"):
                    continue
                if base_store == ".deepx":
                    rel_from_base = r[len(".deepx/") :]
                else:
                    if r == base_store:
                        continue
                    if not r.startswith(base_store + "/"):
                        continue
                    rel_from_base = r[len(base_store) + 1 :]
                if not rel_from_base or "/" in rel_from_base:
                    continue
                if not _glob_match(rel_from_base, pattern):
                    continue
                content = self._files[(sid, r)]
                ap = "/.deepx/" + r[len(".deepx/") :]
                enc = len(content.encode("utf-8", errors="replace"))
                files.append(FileInfo(path=ap, is_dir=False, size=enc))
            files.sort(key=lambda x: x.path)
            return GlobResult(files=files)

        key, err, _ = self._resolve_key(session_id, path)
        if err or key is None:
            return GlobResult(error=err or f"Error: path '{path}' not found.")
        base_rel = key[1]
        files: list[FileInfo] = []
        for (sid, r), content in self._files.items():
            if sid != session_id:
                continue
            if r == ".deepx" or r.startswith(".deepx/"):
                continue
            if base_rel and not (r == base_rel or r.startswith(base_rel + "/")):
                continue
            if not base_rel:
                rel_part = r
            elif r == base_rel:
                rel_part = r.rsplit("/", 1)[-1]
            else:
                rel_part = r[len(base_rel) + 1 :]
            if not rel_part:
                continue
            if not _glob_match(rel_part, pattern):
                continue
            enc = len(content.encode("utf-8", errors="replace"))
            files.append(FileInfo(path="/" + r, is_dir=False, size=enc))
        files.sort(key=lambda x: x.path)
        return GlobResult(files=files)

    def write(self, session_id: str, file_path: str, content: str) -> WriteResult:
        key, err, canon = self._resolve_key(session_id, file_path)
        if err or key is None:
            return WriteResult(error=err or "Cannot write: invalid path.")
        allow_replace = self._is_deepx_agent_path(canon) or canon.startswith(
            OUTPUTS_PREFIX + "/"
        )
        if key in self._files and not allow_replace:
            return WriteResult(
                error=(
                    f"Cannot write to {file_path} because it already exists. "
                    "Use edit_file to modify it, or choose a different path."
                )
            )
        self._files[key] = content
        return WriteResult(path=canon, files_update=None)

    def edit(
        self,
        session_id: str,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        key, err, _ = self._resolve_key(session_id, file_path)
        if err or key is None:
            return EditResult(error=err or f"Error: '{file_path}' not found.")
        content = self._files.get(key)
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
        self._files[key] = new_content
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
