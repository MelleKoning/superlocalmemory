# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Keep details of the SLM computer out of tool answers to remote callers.

A tool answer for a caller on this computer is unchanged. For a remote caller,
before the answer leaves the daemon:

* fields that describe the host (``base_dir``, ``db_path``, ``pid``, ``cwd``,
  ``env``, anything ending in ``_path``/``_dir``/``_root``/``_file`` ...) become
  :data:`REDACTED`;
* in every other string except memory text, absolute paths become
  :data:`HOST_PATH`, and the home directory, the SLM data folder and the
  account name are replaced too.

Memory text (``content`` and the like) is returned exactly as stored: it is the
user's data, and a path the user wrote down is not a host detail.
"""

from __future__ import annotations

import getpass
import json
import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

REDACTED = "[host detail withheld]"
HOST_PATH = "[host path]"

#: Field names that always describe the host, matched case-insensitively.
_HOST_KEYS = frozenset({
    "base_dir", "db_path", "data_dir", "data_root", "path", "paths", "file", "files",
    "filename", "cwd", "home", "home_dir", "pid", "ppid", "env", "environ",
    "environment", "executable", "python", "python_path", "sys_path", "hostname",
    "username", "user_name", "os_user", "socket", "log_file", "log_dir", "logs_dir",
    "install_dir", "repo_path", "project_path", "config_path", "argv", "cmdline",
})
_HOST_SUFFIXES = ("_path", "_dir", "_root", "_file", "_paths", "_dirs")
#: Fields that carry memory text, returned as written.
_CONTENT_KEYS = frozenset({
    "content", "text", "original_content", "fact", "memory", "summary_text", "query",
    "answer", "snippet", "excerpt", "observation", "title", "body", "note",
})

_POSIX_PATH = re.compile(r"(?<![\w.~:/-])/(?:[^\s/\"'`<>|:;,()\[\]{}]+/)+[^\s/\"'`<>|:;,()\[\]{}]*")
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\s\\\"'<>|:;,]+\\)*[^\s\\\"'<>|:;,]*")
_TILDE_PATH = re.compile(r"(?<![\w])~/(?:[^\s\"'`<>|:;,()\[\]{}]+)")


@lru_cache(maxsize=1)
def _host_strings() -> tuple[str, ...]:
    values: list[str] = []
    try:
        values.append(str(Path.home()))
    except Exception:  # noqa: BLE001
        pass
    try:
        from superlocalmemory.infra.data_root import canonical_data_root

        values.append(str(canonical_data_root()))
    except Exception:  # noqa: BLE001
        pass
    try:
        user = getpass.getuser()
        if len(user) >= 3:
            values.append(user)
    except Exception:  # noqa: BLE001
        pass
    for name in ("HOSTNAME", "COMPUTERNAME"):
        if len(os.environ.get(name, "")) >= 3:
            values.append(os.environ[name])
    # Longest first so a data root inside home is replaced whole.
    return tuple(sorted({v for v in values if v}, key=len, reverse=True))


def clear_cache() -> None:
    _host_strings.cache_clear()


def _is_host_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _HOST_KEYS or lowered.endswith(_HOST_SUFFIXES)


def redact_text(text: str) -> str:
    """Host details in a non-memory string."""
    out = _WINDOWS_PATH.sub(HOST_PATH, text)
    out = _POSIX_PATH.sub(HOST_PATH, out)
    out = _TILDE_PATH.sub(HOST_PATH, out)
    for value in _host_strings():
        out = out.replace(value, HOST_PATH if os.sep in value else REDACTED)
    return out


def redact_value(value: Any, key: str | None = None) -> Any:
    """A redacted copy (inputs are never mutated)."""
    if key is not None and key.lower() in _CONTENT_KEYS:
        return value
    if key is not None and _is_host_key(key):
        return REDACTED if value not in (None, "", [], {}) else value
    if isinstance(value, dict):
        return {k: redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, key) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _redact_block_text(text: str) -> str:
    try:
        parsed = json.loads(text)
    except ValueError:
        return redact_text(text)
    if isinstance(parsed, (dict, list)):
        return json.dumps(redact_value(parsed), indent=2, default=str)
    return redact_text(text)


def redact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """A ``tools/call`` result with host details removed."""
    out = dict(result)
    blocks: Iterable[Any] = result.get("content") or []
    out["content"] = [
        dict(b, text=_redact_block_text(b["text"]))
        if isinstance(b, dict) and isinstance(b.get("text"), str) else b
        for b in blocks
    ]
    if isinstance(result.get("structuredContent"), (dict, list)):
        out["structuredContent"] = redact_value(result["structuredContent"])
    return out


__all__ = ["HOST_PATH", "REDACTED", "clear_cache", "redact_text", "redact_tool_result",
           "redact_value"]
