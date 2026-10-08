# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Host-side diagnostics for ``slm doctor``: editor plugins and install method.

Two findings were wrong before 4.1.22:

* "no editor plugin detected" on a machine with both plugins installed. The
  probe looked two directory levels deep, but Claude Code keeps marketplace
  plugins at ``plugins/cache/<marketplace>/<plugin>/<version>/`` and records
  them in ``plugins/installed_plugins.json``; it ignored ``CLAUDE_CONFIG_DIR``;
  and it never recognised the Codex plugin, whose manifest lives in
  ``.codex-plugin/`` under the name ``superlocalmemory-codex``.
* a PEP 668 warning about the system Python while SLM ran from a virtual
  environment. A venv shares its base interpreter's stdlib path, so the
  ``EXTERNALLY-MANAGED`` marker is visible from inside it, but PEP 668 does
  not apply to a venv.

Everything here reads local files only and never writes. A found manifest
proves the plugin content is on disk, not that the host runs its hooks.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PLUGIN_NAMES = frozenset({"superlocalmemory", "superlocalmemory-codex"})
_MANIFEST_DIRS = (".claude-plugin", ".codex-plugin")
_MAX_SCAN_DEPTH = 5  # plugins/cache/<marketplace>/<plugin>/<version>/<dir>


def _version_key(version: str) -> tuple:
    parts = []
    for piece in str(version).replace("-", ".").split("."):
        parts.append((0, int(piece), "") if piece.isdigit() else (1, 0, piece))
    return tuple(parts)


def claude_plugin_roots(home: Path | None = None) -> list[Path]:
    """Claude Code plugin folders: ``CLAUDE_CONFIG_DIR`` first, then ``~/.claude``."""
    home = home or Path.home()
    roots: list[Path] = []
    configured = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    if configured:
        roots.append(Path(configured).expanduser() / "plugins")
    roots.append(home / ".claude" / "plugins")
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _from_installed_registry(root: Path) -> dict[str, str] | None:
    """Claude Code's own record of installed plugins, or ``None`` if unreadable."""
    try:
        data = json.loads((root / "installed_plugins.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    plugins = data.get("plugins") if isinstance(data, dict) else None
    if not isinstance(plugins, dict):
        return None
    found: dict[str, str] = {}
    for key, entries in plugins.items():
        if str(key).split("@", 1)[0] not in PLUGIN_NAMES:
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            if isinstance(entry, dict) and entry.get("version"):
                found[f"claude:{key}"] = str(entry["version"])
    return found


def _manifests_under(root: Path, *, skip: frozenset[str] = frozenset()):
    """Plugin manifests up to the scan depth; hidden dirs other than manifests skipped."""
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base
        if "plugin.json" in filenames:
            yield Path(dirpath) / "plugin.json"
        if depth >= _MAX_SCAN_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if (d in _MANIFEST_DIRS or not d.startswith("."))
            and d != "node_modules" and not (depth == 0 and d in skip)
        ]


def _from_manifests(root: Path, *, skip: frozenset[str] = frozenset()) -> dict[str, str]:
    """Newest SLM manifest per plugin folder (a cache keeps old versions too)."""
    best: dict[str, str] = {}
    for manifest in _manifests_under(root, skip=skip):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # a sibling plugin's bad json is not ours
        name = str(data.get("name", "")) if isinstance(data, dict) else ""
        if name not in PLUGIN_NAMES:
            continue
        holder = manifest.parent
        if holder.name in _MANIFEST_DIRS:
            holder = holder.parent
        # .../<plugin>/<version>/ in a cache: label by the plugin folder.
        label = holder.parent.name if holder.name[:1].isdigit() else holder.name
        version = str(data.get("version", "unknown"))
        current = best.get(label)
        if current is None or _version_key(version) > _version_key(current):
            best[label] = version
    return best


def installed_plugin_versions(home: Path | None = None) -> dict[str, str]:
    """Version of each SLM editor plugin found on this machine, by install name."""
    home = home or Path.home()
    found: dict[str, str] = {}
    for root in claude_plugin_roots(home):
        if not root.is_dir():
            continue
        registry = _from_installed_registry(root)
        if registry is None:
            found.update(_from_manifests(root))
        else:
            # The registry is authoritative for marketplace installs; the
            # cache can still hold copies of uninstalled versions.
            found.update(registry)
            found.update(_from_manifests(root, skip=frozenset({"cache", "marketplaces"})))
    codex_root = home / ".codex" / "plugins"
    if codex_root.is_dir():
        found.update(_from_manifests(codex_root))
    vscode_root = home / ".vscode" / "extensions"
    if vscode_root.is_dir():  # many extensions: shallow look only
        for manifest in vscode_root.glob("*/plugin.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and str(data.get("name", "")) in PLUGIN_NAMES:
                found[manifest.parent.name] = str(data.get("version", "unknown"))
    return found


def running_in_virtualenv() -> bool:
    """True for venv, virtualenv, pipx and uv tool environments."""
    if getattr(sys, "real_prefix", None):  # legacy virtualenv
        return True
    return os.path.realpath(sys.prefix) != os.path.realpath(
        getattr(sys, "base_prefix", sys.prefix)
    )


def install_method_finding() -> tuple[str, str, str]:
    """``(status, detail, fix)`` for the PEP 668 / install-method doctor row."""
    import sysconfig

    if running_in_virtualenv():
        return (
            "PASS",
            f"running from a virtual environment ({sys.prefix}); PEP 668 does not apply",
            "",
        )
    stdlib = sysconfig.get_path("stdlib")
    if stdlib and (Path(stdlib) / "EXTERNALLY-MANAGED").exists():
        return (
            "WARN",
            f"SLM runs on the system Python ({sys.executable}), which is externally "
            "managed (EXTERNALLY-MANAGED marker found); pip install may fail with a "
            "PEP 668 error.",
            "Use an isolated install: pipx install superlocalmemory  "
            "or uv tool install superlocalmemory",
        )
    return ("PASS", "No EXTERNALLY-MANAGED marker — standard pip install supported", "")


def sqlite_extension_finding() -> tuple[str, str, str]:
    """``(status, detail, fix)``: can THIS Python load sqlite-vec?

    Some Pythons (the macOS system one, some managed/enterprise builds) are
    compiled without SQLite extension loading. SLM then cannot load sqlite-vec,
    the vector channel turns itself off with only a debug log line, and recall
    quietly runs on keyword and graph channels alone. That must be loud.
    """
    import sqlite3

    where = f"{sys.executable} (SQLite {sqlite3.sqlite_version})"
    conn = sqlite3.connect(":memory:")
    try:
        if not hasattr(conn, "enable_load_extension"):
            return (
                "FAIL",
                f"VECTOR SEARCH IS OFF: this Python cannot load SQLite extensions; "
                f"{where} was built without them, so sqlite-vec never loads and "
                "recall falls back to keyword and graph channels only.",
                "Run SLM on a Python built with extension loading (Homebrew, "
                "python.org or uv-managed), e.g. uv tool install --python 3.12 "
                "superlocalmemory, then slm restart",
            )
        try:
            import sqlite_vec
        except ImportError:
            return (
                "WARN",
                f"{where} can load SQLite extensions but sqlite-vec is not installed; "
                "vector search is off.",
                "slm doctor --fix   (or: pip install sqlite-vec)",
            )
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            version = conn.execute("select vec_version()").fetchone()[0]
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            return (
                "FAIL",
                f"VECTOR SEARCH IS OFF: sqlite-vec failed to load into {where}: "
                f"{type(exc).__name__}: {exc}",
                "Reinstall sqlite-vec for this Python, then slm restart",
            )
        return ("PASS", f"sqlite-vec {version} loads in {where}", "")
    finally:
        conn.close()
