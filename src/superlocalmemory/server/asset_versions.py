# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Derive the dashboard's asset cache-busters from the files themselves.

``index.html`` referenced 64 static assets. 35 carried a hand-written
``?v=`` literal and 29 carried nothing, and not one of them was derived from the
file it pointed at. So the version strings looked like cache-busting and were
not: editing a JS or CSS file left its ``?v=`` reading whatever the last person
typed, which during 4.0.10 was ``022ff653`` on a file that had changed.

WHAT THIS DOES AND DOES NOT FIX
-------------------------------
It does not fix a live user-facing bug, and it would be dishonest to claim it
does. Three mechanisms already stop a stale asset reaching a browser on this
server, and the first was verified against the running daemon rather than read:

  * ``/static/*`` is served ``Cache-Control: no-cache, must-revalidate`` with an
    ETag (``server/security_middleware.py``), so a browser must revalidate and
    cannot serve a cached copy without asking.
  * The unified daemon copies the whole UI tree into the data directory on every
    start (``unified_daemon.py``, ``state_path("ui")``), so an upgrade refreshes
    the files it serves.
  * ``index.html`` itself is ``no-cache``, so the page is always re-read.

What it fixes is a **trap**, and unblocks a real improvement:

  * 64 references, none tracking content. Anyone reading them concludes
    cache-busting is handled here, which is how the 4.0.10 change shipped with a
    stale literal and how the next one would too. A number that is maintained by
    hand and consulted by nobody is worse than no number.
  * The revalidation policy is the only thing making that safe, and it costs a
    conditional request per asset on every page load — 64 of them. The obvious
    optimisation is ``max-age`` with a long life, and today that change would
    turn every hand-typed literal into an immediate live bug. With versions
    derived from content it becomes safe to make. That policy change is NOT
    made here; it needs its own measurement.
  * A proxy or CDN that ignores ``no-cache`` is defeated by a URL that changes,
    not by a header.

DESIGN
------
Rewrite at serve time rather than at build time, because there is no build step:
the UI ships as source files and is copied into place. The hash is computed from
file bytes and cached on ``(size, mtime_ns)``, so a warm daemon does one ``stat``
per asset per page load and no reads. Unresolvable references keep whatever the
HTML said, so a missing file or an odd path degrades to today's behaviour rather
than breaking the page.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["render_index", "asset_version", "rewrite_asset_versions"]

#: Characters of hex digest used in a URL. Eight is what the existing literals
#: used and is ample: these identify one file's revisions, not a global
#: namespace, so a collision needs two versions of the same file agreeing on
#: eight hex characters.
_HASH_CHARS = 8

#: ``src="static/…"`` / ``href="static/…"`` with an optional existing ``?v=``.
#: Deliberately narrow — only the ``static/`` prefix the dashboard mounts, only
#: double-quoted attributes, and the path is captured without its query so the
#: rewrite cannot alter it.
_ASSET_REF = re.compile(
    r'(?P<attr>\b(?:src|href)=")'
    r'(?P<path>static/[^"?#]+)'
    r'(?P<query>\?[^"#]*)?'
    r'(?P<fragment>#[^"]*)?'
    r'(?P<close>")'
)

#: (resolved path) -> (size, mtime_ns, digest). Keyed on the path so a daemon
#: serving from the data-directory copy and one serving from the source tree do
#: not share entries.
_CACHE: dict[Path, tuple[int, int, str]] = {}


def asset_version(asset_path: Path) -> str | None:
    """Short content hash of ``asset_path``, or None if it cannot be read.

    Cached on ``(size, mtime_ns)``. That pair is what ETag generators use for
    the same reason: it changes on every practical edit, and re-reading a file
    that has not changed costs a page-load's worth of I/O for nothing.

    ``mtime_ns`` rather than ``mtime``: the UI is installed with
    ``shutil.copytree``, which preserves timestamps, so two files written inside
    the same filesystem tick are a real possibility on a fast copy.
    """
    try:
        stat = asset_path.stat()
    except OSError:
        return None

    key = (stat.st_size, stat.st_mtime_ns)
    cached = _CACHE.get(asset_path)
    if cached is not None and cached[:2] == key:
        return cached[2]

    try:
        digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()[:_HASH_CHARS]
    except OSError as exc:
        logger.debug("asset version unavailable for %s: %s", asset_path, exc)
        return None

    _CACHE[asset_path] = (*key, digest)
    return digest


def rewrite_asset_versions(html: str, ui_root: Path) -> str:
    """Replace every ``static/…?v=`` with a version derived from the file.

    ``ui_root`` is the directory mounted at ``/static``, so a reference to
    ``static/js/core.js`` resolves to ``ui_root/js/core.js`` — the ``static/``
    segment is the mount point, not a directory on disk. Getting that wrong
    silently resolves nothing and leaves all 64 literals in place, which is why
    the test asserts a version actually moved rather than only that the call
    returned.

    Assets with no existing query gain one. That is a URL change, and it is the
    point: 29 of the 64 references had no cache-buster at all, so they were the
    ones a policy change would break first.
    """

    def _replace(match: re.Match[str]) -> str:
        path = match.group("path")
        version = asset_version(ui_root / path[len("static/"):])
        if version is None:
            # Keep whatever the HTML said. A reference we cannot resolve is not
            # a reason to serve a page that cannot load its own stylesheet.
            return match.group(0)
        return (
            f"{match.group('attr')}{path}?v={version}"
            f"{match.group('fragment') or ''}{match.group('close')}"
        )

    return _ASSET_REF.sub(_replace, html)


def render_index(
    index_path: Path,
    ui_root: Path | None = None,
    *,
    substitutions: dict[str, str] | None = None,
) -> str:
    """Read ``index.html`` and prepare it for serving.

    One function for the three ``root()`` handlers (``api.py``, ``ui.py``,
    ``unified_daemon.py``) that each read this file and returned it. They had
    drifted: only the daemon substituted ``__SLM_VERSION__``, so the upgrade
    detector the dashboard relies on silently did nothing on the other two.

    ``ui_root`` defaults to the file's own directory, which is correct for every
    caller today — ``index.html`` sits at the root of the tree mounted at
    ``/static``.

    Raises ``OSError`` if the index itself cannot be read; every caller already
    checks ``exists()`` and has its own fallback page.
    """
    html = index_path.read_text(encoding="utf-8")
    html = rewrite_asset_versions(html, ui_root or index_path.parent)
    for placeholder, value in (substitutions or {}).items():
        html = html.replace(placeholder, value)
    return html
