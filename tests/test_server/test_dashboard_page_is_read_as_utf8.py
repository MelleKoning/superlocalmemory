# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The dashboard page loads the same whatever the computer's language setting.

``index.html`` contains characters outside the Windows code page cp1252. It was
read with the locale's encoding, so on Windows the dashboard failed with
``UnicodeDecodeError`` and on other single-byte locales it was served garbled.

Windows cannot run here, so this runs the real ``render_index`` in a child
Python under a single-byte locale (ISO-8859-1), which takes the same code path
as cp1252: the locale's encoding is used whenever none is named.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import superlocalmemory

INDEX = Path(superlocalmemory.__file__).parent / "ui" / "index.html"
LOCALE = "en_US.ISO8859-1"


def _locale_available() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import locale; print(locale.getpreferredencoding(False))"],
        env={**os.environ, "LC_ALL": LOCALE, "PYTHONUTF8": "0"},
        capture_output=True, text=True, check=False,
    )
    return probe.stdout.strip().upper().replace("-", "") == "ISO88591"


def test_the_page_really_contains_characters_a_single_byte_code_page_lacks():
    text = INDEX.read_text(encoding="utf-8")
    assert any(ord(ch) > 0xFF for ch in text)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX locales; Windows runs the guard test")
def test_the_page_is_served_intact_under_a_single_byte_locale():
    if not _locale_available():
        pytest.skip(f"the {LOCALE} locale is not installed on this machine")
    child = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from superlocalmemory.server.asset_versions import render_index\n"
        "html = render_index(Path(sys.argv[1]))\n"
        "sys.stdout.buffer.write(json.dumps(html).encode('ascii'))\n"
    )
    env = {**os.environ, "LC_ALL": LOCALE, "PYTHONUTF8": "0"}
    env.pop("PYTHONIOENCODING", None)
    done = subprocess.run(
        [sys.executable, "-c", child, str(INDEX)],
        env=env, capture_output=True, check=False, timeout=120,
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    served = json.loads(done.stdout)
    expected = INDEX.read_text(encoding="utf-8")
    wide = sorted({ch for ch in expected if ord(ch) > 0x7F})
    missing = [ch for ch in wide if ch not in served]
    assert not missing, f"characters lost when the page was read: {missing[:10]}"
