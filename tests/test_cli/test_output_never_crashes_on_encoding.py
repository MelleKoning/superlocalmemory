# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""No ``slm`` command can crash because of the characters it prints.

On Windows, when output goes to a pipe or a file (an installer, an editor, a
script, ``slm ... > out.txt``), Python encodes it in the code page, usually
cp1252. SLM prints characters cp1252 lacks (✓ ✗ → …), so such a command died
with ``UnicodeEncodeError`` after doing its work. Here a cp1252 stdout and
stderr are forced with PYTHONIOENCODING, which is how Python behaves there.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

from superlocalmemory.cli import stdio_safety

SRC = Path(__file__).resolve().parents[2] / "src"


def _run_cp1252(code: str, tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0",
           "PYTHONPATH": str(SRC), "SLM_DATA_DIR": str(tmp_path / "data"),
           "HOME": str(tmp_path / "home"), "USERPROFILE": str(tmp_path / "home")}
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, timeout=120, check=False)


def test_the_cli_entry_point_makes_both_streams_safe(tmp_path):
    """Through the real ``slm`` entry: a handler that prints ✓ to stdout and
    → to stderr finishes, and the output stays cp1252 (what the reader of
    that pipe expects), with ``?`` where a character has no cp1252 form."""
    code = (
        "import sys\n"
        "from superlocalmemory.cli import commands, daemon, setup_wizard\n"
        "from superlocalmemory.cli import main as m\n"
        "def show(_args):\n"
        "    print('  \\u2713 Daemon running')\n"
        "    print('step \\u2192 done \\u00e9', file=sys.stderr)\n"
        "commands.dispatch = show\n"                        # the handler, not a daemon
        "daemon.ensure_daemon = lambda **_k: True\n"
        "setup_wizard.check_first_use = lambda _c: None\n"
        "sys.argv = ['slm', 'status']\n"
        "m.main()\n"
    )
    done = _run_cp1252(code, tmp_path)
    assert done.returncode == 0, done.stderr.decode("cp1252", "replace")
    assert done.stdout.decode("cp1252").endswith("  ? Daemon running" + os.linesep)
    assert "step ? done é" in done.stderr.decode("cp1252")


def test_a_stream_that_can_carry_everything_is_left_alone():
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    stdio_safety.make_unencodable_output_safe(stream)
    assert stream.errors == "strict"


def test_a_cp1252_stream_replaces_instead_of_raising():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    stdio_safety.make_unencodable_output_safe(stream)
    stream.write("✓ ok\n")
    stream.flush()
    assert stream.encoding == "cp1252"
    assert raw.getvalue() == b"? ok\n"


def test_a_stream_that_cannot_be_reconfigured_is_left_as_it_is():
    class _Plain:
        encoding = "cp1252"

    stdio_safety.make_unencodable_output_safe(_Plain())  # no reconfigure: no error
    stdio_safety.make_unencodable_output_safe(None)       # pythonw: no stream at all
