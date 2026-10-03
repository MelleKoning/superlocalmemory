# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""pyproject.toml, bin/slm, bin/slm.bat, and scripts/postinstall.js all
require Python 3.12+ (the patched-cryptography floor for V4). Three other
places still accepted 3.11: `slm doctor`'s Python check, the setup wizard's
System Check step, and the dashboard's component-health probe. A user on
3.11 got a green PASS from doctor and the wizard right up until something
3.12-only broke underneath them (L3-17).
"""

from __future__ import annotations

import io
from argparse import Namespace
from collections import namedtuple
from unittest import mock

import pytest

_FakeVersion = namedtuple("version_info", ["major", "minor", "micro", "releaselevel", "serial"])


def _fake_version(major: int, minor: int, micro: int = 0) -> _FakeVersion:
    return _FakeVersion(major, minor, micro, "final", 0)


# ---------------------------------------------------------------------------
# slm doctor (src/superlocalmemory/cli/commands.py::cmd_doctor)
# ---------------------------------------------------------------------------

def _run_doctor(monkeypatch, version) -> str:
    import sys

    from superlocalmemory.cli.commands import cmd_doctor

    monkeypatch.setattr(sys, "version_info", version)
    args = Namespace(json=False, quick=True, fix=False)
    captured = io.StringIO()
    with mock.patch(
        "superlocalmemory.core.install_detector._detect_all_installs", return_value=[],
    ), mock.patch(
        "superlocalmemory.cli.commands._detect_all_installs", return_value=[],
    ), mock.patch("sys.stdout", captured):
        try:
            cmd_doctor(args)
        except SystemExit:
            pass
    return captured.getvalue()


def test_doctor_fails_python_3_11(monkeypatch):
    output = _run_doctor(monkeypatch, _fake_version(3, 11, 9))
    assert "[FAIL] Python" in output, output
    assert "3.12" in output


def test_doctor_passes_python_3_12(monkeypatch):
    output = _run_doctor(monkeypatch, _fake_version(3, 12, 0))
    assert "[PASS] Python" in output, output


# ---------------------------------------------------------------------------
# setup wizard (src/superlocalmemory/cli/setup_wizard.py::run_wizard)
# ---------------------------------------------------------------------------

def _stub_sentence_transformers(monkeypatch):
    """The wizard's System Check step probes for sentence-transformers before
    it checks the version gate. The real package transitively imports torch,
    which is unrelated to this fix and, on some hosts/toolchains, raises
    (not ImportError) well before the version gate is ever reached. Stub it
    so these tests exercise the version gate in isolation."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.ModuleType("sentence_transformers"))


class _PastTheGateSentinel(Exception):
    """Raised by a Step-2 probe stub so the test can confirm the wizard got
    past the Step-1 version gate, without running the rest of the (slow,
    network-touching, model-downloading) wizard flow."""


def test_setup_wizard_rejects_python_3_11(monkeypatch, capsys):
    """Stub `_ollama_available` to raise instead of making a real network
    call: if the version gate wrongly accepts 3.11 (the bug), the wizard
    proceeds into Step 2 and the stub raises immediately — a fast, clean
    failure instead of a test that hangs on a real probe."""
    import sys

    import superlocalmemory.cli.setup_wizard as sw

    _stub_sentence_transformers(monkeypatch)
    monkeypatch.setattr(sys, "version_info", _fake_version(3, 11, 9))

    def _boom():
        raise _PastTheGateSentinel("wizard proceeded past the version gate on Python 3.11")

    monkeypatch.setattr(sw, "_ollama_available", _boom)

    sw.run_wizard(auto=True)  # must return before reaching the stub above

    output = capsys.readouterr().out
    assert "3.12" in output
    assert "Step 2" not in output, "wizard must stop at the system check, not proceed"


def test_setup_wizard_accepts_python_3_12(monkeypatch, capsys):
    import sys

    import superlocalmemory.cli.setup_wizard as sw

    _stub_sentence_transformers(monkeypatch)
    monkeypatch.setattr(sys, "version_info", _fake_version(3, 12, 0))
    monkeypatch.setattr(sw, "_get_ram_gb", lambda: 0)

    def _boom():
        raise _PastTheGateSentinel

    monkeypatch.setattr(sw, "_ollama_available", _boom)

    with pytest.raises(_PastTheGateSentinel):
        sw.run_wizard(auto=True)

    output = capsys.readouterr().out
    assert "✗ (3.12+ required)" not in output
    assert "Python 3.12+ is required" not in output


# ---------------------------------------------------------------------------
# dashboard health probe (src/superlocalmemory/core/component_registry.py)
# ---------------------------------------------------------------------------

def test_probe_python_reports_missing_below_3_12(monkeypatch):
    import sys

    from superlocalmemory.core.component_registry import STATUS_MISSING, probe_python

    monkeypatch.setattr(sys, "version_info", _fake_version(3, 11, 9))
    component = probe_python()
    assert component.status == STATUS_MISSING
    assert "3.12" in component.detail
    assert "3.12" in component.fix_cmd


def test_probe_python_reports_ok_at_3_12(monkeypatch):
    import sys

    from superlocalmemory.core.component_registry import STATUS_OK, probe_python

    monkeypatch.setattr(sys, "version_info", _fake_version(3, 12, 0))
    component = probe_python()
    assert component.status == STATUS_OK
