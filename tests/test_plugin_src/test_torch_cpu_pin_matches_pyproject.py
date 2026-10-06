# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""GB4 — the CPU-only torch pin must never drift from pyproject.toml.

plugin-src/requirements-cpu-torch.txt names the exact torch version
ensure-venv.sh (and scripts/postinstall.js) pre-install from the official CPU
wheel index before the rest of requirements.txt resolves. It is a second copy
of pyproject.toml's `torch==` pin by necessity (the pre-install has to name an
exact version to request from a *different* index), so this test is the one
thing standing between a routine torch bump in pyproject.toml and a silently
stale CPU pin that would start failing the "already satisfied" trick quietly
on the next release.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).parent.parent.parent


def _torch_pins(text: str) -> set[str]:
    return set(re.findall(r'"torch==([0-9][^"]*)"', text))


def test_requirements_cpu_torch_pin_matches_pyproject():
    pyproject_text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    pyproject_pins = _torch_pins(pyproject_text)
    assert pyproject_pins, "expected at least one torch== pin in pyproject.toml"
    assert len(pyproject_pins) == 1, (
        f"pyproject.toml pins more than one distinct torch version: {pyproject_pins} "
        "— this test assumes core deps and extras always agree"
    )
    (expected_version,) = pyproject_pins

    cpu_torch_text = (REPO / "plugin-src" / "requirements-cpu-torch.txt").read_text(encoding="utf-8")
    match = re.search(r"^torch==(\S+)$", cpu_torch_text, re.MULTILINE)
    assert match, "plugin-src/requirements-cpu-torch.txt has no torch== line"
    assert match.group(1) == expected_version, (
        f"plugin-src/requirements-cpu-torch.txt pins torch=={match.group(1)} but "
        f"pyproject.toml pins torch=={expected_version} — bump both together"
    )


def test_built_plugin_copy_matches_too():
    """The generated plugin/ (and copilot-plugin/, via the same copy loop) must
    carry the identical pin — catches a stale build as well as a stale source."""
    source = (REPO / "plugin-src" / "requirements-cpu-torch.txt").read_text(encoding="utf-8")
    built = REPO / "plugin" / "requirements-cpu-torch.txt"
    assert built.is_file(), "run `node scripts/build-plugin.mjs` to regenerate plugin/"
    assert built.read_text(encoding="utf-8") == source
