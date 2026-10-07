# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""#140 — the venv bootstrap must not fail a session over a venv nothing uses.

SessionStart runs ensure-venv.sh on every session. It used to demand Python
3.12+ unconditionally, so a machine with SLM installed (pipx, pip, npm) and an
older python3 first on PATH got a failing SessionStart hook for a venv the
launcher would never touch. The bootstrap now asks the same question the
launcher asks, through the same code (slm-resolve.sh), and only builds the venv
when the answer is "the plugin venv".

Each case RUNS the script in a sandbox: own HOME, own data dirs, and a PATH of
stubs plus /usr/bin:/bin — so the developer's own slm cannot decide the result.
"""

from __future__ import annotations

import platform

import pytest

from tests.test_plugin._plugin_sandbox import (
    REPO,
    SRC_SCRIPTS,
    requirements_sentinel_digest,
    run,
    sandbox_env,
    stub_python,
    stub_slm,
    system_path_has_slm,
    write_exe,
)

pytestmark = [
    pytest.mark.skipif(platform.system() == "Windows", reason="POSIX bash script"),
    pytest.mark.skipif(system_path_has_slm(), reason="an slm in /usr/bin or /bin defeats the sandbox"),
]

ENSURE = SRC_SCRIPTS / "ensure-venv.sh"
PLUGIN_SRC = REPO / "plugin-src"


def _env(tmp_path, stubs, **extra):
    return sandbox_env(tmp_path, stubs, CLAUDE_PLUGIN_ROOT=str(PLUGIN_SRC), **extra)


def _venv_dir(tmp_path):
    return tmp_path / "plugin-data" / "venv"


@pytest.mark.parametrize("launcher", [None, "auto", "system"])
def test_old_python_is_fine_when_the_installed_slm_will_be_used(tmp_path, launcher):
    stubs = tmp_path / "stubs"
    stub_python(stubs, "3.11.9")
    stub_slm(stubs / "slm", "system")

    result = run(["bash", str(ENSURE)], _env(tmp_path, stubs, SLM_LAUNCHER=launcher))

    assert result.returncode == 0, result.stderr
    assert not _venv_dir(tmp_path).exists(), "built a venv nothing will run"
    assert "not needed" in result.stderr, result.stderr
    assert "requires Python" not in result.stderr


def test_old_python_is_fine_when_an_explicit_binary_is_named(tmp_path):
    stubs = tmp_path / "stubs"
    stub_python(stubs, "3.11.9")
    explicit = stub_slm(tmp_path / "elsewhere" / "slm", "explicit")

    result = run(["bash", str(ENSURE)], _env(tmp_path, stubs, SLM_LAUNCHER=str(explicit)))

    assert result.returncode == 0, result.stderr
    assert not _venv_dir(tmp_path).exists()


def test_explicit_tilde_path_is_expanded_like_the_launcher(tmp_path):
    stubs = tmp_path / "stubs"
    stub_python(stubs, "3.11.9")
    stub_slm(tmp_path / "home" / "tools" / "slm", "explicit")

    result = run(["bash", str(ENSURE)], _env(tmp_path, stubs, SLM_LAUNCHER="~/tools/slm"))

    assert result.returncode == 0, result.stderr
    assert not _venv_dir(tmp_path).exists()


def test_old_python_still_fails_when_the_venv_is_the_only_slm(tmp_path):
    """auto with nothing installed: the venv IS going to be used, so the guard stands."""
    stubs = tmp_path / "stubs"
    stub_python(stubs, "3.11.9")

    result = run(["bash", str(ENSURE)], _env(tmp_path, stubs))

    assert result.returncode != 0
    assert "requires Python >= 3.12" in result.stderr


def test_old_python_still_fails_when_the_plugin_venv_is_forced(tmp_path):
    """SLM_LAUNCHER=plugin: an installed slm on PATH does not excuse the guard."""
    stubs = tmp_path / "stubs"
    stub_python(stubs, "3.11.9")
    stub_slm(stubs / "slm", "system")

    result = run(["bash", str(ENSURE)], _env(tmp_path, stubs, SLM_LAUNCHER="plugin"))

    assert result.returncode != 0
    assert "requires Python >= 3.12" in result.stderr


def test_new_python_passes_the_guard_when_the_venv_is_used(tmp_path):
    """3.12 passes the guard and reaches the venv logic (fast path, no network)."""
    stubs = tmp_path / "stubs"
    stub_python(stubs, "3.12.4")
    venv = _venv_dir(tmp_path)
    write_exe(venv / "bin" / "python3", "#!/bin/sh\nexit 0\n")
    digest = requirements_sentinel_digest(PLUGIN_SRC)
    (tmp_path / "plugin-data" / ".venv-reqs.sha256").write_text(digest + "\n", encoding="utf-8")

    result = run(["bash", str(ENSURE)], _env(tmp_path, stubs))

    assert result.returncode == 0, result.stderr
    assert "venv up-to-date" in result.stderr


def test_a_misconfigured_launcher_is_not_hidden(tmp_path):
    """SLM_LAUNCHER=system with no slm: the launcher will fail, so say so —
    but a venv cannot fix it, so building one would be wrong, and failing the
    session over the venv would blame the wrong thing."""
    stubs = tmp_path / "stubs"
    stub_python(stubs, "3.12.4")

    result = run(["bash", str(ENSURE)], _env(tmp_path, stubs, SLM_LAUNCHER="system"))

    assert result.returncode == 0, result.stderr
    assert "SLM_LAUNCHER=system but no 'slm' on PATH." in result.stderr
    assert not _venv_dir(tmp_path).exists()


def test_required_env_is_still_checked_first(tmp_path):
    stubs = tmp_path / "stubs"
    stub_slm(stubs / "slm", "system")

    result = run(["bash", str(ENSURE)], _env(tmp_path, stubs, CLAUDE_PLUGIN_DATA=None))

    assert result.returncode != 0
    assert "CLAUDE_PLUGIN_DATA must be set" in result.stderr
