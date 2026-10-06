"""`slm doctor` host diagnostics (G08, 4.1.22) — local files only, fake HOME.

Three findings were wrong on a real machine:
* "no editor plugin detected" with both plugins installed: marketplace
  installs live four folders deep and are recorded in installed_plugins.json,
  CLAUDE_CONFIG_DIR was ignored, and the Codex plugin (.codex-plugin/,
  "superlocalmemory-codex") was never recognised;
* a PEP 668 warning about the system Python while SLM ran in a venv;
* nothing at all when the Python cannot load SQLite extensions, which turns
  vector search off silently.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from superlocalmemory.cli import doctor_hosts


def _manifest(folder: Path, name: str, version: str, kind: str = ".claude-plugin") -> None:
    (folder / kind).mkdir(parents=True, exist_ok=True)
    (folder / kind / "plugin.json").write_text(
        json.dumps({"name": name, "version": version}), encoding="utf-8",
    )


@pytest.fixture
def home(tmp_path, monkeypatch):
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setenv("HOME", str(fake))
    monkeypatch.setenv("USERPROFILE", str(fake))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return fake


def _claude_marketplace(root: Path, version: str) -> None:
    plugins = root / "plugins"
    install = plugins / "cache" / "qualixar" / "superlocalmemory" / version
    _manifest(install, "superlocalmemory", version)
    (plugins / "installed_plugins.json").write_text(json.dumps({
        "version": 2,
        "plugins": {"superlocalmemory@qualixar": [
            {"scope": "user", "installPath": str(install), "version": version},
        ], "other@elsewhere": [{"scope": "user", "installPath": "/x", "version": "9.9.9"}]},
    }), encoding="utf-8")


def test_a_claude_marketplace_install_is_detected(home):
    _claude_marketplace(home / ".claude", "4.1.21")
    found = doctor_hosts.installed_plugin_versions()
    assert found == {"claude:superlocalmemory@qualixar": "4.1.21"}


def test_claude_config_dir_is_honoured(home, tmp_path, monkeypatch):
    custom = tmp_path / "claude-desktop"
    _claude_marketplace(custom, "4.1.21")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
    assert "4.1.21" in doctor_hosts.installed_plugin_versions().values()


def test_an_uninstalled_copy_left_in_the_cache_is_not_reported(home):
    plugins = home / ".claude" / "plugins"
    stale = plugins / "cache" / "qualixar" / "superlocalmemory" / "4.0.1"
    _manifest(stale, "superlocalmemory", "4.0.1")
    registry = json.dumps({"version": 2, "plugins": {}})
    (plugins / "installed_plugins.json").write_text(registry, encoding="utf-8")
    assert doctor_hosts.installed_plugin_versions() == {}


def test_the_codex_plugin_is_detected_at_its_newest_cached_version(home):
    cache = home / ".codex" / "plugins" / "cache" / "qualixar" / "superlocalmemory-codex"
    _manifest(cache / "4.1.7", "superlocalmemory-codex", "4.1.7", ".codex-plugin")
    _manifest(cache / "4.1.21", "superlocalmemory-codex", "4.1.21", ".codex-plugin")
    assert doctor_hosts.installed_plugin_versions() == {"superlocalmemory-codex": "4.1.21"}


def test_another_vendors_deep_plugin_is_not_ours(home):
    cache = home / ".codex" / "plugins" / "cache" / "acme" / "acme-tools" / "1.0.0"
    _manifest(cache, "acme-tools", "1.0.0", ".codex-plugin")
    assert doctor_hosts.installed_plugin_versions() == {}


def test_a_venv_is_not_warned_about_pep_668(monkeypatch, tmp_path):
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    (stdlib / "EXTERNALLY-MANAGED").write_text("[externally-managed]\n", encoding="utf-8")
    monkeypatch.setattr("sysconfig.get_path", lambda name: str(stdlib))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    status, detail, _fix = doctor_hosts.install_method_finding()
    assert status == "PASS" and "virtual environment" in detail


def test_the_system_python_with_the_marker_is_still_warned(monkeypatch, tmp_path):
    stdlib = tmp_path / "stdlib"
    stdlib.mkdir()
    (stdlib / "EXTERNALLY-MANAGED").write_text("[externally-managed]\n", encoding="utf-8")
    monkeypatch.setattr("sysconfig.get_path", lambda name: str(stdlib))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "base"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.delattr(sys, "real_prefix", raising=False)
    status, detail, fix = doctor_hosts.install_method_finding()
    assert status == "WARN" and "PEP 668" in detail and "pipx" in fix


def test_a_python_that_cannot_load_sqlite_extensions_is_a_loud_failure(monkeypatch):
    class _NoExtensions:
        def close(self) -> None:
            pass

    fake_sqlite = types.SimpleNamespace(connect=lambda _p: _NoExtensions(), sqlite_version="3.43.2")
    monkeypatch.setitem(sys.modules, "sqlite3", fake_sqlite)
    status, detail, fix = doctor_hosts.sqlite_extension_finding()
    assert status == "FAIL"
    assert detail.startswith("VECTOR SEARCH IS OFF")
    assert "slm restart" in fix


def test_this_python_reports_its_sqlite_extension_state_truthfully():
    import sqlite3

    status, detail, _fix = doctor_hosts.sqlite_extension_finding()
    can_load = hasattr(sqlite3.connect(":memory:"), "enable_load_extension")
    if not can_load:
        assert status == "FAIL"
        return
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        assert status == "WARN"
        return
    assert status == "PASS" and "sqlite-vec" in detail


def test_the_doctor_entry_point_uses_these_probes(home):
    from superlocalmemory.cli.commands import _installed_plugin_versions

    _claude_marketplace(home / ".claude", "4.1.21")
    cache = home / ".codex" / "plugins" / "cache" / "qualixar" / "superlocalmemory-codex"
    _manifest(cache / "4.1.21", "superlocalmemory-codex", "4.1.21", ".codex-plugin")
    assert _installed_plugin_versions() == {
        "claude:superlocalmemory@qualixar": "4.1.21",
        "superlocalmemory-codex": "4.1.21",
    }
