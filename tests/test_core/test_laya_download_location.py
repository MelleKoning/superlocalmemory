# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The weights land where the install then looks for them.

``snapshot_download(cache_dir=X)`` stores a repo at ``X/models--org--name``;
only the library's DEFAULT location adds a ``hub/`` level (HF_HOME/hub). The
install passed its ``hf-cache`` folder as ``cache_dir`` but then looked under
``hf-cache/hub/...``, so every managed install downloaded ~800 MB and failed
its check with "model folder not found".

Nothing here mocks where files go. The download step's own arguments are
captured (no network), the files are put exactly where the real library puts
them for those arguments, the real library is asked to find them again, and
the real verify() then has to load the folder the install recorded.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

# Writes Laya runtime state: pin the data root to tmp_path even without the
# root conftest (--noconftest), see tests/isolation_guard.py.
from ..isolation_guard import explicit_slm_root  # noqa: F401

hub = pytest.importorskip("huggingface_hub")
from huggingface_hub.file_download import repo_folder_name  # noqa: E402

from superlocalmemory.core import laya_runtime as lr  # noqa: E402
from tests.helpers.owned_python import owned_python  # noqa: E402

#: Loads only a folder that exists, as the real worker's _resolve_model does.
_WORKER = r'''
import json, os, sys
for raw in sys.stdin:
    req = json.loads(raw)
    if req.get("cmd") == "load":
        ok = os.path.isdir(req.get("model", ""))
        print(json.dumps({"ok": ok, "error": "" if ok else "model folder not found"}), flush=True)
    elif req.get("cmd") == "judge":
        print(json.dumps({"ok": True, "probabilities": [0.9, 0.1]}), flush=True)
'''


#: What a finished download of the pinned snapshot contains, as far as the
#: on-device model's loader is concerned.
_SNAPSHOT_FILES = ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
                   "mlx_config.json", "tokenizer/tokenizer.json")


def _complete_snapshot(snapshot: Path) -> None:
    for name in _SNAPSHOT_FILES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():  # a finished download replaces a dangling link
            path.unlink()
        path.write_text("{}")


@pytest.fixture(autouse=True)
def _apple_silicon(monkeypatch):
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)


class _FinishedDownload:
    """What Popen returns for the download step: it already finished."""

    returncode = 0
    stderr = io.StringIO("")

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture()
def offline_install(monkeypatch, tmp_path):
    """Every step but the network: the venv is a real interpreter, pip is a
    no-op, and the download puts files where the real library would.

    The venv's python links to a launcher this test owns (below), so
    ``lr.install()``'s final ownership check runs unchanged and passes on
    every machine, whoever can change the Python running the suite.
    """
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER)
    monkeypatch.setattr(lr, "WORKER_PATH", worker)
    monkeypatch.setattr(lr, "_check_disk_space", lambda path: True)

    def _venv(venv_dir, *, timeout_s):
        (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
        (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin\n")
        python = venv_dir / "bin" / "python"
        if not python.exists():
            python.symlink_to(owned_python(tmp_path))
        return True, "", ""

    monkeypatch.setattr(lr, "_create_venv", _venv)
    monkeypatch.setattr(lr, "_pip_install", lambda *a, **k: (True, "", ""))

    downloads: list[Path] = []
    real_popen = subprocess.Popen

    def _popen(argv, *args, **kwargs):
        if len(argv) > 2 and argv[1] == "-c" and "snapshot_download" in argv[2]:
            repo, revision, cache_dir = argv[3], argv[4], Path(argv[5])
            downloads.append(cache_dir)
            snapshot = cache_dir / repo_folder_name(repo_id=repo, repo_type="model") \
                / "snapshots" / revision
            snapshot.mkdir(parents=True, exist_ok=True)
            _complete_snapshot(snapshot)
            # The real library must find it there, offline.
            found = hub.snapshot_download(repo, revision=revision, cache_dir=str(cache_dir),
                                          local_files_only=True)
            assert Path(found) == snapshot
            return _FinishedDownload()
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(lr.subprocess, "Popen", _popen)
    return downloads


def test_a_managed_install_records_a_model_folder_that_exists(offline_install):
    result = lr.install()
    assert offline_install, "the download step never ran"
    assert result.state == lr.STATE_READY, result.error
    assert Path(result.model_path).is_dir()
    record = json.loads((lr.runtime_dir() / ".slm-managed").read_text())
    assert record["verified"] is True
    assert Path(record["model_path"]).is_dir()


def test_the_library_finds_the_recorded_weights_from_the_workers_hf_home(offline_install):
    """The worker runs with HF_HOME = the install's hf_home; the library's
    default cache under it is HF_HOME/hub — the recorded folder must be there."""
    result = lr.install()
    found = hub.snapshot_download(lr.LAYA_MODEL_REPO, revision=lr.LAYA_MODEL_REVISION,
                                  cache_dir=str(Path(result.hf_home) / "hub"),
                                  local_files_only=True)
    assert Path(found) == Path(result.model_path)


def test_weights_left_where_an_earlier_build_put_them_are_used_not_refetched(
        offline_install, monkeypatch):
    """A machine that ran the broken setup has the weights one level up; a new
    Set up moves them into place instead of downloading 800 MB again."""
    run_dir = lr.runtime_dir()
    old = run_dir / "hf-cache" / repo_folder_name(repo_id=lr.LAYA_MODEL_REPO, repo_type="model") \
        / "snapshots" / lr.LAYA_MODEL_REVISION
    old.mkdir(parents=True)
    _complete_snapshot(old)
    (run_dir / ".install-steps.json").write_text(json.dumps({"venv": True, "pip": True,
                                                             "weights": True}))
    lr._create_venv(run_dir / "venv", timeout_s=1)
    result = lr.install()
    assert result.state == lr.STATE_READY, result.error
    assert offline_install == [], "it downloaded again instead of using what was there"
    assert Path(result.model_path).is_dir()


def test_a_weights_step_marked_done_without_its_folder_is_redone(offline_install):
    run_dir = lr.runtime_dir()
    run_dir.mkdir(parents=True)
    (run_dir / ".install-steps.json").write_text(json.dumps({"venv": True, "pip": True,
                                                             "weights": True}))
    lr._create_venv(run_dir / "venv", timeout_s=1)
    result = lr.install()
    assert offline_install, "the stamp said done, the folder was missing, nothing redid it"
    assert result.state == lr.STATE_READY, result.error


def test_an_interrupted_download_is_fetched_again_on_set_up(offline_install):
    """An interrupted download leaves the snapshot folder with its weights
    missing — here a dangling link, as the library leaves one. The stamp says
    done; Set up must still fetch again, or the person is stuck for good."""
    run_dir = lr.runtime_dir()
    snapshot = run_dir / "hf-cache" / "hub" / repo_folder_name(
        repo_id=lr.LAYA_MODEL_REPO, repo_type="model") / "snapshots" / lr.LAYA_MODEL_REVISION
    snapshot.mkdir(parents=True)
    _complete_snapshot(snapshot)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to("../../blobs/not-downloaded-yet")
    (run_dir / ".install-steps.json").write_text(json.dumps({"venv": True, "pip": True,
                                                             "weights": True}))
    lr._create_venv(run_dir / "venv", timeout_s=1)
    result = lr.install()
    assert offline_install, "the weights were incomplete and Set up never fetched them again"
    assert result.state == lr.STATE_READY, result.error


def test_adopt_also_finds_a_snapshot_in_a_hub_cache_folder(tmp_path, monkeypatch):
    """People paste either HF_HOME or its hub/ cache folder; both hold the model."""
    monkeypatch.setattr(lr, "_is_in_temp_dir", lambda p: False)
    hub_cache = tmp_path / "hfhome" / "hub"
    snap = hub_cache / repo_folder_name(repo_id=lr.LAYA_MODEL_REPO, repo_type="model") \
        / "snapshots" / lr.LAYA_MODEL_REVISION
    snap.mkdir(parents=True)
    assert lr._find_snapshot(tmp_path / "hfhome", lr.LAYA_MODEL_REPO,
                             lr.LAYA_MODEL_REVISION) == snap
    assert lr._find_snapshot(hub_cache, lr.LAYA_MODEL_REPO, lr.LAYA_MODEL_REVISION) == snap
