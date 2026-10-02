# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Installs, finds, checks and removes the local answer-check model (Laya).

Laya runs in its own Python environment under ``runtime_dir()``, never inside
SLM's: SLM's dependency pins must not move at runtime, and an environment SLM
manages survives every way SLM itself is upgraded (pip, pipx, uv, npm).

Three install sources can exist at once; ``detect()`` (below) picks between
them and explains, in its own docstring, exactly when each counts as READY.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from superlocalmemory.core import laya_interpreter
from superlocalmemory.infra.data_root import canonical_data_root

try:
    import fcntl
except ImportError:  # Windows: Laya is Apple-Silicon only; detect() already
    fcntl = None  # type: ignore[assignment]  # refuses before any lock is taken.

logger = logging.getLogger(__name__)

LAYA_MLX_REQUIREMENT = "laya-mlx==0.2.0"
LAYA_MODEL_REPO = "aac6fef/laya-mlx"
#: The weights the shipped threshold was measured on (807 MB).
LAYA_MODEL_REVISION = "20aed815fc6acde75733882e7ec0e3f28aeb9717"

STATE_UNSUPPORTED = "unsupported"      # not Apple Silicon
STATE_NOT_INSTALLED = "not_installed"
STATE_INSTALLING = "installing"
STATE_READY = "ready"                  # installed AND the canary check passed
STATE_FAILED = "failed"

_NEEDS_CHECK_STEP = "Needs a check — choose Set up again."

#: ~807 MB — see LAYA_MODEL_REVISION. Used only to estimate install progress.
_EXPECTED_DOWNLOAD_BYTES = 807 * 1024 * 1024
_MIN_FREE_BYTES = int(1.5 * 1024 ** 3)

_VENV_TIMEOUT_S = 120.0
_PIP_TIMEOUT_S = 600.0
_DOWNLOAD_TIMEOUT_S = 1800.0

_NETWORK_MESSAGE = (
    "Couldn't reach the download server. If your network blocks downloads, "
    "use 'Use an existing install'."
)
_NETWORK_HINTS = (
    "timeout", "timed out", "connection", "network", "resolve", "unreachable",
    "ssl", "certificate", "name or service not known",
    "temporary failure in name resolution", "max retries exceeded",
    "connection refused", "connection reset", "no route to host",
)


@dataclass(frozen=True)
class LayaRuntimeStatus:
    state: str
    managed: bool = False      # True = installed by SLM under runtime_dir()
    python: str = ""
    hf_home: str = ""
    model_path: str = ""       # local snapshot folder handed to laya_mlx.load
    model_revision: str = ""
    progress: float = 0.0      # 0..1 while installing
    step: str = ""             # plain-language current step, for the dashboard
    error: str = ""            # plain-language reason; never contains secrets

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressFn = Callable[[float, str], None]

WORKER_PATH = Path(__file__).resolve().parent / "laya_worker.py"


# Small stdlib-only helpers (path / json / time) — no third-party imports so
# this module stays importable even before Laya's own venv exists.

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def runtime_dir(slm_home: Path | None = None) -> Path:
    """``<SLM_HOME>/runtimes/laya``."""
    base = Path(slm_home) if slm_home is not None else canonical_data_root()
    return base / "runtimes" / "laya"


def check_interpreter(python: str, *, strict_location: bool = True) -> str:
    """Why ``python`` must not be run for Laya, or "" when it may be.

    The rule is in core/laya_interpreter.py. ``strict_location`` (a path a
    person typed) also refuses temporary and shared folders; a path SLM saved
    earlier is re-checked without it, since its ownership checks already catch
    anyone else changing it.
    """
    return laya_interpreter.refusal(python, strict_location=strict_location,
                                    in_temp_dir=lambda path: _is_in_temp_dir(path))


# verify() lives in core/laya_verify.py; re-exported here, where callers
# (and tests that replace it) have always found it.
from superlocalmemory.core.laya_verify import VERIFY_BUSY, verify  # noqa: E402


# detect() — fast, offline, never raises.

def _record_matches(record: dict[str, Any] | None, python: str, model_path: str) -> bool:
    return (bool(record) and record.get("python") == python
            and record.get("model_path") == model_path)


def _status_from_record(record: dict[str, Any], *, managed: bool,
                          hf_home_override: str = "") -> LayaRuntimeStatus:
    python = str(record.get("python", ""))
    model_path = str(record.get("model_path", ""))
    hf_home = hf_home_override or str(record.get("hf_home", ""))
    revision = str(record.get("revision", ""))
    verified = bool(record.get("verified"))
    files_ok = (bool(python) and Path(python).exists()
                and bool(model_path) and Path(model_path).is_dir())

    refused = check_interpreter(python, strict_location=False) if files_ok else ""
    if verified and files_ok and not refused:
        return LayaRuntimeStatus(
            state=STATE_READY, managed=managed, python=python, hf_home=hf_home,
            model_path=model_path, model_revision=revision, progress=1.0, step="Ready",
        )
    if refused:
        return LayaRuntimeStatus(
            state=STATE_FAILED, managed=managed, python=python, hf_home=hf_home,
            model_path=model_path, model_revision=revision, step=_NEEDS_CHECK_STEP,
            error=refused,
        )
    return LayaRuntimeStatus(
        state=STATE_FAILED, managed=managed, python=python, hf_home=hf_home,
        model_path=model_path, model_revision=revision, step=_NEEDS_CHECK_STEP,
        error="" if files_ok else "The install can't be found.",
    )


def detect(retrieval_config: Any = None) -> LayaRuntimeStatus:
    """Where Laya is, if anywhere. No network, never raises, under a second."""
    try:
        if not _apple_silicon():
            return LayaRuntimeStatus(state=STATE_UNSUPPORTED)

        for job in (LayaInstallJob.instance(), LayaAdoptJob.instance()):
            job_status = job.status()
            if job_status.state == STATE_INSTALLING:
                return job_status

        run_dir = runtime_dir()
        managed_record = _read_json(run_dir / ".slm-managed")
        adopted_record = _read_json(run_dir / "adopted.json")

        cfg_python = str(getattr(retrieval_config, "sufficiency_python", "") or "").strip()
        cfg_model = str(getattr(retrieval_config, "sufficiency_model", "") or "").strip()
        cfg_hf_home = str(getattr(retrieval_config, "sufficiency_hf_home", "") or "").strip()

        if cfg_python and cfg_model and Path(cfg_model).is_absolute():
            for record, managed in ((adopted_record, False), (managed_record, True)):
                if _record_matches(record, cfg_python, cfg_model):
                    return _status_from_record(
                        record, managed=managed, hf_home_override=cfg_hf_home)
            return LayaRuntimeStatus(
                state=STATE_FAILED, python=cfg_python, hf_home=cfg_hf_home, model_path=cfg_model,
                step=_NEEDS_CHECK_STEP, error="This install hasn't been checked yet.",
            )

        if adopted_record is not None:
            return _status_from_record(adopted_record, managed=False)

        if managed_record is not None:
            return _status_from_record(managed_record, managed=True)

        return LayaRuntimeStatus(state=STATE_NOT_INSTALLED)
    except Exception as exc:  # noqa: BLE001 — detect() must never raise.
        logger.warning("Laya detect() failed unexpectedly: %s", exc)
        return LayaRuntimeStatus(state=STATE_NOT_INSTALLED, error="Couldn't check the install.")


# install() support — one small function per step, so tests can replace any
# one of them without touching the others.

def _venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _hf_cache_model_dir(hf_cache: Path, repo: str) -> Path:
    """Where a repo lives under an HF_HOME: the library's default cache is
    ``HF_HOME/hub``, and the worker runs with HF_HOME = this folder."""
    return hf_cache / "hub" / _repo_folder(repo)


def _repo_folder(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def _managed_snapshot(hf_cache: Path) -> Path:
    return _hf_cache_model_dir(hf_cache, LAYA_MODEL_REPO) / "snapshots" / LAYA_MODEL_REVISION


#: What the on-device model's loader refuses to start without. An interrupted
#: download leaves the snapshot folder in place with some of these absent, or
#: present only as a link to a file that never arrived.
_REQUIRED_WEIGHT_FILES = ("model.safetensors", "rl_agent_config.json",
                          "encoder/config.json", "mlx_config.json",
                          "tokenizer/tokenizer.json")


def _weights_complete(snapshot: Path) -> bool:
    """Whether every file the loader needs is there and readable (links followed)."""
    return snapshot.is_dir() and all((snapshot / name).is_file()
                                     for name in _REQUIRED_WEIGHT_FILES)


def _move_misplaced_weights(hf_cache: Path) -> None:
    """Earlier 4.1.18 builds downloaded into ``hf-cache/models--…`` (one level
    too high). Move such a download into place rather than fetch 800 MB again.
    The cache's links are relative inside the repo folder, so it moves whole."""
    old = hf_cache / _repo_folder(LAYA_MODEL_REPO)
    new = _hf_cache_model_dir(hf_cache, LAYA_MODEL_REPO)
    if old.is_dir() and not old.is_symlink() and not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        os.replace(old, new)


def _folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _check_disk_space(path: Path) -> bool:
    probe = next((p for p in (path, *path.parents) if p.exists()), path)
    try:
        return shutil.disk_usage(probe).free >= _MIN_FREE_BYTES
    except OSError:
        return False


def _classify_subprocess_error(text: str) -> str:
    lowered = (text or "").lower()
    if any(hint in lowered for hint in _NETWORK_HINTS):
        return "network"
    return "other"


def _run_subprocess(cmd: list[str], *, timeout_s: float,
                     env: dict[str, str] | None = None) -> tuple[bool, str, str]:
    """(ok, error_kind, raw_detail-for-logs-only). Never raises."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as exc:
        return False, "timeout", str(exc)
    except OSError as exc:
        return False, "other", str(exc)
    if result.returncode == 0:
        return True, "", ""
    detail = (result.stderr or result.stdout or "")[-2000:]
    return False, _classify_subprocess_error(detail), detail


def _create_venv(venv_dir: Path, *, timeout_s: float) -> tuple[bool, str, str]:
    return _run_subprocess([sys.executable, "-m", "venv", str(venv_dir)], timeout_s=timeout_s)


def _pip_install(python: Path, requirement: str, *, timeout_s: float) -> tuple[bool, str, str]:
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    return _run_subprocess(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", requirement],
        timeout_s=timeout_s, env=env,
    )


def _download_weights(python: Path, repo: str, revision: str, cache_dir: Path, *,
                       progress: ProgressFn | None, timeout_s: float) -> tuple[bool, str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    script = (
        "import sys; from huggingface_hub import snapshot_download; "
        "snapshot_download(sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3]); "
        "print('OK')"
    )
    # Downloading needs the network; never inherit the worker's offline pins.
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    env["TOKENIZERS_PARALLELISM"] = "false"
    try:
        proc = subprocess.Popen(
            [str(python), "-c", script, repo, revision, str(cache_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env,
        )
    except OSError as exc:
        return False, "other", str(exc)

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            proc.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                proc.kill()
                proc.wait(timeout=5)
                return False, "timeout", "download exceeded its time budget"
            if progress is not None:
                done_fraction = min(_folder_size(cache_dir) / _EXPECTED_DOWNLOAD_BYTES, 1.0)
                fraction = 0.35 + 0.55 * done_fraction
                try:
                    progress(fraction, "Downloading the model weights")
                except Exception:  # noqa: BLE001 — a bad UI callback can't abort an install
                    pass
    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.returncode == 0:
        return True, "", ""
    return False, _classify_subprocess_error(stderr), stderr


def _failure_message(kind: str, *, doing: str, do: str) -> str:
    """One plain-language line per step failure. ``doing``/``do`` are a
    gerund and an infinitive phrase for the timeout/generic cases."""
    if kind == "network":
        return _NETWORK_MESSAGE
    if kind == "timeout":
        return f"{doing} took too long and timed out. Please try again."
    return f"Couldn't {do}. Please try again."


def _progress_stamp_path(run_dir: Path) -> Path:
    return run_dir / ".install-steps.json"


def _load_stamp(run_dir: Path) -> dict[str, Any]:
    return _read_json(_progress_stamp_path(run_dir)) or {}


def _mark_stamp(run_dir: Path, step: str) -> None:
    data = _load_stamp(run_dir)
    data[step] = True
    _write_json_atomic(_progress_stamp_path(run_dir), data)


def _existing_created_at(run_dir: Path) -> str:
    data = _read_json(run_dir / ".slm-managed")
    return str(data.get("created_at", "")) if data else ""


def _acquire_install_lock(lock_path: Path):
    """Non-blocking cross-process lock. Returns an open file handle, or None
    when another process already holds it (fails fast; never waits)."""
    if fcntl is None:
        return None
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")  # noqa: SIM115 — lifetime matches install(), closed explicitly
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _release_install_lock(fh) -> None:
    if fh is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        fh.close()


_STEP_VENV, _STEP_PIP, _STEP_WEIGHTS = "venv", "pip", "weights"


def _run_resumable_steps(run_dir: Path, venv_dir: Path, venv_python: Path,
                          hf_cache: Path, report: ProgressFn) -> str | None:
    """venv -> pip -> weights, skipping whatever the stamp says is already
    done. Returns a plain-language error, or None once all three succeed."""
    stamp = _load_stamp(run_dir)

    if not (stamp.get(_STEP_VENV) and venv_python.exists()):
        report(0.0, "Setting up a private Python environment")
        ok, kind, detail = _create_venv(venv_dir, timeout_s=_VENV_TIMEOUT_S)
        if not ok:
            logger.warning("Laya install: venv step failed: %s", detail)
            return _failure_message(kind, doing="Setting up the environment",
                                     do="set up a private Python environment")
        _mark_stamp(run_dir, _STEP_VENV)
    report(0.05, "Python environment ready")

    if not stamp.get(_STEP_PIP):
        report(0.05, "Installing the answer-check package")
        ok, kind, detail = _pip_install(venv_python, LAYA_MLX_REQUIREMENT, timeout_s=_PIP_TIMEOUT_S)
        if not ok:
            logger.warning("Laya install: pip step failed: %s", detail)
            return _failure_message(kind, doing="Installing the answer-check package",
                                     do="install the answer-check package")
        _mark_stamp(run_dir, _STEP_PIP)
    report(0.35, "Package installed")

    _move_misplaced_weights(hf_cache)
    # Done only when the weights are complete: the library resumes a partial
    # download, so fetching again costs only what is missing.
    if not (stamp.get(_STEP_WEIGHTS) and _weights_complete(_managed_snapshot(hf_cache))):
        # cache_dir is the hub folder itself: snapshot_download(cache_dir=X)
        # stores the repo at X/models--…, with no hub/ level of its own.
        ok, kind, detail = _download_weights(
            venv_python, LAYA_MODEL_REPO, LAYA_MODEL_REVISION, hf_cache / "hub",
            progress=report, timeout_s=_DOWNLOAD_TIMEOUT_S,
        )
        if not ok:
            logger.warning("Laya install: download step failed: %s", detail)
            return _failure_message(kind, doing="The download", do="download the model weights")
        _mark_stamp(run_dir, _STEP_WEIGHTS)
    report(0.90, "Weights downloaded")
    return None


def _verify_and_record(run_dir: Path, venv_python: Path, hf_cache: Path,
                        model_path: Path) -> tuple[bool, str]:
    """Runs verify() and writes .slm-managed either way — a failed verify
    still leaves evidence on disk, so detect() reports 'needs a check'
    rather than 'not installed'."""
    ok, reason = verify(str(venv_python), str(hf_cache), str(model_path))
    previous = _read_json(run_dir / ".slm-managed")
    if reason == VERIFY_BUSY and previous is not None and _record_matches(
            previous, str(venv_python), str(model_path)):
        return bool(previous.get("verified")), reason
    record = {
        "requirement": LAYA_MLX_REQUIREMENT,
        "repo": LAYA_MODEL_REPO,
        "revision": LAYA_MODEL_REVISION,
        "created_at": _existing_created_at(run_dir) or _now_iso(),
        "verified": ok,
        "verified_at": _now_iso() if ok else "",
        "python": str(venv_python),
        "model_path": str(model_path),
    }
    _write_json_atomic(run_dir / ".slm-managed", record)
    return ok, reason


def _install_body(run_dir: Path, report: ProgressFn,
                  before_verify: Callable[[], None] | None = None) -> LayaRuntimeStatus:
    """Everything install() does once it holds the lock: disk check, the
    three resumable steps, then verify-and-record."""
    report(0.0, "Checking free disk space")
    if not _check_disk_space(run_dir):
        return LayaRuntimeStatus(
            state=STATE_FAILED, error="Not enough free disk space (needs about 1.5 GB)")

    venv_dir = run_dir / "venv"
    venv_python = _venv_python_path(venv_dir)
    hf_cache = run_dir / "hf-cache"

    error = _run_resumable_steps(run_dir, venv_dir, venv_python, hf_cache, report)
    if error is not None:
        return LayaRuntimeStatus(state=STATE_FAILED, error=error)

    model_path = _managed_snapshot(hf_cache)
    report(0.90, "Checking the install works")
    if before_verify is not None:
        before_verify()
    ok, reason = _verify_and_record(run_dir, venv_python, hf_cache, model_path)

    if not ok:
        return LayaRuntimeStatus(
            state=STATE_FAILED, managed=True, python=str(venv_python), hf_home=str(hf_cache),
            model_path=str(model_path), model_revision=LAYA_MODEL_REVISION,
            step=_NEEDS_CHECK_STEP, error=reason,
        )

    report(1.0, "Ready")
    return LayaRuntimeStatus(
        state=STATE_READY, managed=True, python=str(venv_python), hf_home=str(hf_cache),
        model_path=str(model_path), model_revision=LAYA_MODEL_REVISION, progress=1.0, step="Ready",
    )


def install(*, progress: ProgressFn | None = None,
            slm_home: Path | None = None,
            before_verify: Callable[[], None] | None = None) -> LayaRuntimeStatus:
    """Create the environment, install the pinned package, download the pinned
    weights, verify. Blocking, idempotent, resumable. Needs the network.
    ``before_verify`` runs right before the model is loaded for its check."""

    def _report(fraction: float, step: str) -> None:
        if progress is not None:
            try:
                progress(fraction, step)
            except Exception:  # noqa: BLE001 — a bad UI callback can't abort an install
                pass

    if not _apple_silicon():
        return LayaRuntimeStatus(state=STATE_UNSUPPORTED, error="Laya needs Apple Silicon.")

    try:
        run_dir = runtime_dir(slm_home)
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return LayaRuntimeStatus(
            state=STATE_FAILED, error=f"Couldn't set up the install folder: {exc}")

    lock_fh = _acquire_install_lock(run_dir / "install.lock")
    if fcntl is not None and lock_fh is None:
        return LayaRuntimeStatus(
            state=STATE_FAILED,
            error="Another install is already running. Please wait for it to finish.",
        )

    try:
        return _install_body(run_dir, _report, before_verify)
    except Exception as exc:  # noqa: BLE001 — install() must report, never raise.
        logger.exception("Laya install() hit an unexpected error")
        return LayaRuntimeStatus(state=STATE_FAILED, error=f"Unexpected install error: {exc}")
    finally:
        _release_install_lock(lock_fh)


# adopt() — use an install that already exists.

def _is_in_temp_dir(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path
    try:
        temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
        resolved.relative_to(temp_root)
        return True
    except ValueError:
        pass
    text = str(resolved)
    markers = ("/private/var/folders", "/private/tmp", "/var/folders")
    return any(marker in text for marker in markers)


def _find_snapshot(hf_home: Path | None, repo: str, revision: str) -> Path | None:
    """The pinned (or refs/main) snapshot under an HF_HOME — or under the hub
    cache folder itself, which people paste just as often."""
    if hf_home is None:
        return None
    for model_dir in (_hf_cache_model_dir(hf_home, repo), hf_home / _repo_folder(repo)):
        found = _snapshot_in(model_dir, revision)
        if found is not None:
            return found
    return None


def _snapshot_in(model_dir: Path, revision: str) -> Path | None:
    snapshots = model_dir / "snapshots"
    preferred = snapshots / revision
    if preferred.is_dir():
        return preferred
    refs_main = model_dir / "refs" / "main"
    if refs_main.is_file():
        try:
            commit = refs_main.read_text(encoding="utf-8").strip()
        except OSError:
            commit = ""
        if commit:
            candidate = snapshots / commit
            if candidate.is_dir():
                return candidate
    return None


_NOT_A_REAL_INSTALL = "That looks like a temporary location, not a real install."


def _resolve_adopt_target(
    python: str, hf_home: str, model_path: str,
) -> tuple[Path, Path | None, str] | LayaRuntimeStatus:
    """Validates adopt()'s inputs. Returns the resolved (python_path,
    hf_home_path, model_path) on success, or the FAILED status to return."""
    refused = check_interpreter(python, strict_location=True)
    if refused:
        return LayaRuntimeStatus(state=STATE_FAILED, error=refused)
    python_path = Path(python)

    hf_home_path = Path(hf_home) if hf_home else None
    if hf_home_path is not None:
        if not hf_home_path.exists():
            return LayaRuntimeStatus(
                state=STATE_FAILED, error="That model cache folder can't be found.")
        if _is_in_temp_dir(hf_home_path):
            return LayaRuntimeStatus(state=STATE_FAILED, error=_NOT_A_REAL_INSTALL)

    resolved_model_path = model_path.strip() if model_path else ""
    if not resolved_model_path:
        found = _find_snapshot(hf_home_path, LAYA_MODEL_REPO, LAYA_MODEL_REVISION)
        if found is None:
            return LayaRuntimeStatus(
                state=STATE_FAILED, error="Couldn't find the model under that cache folder.")
        resolved_model_path = str(found)

    model_dir = Path(resolved_model_path)
    if not model_dir.is_dir():
        return LayaRuntimeStatus(state=STATE_FAILED, error="That model folder can't be found.")
    if _is_in_temp_dir(model_dir):
        return LayaRuntimeStatus(state=STATE_FAILED, error=_NOT_A_REAL_INSTALL)

    return python_path, hf_home_path, resolved_model_path


def adopt(python: str, hf_home: str, model_path: str = "") -> LayaRuntimeStatus:
    """Use a Laya install that already exists (for networks that block downloads)."""
    resolved = _resolve_adopt_target(python, hf_home, model_path)
    if isinstance(resolved, LayaRuntimeStatus):
        return resolved
    python_path, _hf_home_path, resolved_model_path = resolved

    ok, reason = verify(str(python_path), hf_home, resolved_model_path)
    if reason == VERIFY_BUSY:
        return LayaRuntimeStatus(state=STATE_FAILED, python=str(python_path),
                                 hf_home=hf_home, model_path=resolved_model_path,
                                 error=reason)

    run_dir = runtime_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "python": str(python_path),
        "hf_home": hf_home,
        "model_path": resolved_model_path,
        "verified": ok,
        "verified_at": _now_iso() if ok else "",
        "reason": reason,
    }
    _write_json_atomic(run_dir / "adopted.json", record)

    if ok:
        return LayaRuntimeStatus(
            state=STATE_READY, managed=False, python=str(python_path),
            hf_home=hf_home, model_path=resolved_model_path,
        )
    return LayaRuntimeStatus(
        state=STATE_FAILED, managed=False, python=str(python_path), hf_home=hf_home,
        model_path=resolved_model_path, step=_NEEDS_CHECK_STEP, error=reason,
    )


# remove() — delete ONLY the managed install.

def remove(slm_home: Path | None = None) -> LayaRuntimeStatus:
    """Delete the SLM-managed install only. Never touches an adopted one."""
    run_dir = runtime_dir(slm_home)
    marker = run_dir / ".slm-managed"
    if not marker.exists():
        return LayaRuntimeStatus(state=STATE_FAILED, error="No managed install to remove.")

    base = Path(slm_home) if slm_home is not None else canonical_data_root()
    try:
        resolved_run_dir = run_dir.resolve(strict=False)
        resolved_base = base.resolve(strict=False)
    except OSError as exc:
        return LayaRuntimeStatus(
            state=STATE_FAILED, error=f"Couldn't resolve the install path: {exc}")

    try:
        resolved_run_dir.relative_to(resolved_base)
    except ValueError:
        return LayaRuntimeStatus(
            state=STATE_FAILED,
            error="Refusing to remove: the install path is outside the data directory.")

    if resolved_run_dir.parts[-2:] != ("runtimes", "laya"):
        return LayaRuntimeStatus(
            state=STATE_FAILED, error="Refusing to remove: unexpected install path.")

    check = run_dir
    while True:
        if check.is_symlink():
            return LayaRuntimeStatus(
                state=STATE_FAILED, error="Refusing to remove: a symlink is in the way.")
        if check == base or check.parent == check:
            break
        check = check.parent

    try:
        shutil.rmtree(run_dir)
    except OSError as exc:
        return LayaRuntimeStatus(state=STATE_FAILED, error=f"Couldn't remove the install: {exc}")

    return LayaRuntimeStatus(state=STATE_NOT_INSTALLED)


# The background jobs the dashboard starts and polls live in core/laya_jobs.py;
# re-exported here, where callers have always found them.
from superlocalmemory.core.laya_jobs import LayaAdoptJob, LayaInstallJob  # noqa: E402
