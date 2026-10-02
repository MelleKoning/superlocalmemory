# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""verify(): load the on-device model once and ask it the canary question.

Split out of core/laya_runtime.py, which re-exports ``verify`` and
``VERIFY_BUSY``. The worker script and the interpreter rule are read from
laya_runtime when verify() runs, so replacing ``laya_runtime.WORKER_PATH``
(tests do) still takes effect.
"""

from __future__ import annotations

import json
import logging
import os
import selectors
import subprocess
import time
from typing import Any

logger = logging.getLogger("superlocalmemory.core.laya_runtime")


def _runtime():
    from superlocalmemory.core import laya_runtime

    return laya_runtime


def _readline(stream, timeout: float) -> str:
    """POSIX-only timed readline, mirroring retrieval/sufficiency.py."""
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as sel:
        sel.register(stream, selectors.EVENT_READ)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not sel.select(timeout=remaining):
            return ""
    return stream.readline() or ""


def _verify_load(proc: subprocess.Popen, model_path: str, hf_home: str,
                  timeout_s: float) -> tuple[bool, str]:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps({"cmd": "load", "model": model_path, "hf_home": hf_home,
                                 "memory_limit_mb": _VERIFY_MEMORY_LIMIT_MB}) + "\n")
    proc.stdin.flush()
    line = _readline(proc.stdout, timeout_s)
    if not line:
        return False, "The check didn't respond in time."
    resp = json.loads(line)
    if not resp.get("ok"):
        logger.warning("Laya verify: load failed: %s", resp.get("error"))
        return False, "Couldn't load the local model."
    return True, ""


def _verify_judge(proc: subprocess.Popen, timeout_s: float) -> tuple[bool, str]:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps({
        "cmd": "judge", "query": "What colour is the sky on a clear day?",
        "documents": ["On a clear day the sky is blue.", "The meeting moved to Thursday."],
    }) + "\n")
    proc.stdin.flush()
    line = _readline(proc.stdout, timeout_s)
    if not line:
        return False, "The check didn't respond in time."
    resp = json.loads(line)
    if not resp.get("ok"):
        logger.warning("Laya verify: judge failed: %s", resp.get("error"))
        return False, "The check could not run."

    probabilities = resp.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != 2:
        return False, "The check returned something unexpected."
    try:
        p_answerable, p_unanswerable = float(probabilities[0]), float(probabilities[1])
    except (TypeError, ValueError):
        return False, "The check returned something unexpected."

    if p_answerable > 0.5 and p_unanswerable < 0.5:
        return True, "Looks good."
    return False, "The check did not pass."


def _kill_worker(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001 — best-effort cleanup only
        pass
    for stream in (proc.stdin, proc.stdout):
        try:
            if stream:
                stream.close()
        except Exception:  # noqa: BLE001
            pass


#: Returned by verify() when another answer check already holds this data
#: folder's one model slot: the canary did not run, so nothing was learned.
VERIFY_BUSY = ("The answer check is running in another SLM process. "
               "Close it, then choose Set up again.")

#: The same ceiling the running judge uses, so the canary is never uncapped.
_VERIFY_MEMORY_LIMIT_MB = 2048


def _take_model_slot() -> tuple[bool, Any]:
    """(free, handle): the one-model-per-data-folder slot the judge itself uses."""
    from superlocalmemory.retrieval import sufficiency

    take = getattr(sufficiency, "_take_slot", None)
    if take is None:  # pragma: no cover — the judge module always has it
        return True, None
    handle = take()
    return handle is not None, handle


def _release_model_slot(handle: Any) -> None:
    if handle is None:
        return
    from superlocalmemory.retrieval import sufficiency

    release = getattr(sufficiency, "_release_slot", None)
    if release is not None:
        release(handle)


def verify(python: str, hf_home: str, model_path: str, *,
           timeout_s: float = 180.0) -> tuple[bool, str]:
    """Load the model offline and require it to separate a known answerable
    pair from a known unanswerable one. (ok, plain-language reason).

    Checks the interpreter again first, and never loads a second model: it
    takes the same slot the running answer check holds, or reports
    ``VERIFY_BUSY`` without starting anything.
    """
    lr = _runtime()
    refused = lr.check_interpreter(python, strict_location=False)
    if refused:
        logger.warning("Laya verify: refusing to run that interpreter: %s", refused)
        return False, refused
    free, slot = _take_model_slot()
    if not free:
        return False, VERIFY_BUSY
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if hf_home:
        env["HF_HOME"] = hf_home
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            [python, str(lr.WORKER_PATH)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env,
        )
        ok, reason = _verify_load(proc, model_path, hf_home, timeout_s)
        if not ok:
            return False, reason
        return _verify_judge(proc, timeout_s)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Laya verify: transport failed: %s", exc)
        return False, "The check failed to run."
    finally:
        _kill_worker(proc)
        _release_model_slot(slot)


__all__ = ["VERIFY_BUSY", "verify"]
