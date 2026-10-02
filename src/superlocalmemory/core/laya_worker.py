# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Laya sufficiency worker — runs in its own process, never imported by the daemon.

Answers one question about a recall: does each of these memories contain the
specific information that answers this question? The daemon uses the answer to
say "I found nothing that answers this" instead of handing back the best of a
bad set with a confident face.

Self-contained on purpose: it imports nothing from superlocalmemory, so it can
run under any interpreter that has ``laya_mlx`` — SLM's own when the ``laya``
extra is installed, or a separate venv named in config.

Protocol, JSON lines on stdin/stdout, mirroring the reranker worker:

  {"cmd": "ping"}
      -> {"ok": true, "backend": "laya-mlx", "model": "...", "loaded": bool}
  {"cmd": "load", "model": "...", "hf_home": "...", "memory_limit_mb": 2048}
      -> {"ok": bool, "model": "...", "error": "..."}
      "model" is a Hugging Face repo id, or a local folder ("/...", "~/...").
  {"cmd": "judge", "query": "...", "documents": ["...", ...], "question": "..."}
      -> {"ok": true, "probabilities": [0.91, 0.12, ...]}
      "question" is the recipe's wording; without it the built-in one is asked.
  {"cmd": "quit"}

Every request may carry an ``"id"``; its reply then echoes it, so a judge that
gave up on a slow answer never reads that late answer as the reply to its next
question. A request without one gets exactly the reply above. A failed load
also says ``"error_kind": "permanent"`` (the library, the weights or the folder
is missing: retrying cannot help) or ``"transient"`` (anything else).

stdout carries the protocol and nothing else. laya-mlx prints download progress
and calibration warnings; every library call therefore runs with stdout
redirected to stderr, or a stray line would corrupt the next response.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time

# Offline before any Hugging Face import: the loader calls snapshot_download()
# even when the weights are cached, and must never reach the network from here.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if sys.platform != "win32":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

#: Measured on a real store (2026-10-02): this wording, judged over the top 3
#: results, separated "the set holds the answer" from "it does not" at AUC 0.837
#: where the retrieval score itself scored 0.524. Changing it changes that number.
#: The daemon sends its recipe's wording with each request; this copy is what an
#: older daemon that sends none gets, and must stay equal to RECIPE_V1.question.
DEFAULT_QUESTION = (
    "Read `question` and `memory`. Is this statement true: the memory "
    "contains the specific information that answers the question."
)
DEFAULT_MODEL = "aac6fef/laya-mlx"
MAX_QUESTION_CHARS = 1000
#: The English checkpoint reads 512 tokens; longer memories are cut, not refused.
MAX_DOCUMENT_CHARS = 1800
MAX_DOCUMENTS = 16
#: Freed GPU buffers MLX may keep for reuse. See ``_limit_memory``.
CACHE_LIMIT_MB = 128
#: How often the watchdog checks that its parent is still the one that started it.
WATCH_INTERVAL_S = 5.0
#: Failures retrying cannot fix: the library, the weights or the folder is not
#: there. Named, not imported — the worker must not import huggingface_hub to
#: classify an error it raised.
_PERMANENT_ERRORS = frozenset({
    "ImportError", "ModuleNotFoundError", "FileNotFoundError", "NotADirectoryError",
    "LocalEntryNotFoundError", "EntryNotFoundError", "RepositoryNotFoundError",
    "RevisionNotFoundError", "GatedRepoError", "OfflineModeIsEnabled",
})


def _questions(instructions: str) -> dict:
    return {"sufficient": {"type": "noul", "instructions": instructions}}


QUESTION = _questions(DEFAULT_QUESTION)


#: The id of the request being answered, echoed in its reply. One request at a
#: time is read and answered, so a module-level value is enough.
_CURRENT_ID: list = [None]


def _respond(payload: dict) -> None:
    if _CURRENT_ID[0] is not None:
        payload = {**payload, "id": _CURRENT_ID[0]}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _error_kind(exc: BaseException) -> str:
    names = {cls.__name__ for cls in type(exc).__mro__}
    return "permanent" if names & _PERMANENT_ERRORS else "transient"


def _watch_parent(*, start_thread: bool = True, stop=None) -> None:
    """Exit when the parent is gone. An orphaned worker holds the model resident.

    Compares ``os.getppid()`` with the parent this worker started under: when
    the parent dies the worker is re-parented, so the id changes. Probing the
    OLD id instead cannot tell a dead parent from a new process that reused
    its number.
    """
    parent = os.getppid()
    if parent <= 1:
        return

    def watch() -> None:
        while not (stop() if stop is not None else False):
            time.sleep(WATCH_INTERVAL_S)
            if os.getppid() != parent:
                os._exit(0)

    if start_thread:
        threading.Thread(target=watch, daemon=True, name="parent-watchdog").start()
    else:
        watch()


def _resolve_model(model: str) -> str:
    """A local folder by its absolute path; anything else is a repo id.

    A missing folder is refused here: handed to the library it would be read
    as a repo id and fail offline with an error about the network. The error
    names no path, because it is logged by the daemon.
    """
    if not model.startswith(("/", "~", "./", "../")):
        return model
    folder = os.path.abspath(os.path.expanduser(model))
    if not os.path.isdir(folder):
        raise FileNotFoundError("model folder not found")
    return folder


def _question_from(req: dict) -> str | None:
    """The wording to ask: the request's, the built-in one when it sent none,
    and None when it sent something malformed — never a silent substitute."""
    if "question" not in req:
        return DEFAULT_QUESTION
    question = req["question"]
    if isinstance(question, str) and question.strip() and len(question) <= MAX_QUESTION_CHARS:
        return question
    return None


def _load(model: str, hf_home: str, memory_limit_mb: int):
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    with contextlib.redirect_stdout(sys.stderr):
        import laya_mlx

        if memory_limit_mb > 0:
            _limit_memory(memory_limit_mb)
        agent = laya_mlx.load(model, dtype="float16", batch_size=16,
                              compile=False, cache_prompts=True)
        # Pay the first-call cost here, not on a user's recall.
        agent.predict({"question": "warm-up", "memory": "warm-up"}, QUESTION)
    return agent


def _limit_memory(memory_limit_mb: int) -> None:
    """Bound MLX's working memory, and cap the freed buffers it keeps.

    The memory limit alone does not cap the cache of freed buffers: MLX keeps
    those up to the memory limit by default. Measured on Apple Silicon, that
    held the worker at 2.2 GB; with the cache capped at ``CACHE_LIMIT_MB`` it
    held 1.1 GB, answered identically and was no slower.
    """
    try:
        import mlx.core as mx
    except Exception:
        return
    _call_first(mx, "set_memory_limit", memory_limit_mb * 1024 * 1024)
    _call_first(mx, "set_cache_limit", CACHE_LIMIT_MB * 1024 * 1024)


def _call_first(mx, name: str, value: int) -> None:
    """``mx.<name>(value)``, or the older ``mx.metal.<name>``; absent is fine."""
    for setter in (getattr(mx, name, None),
                   getattr(getattr(mx, "metal", None), name, None)):
        if callable(setter):
            try:
                setter(value)
                return
            except Exception:
                continue


def _judge(agent, query: str, documents: list[str], question: str) -> list[float]:
    probabilities: list[float] = []
    questions = _questions(question)
    with contextlib.redirect_stdout(sys.stderr):
        for document in documents[:MAX_DOCUMENTS]:
            answer = agent.predict(
                {"question": query, "memory": document[:MAX_DOCUMENT_CHARS]},
                questions,
            )["answers"]["sufficient"]
            value = answer.get("noul", answer.get("probability"))
            probabilities.append(float(value))
    return probabilities


def _handle_load(req: dict):
    """(agent or None, model name). A failed load is reported, never raised."""
    model_name = str(req.get("model") or DEFAULT_MODEL)
    try:
        agent = _load(_resolve_model(model_name), str(req.get("hf_home") or ""),
                      int(req.get("memory_limit_mb") or 0))
    except Exception as exc:  # noqa: BLE001 — reported, never raised
        _respond({"ok": False, "model": model_name,
                  "error": f"{type(exc).__name__}: {exc}",
                  "error_kind": _error_kind(exc)})
        return None, model_name
    _respond({"ok": True, "model": model_name})
    return agent, model_name


def _handle_judge(agent, req: dict) -> None:
    if agent is None:
        _respond({"ok": False, "error": "not loaded"})
        return
    query = req.get("query")
    documents = req.get("documents")
    if not isinstance(query, str) or not isinstance(documents, list):
        _respond({"ok": False, "error": "missing query or documents"})
        return
    question = _question_from(req)
    if question is None:
        _respond({"ok": False, "error": "invalid question"})
        return
    try:
        _respond({"ok": True, "probabilities":
                  _judge(agent, query, [str(d) for d in documents], question)})
    except Exception as exc:  # noqa: BLE001
        _respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    _watch_parent()
    agent = None
    model_name = ""
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            _CURRENT_ID[0] = None
            _respond({"ok": False, "error": "invalid JSON"})
            continue
        cmd = req.get("cmd", "") if isinstance(req, dict) else ""
        request_id = req.get("id") if isinstance(req, dict) else None
        _CURRENT_ID[0] = request_id if isinstance(request_id, (int, str)) else None
        if cmd == "quit":
            break
        if cmd == "ping":
            _respond({"ok": True, "backend": "laya-mlx", "model": model_name,
                      "loaded": agent is not None})
        elif cmd == "load":
            agent, model_name = _handle_load(req)
        elif cmd == "judge":
            _handle_judge(agent, req)
        else:
            _respond({"ok": False, "error": f"unknown command: {cmd}"})


if __name__ == "__main__":
    main()
