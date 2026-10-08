# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Does a recall actually answer the question? A separate decision from ranking.

Retrieval produces an ordering, and an ordering cannot say "none of these".
Measured on a real five-month store, the top retrieval score was the same
whether the top result was right (0.654), wrong (0.659) or the question had no
answer in memory at all (0.660) — so no threshold on it can produce
abstention. A typed decision model can: judged over the top three results,
Laya separated "this set holds the answer" from "it does not" at AUC 0.837,
against 0.524 for the retrieval score.

The judge runs Laya in its own process (``laya_worker.py``) and owns its own
readiness. The caller never inspects a private flag to decide whether to ask —
the reranker did, and a dead worker then stayed dead for seven days because the
code that re-warms it was behind that very check.

Fails open, always, to today's behaviour: off Apple Silicon, without the
weights, while warming, when busy, on timeout or on a malformed answer,
``judge()`` returns None and the recall is reported exactly as before —
``assess()`` says which of those it was.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from superlocalmemory.core import recall_gate
from superlocalmemory.encoding.memory_kind_recipe import KindAnswer, KindRecipe
from superlocalmemory.retrieval import answer_question_forms as question_forms
from superlocalmemory.retrieval import laya_kinds
from superlocalmemory.retrieval.answer_check_status import (
    DETAIL_REUSED,
    JUDGE_FLOOR_S,
    STATUS_BUSY,
    STATUS_JUDGED,
    STATUS_SKIPPED,
    STATUS_UNAVAILABLE,
    STATUS_WARMING,
    JudgeOutcome,
    effective_deadline,
    seconds_left,
)
from superlocalmemory.retrieval.judge_recipe import (
    ACTIVE_RECIPE,
    UNMEASURED,
    Calibration,
    JudgeDocument,
    JudgeRecipe,
    calibration_or_unmeasured,
    coerce_documents,
)
from superlocalmemory.retrieval.laya_transport import (
    KIND_PERMANENT,
    LineReader,
    cooldown_s,
    failure_kind,
    write_request,
)
from superlocalmemory.retrieval.laya_transport import close_pipes as _close_pipes
from superlocalmemory.retrieval.laya_transport import stop_process as _stop_process

logger = logging.getLogger(__name__)

WORKER_PATH = Path(__file__).resolve().parent.parent / "core" / "laya_worker.py"
DEFAULT_MODEL = "aac6fef/laya-mlx"
#: Judged K. Measured: top-1 AUC 0.787, top-3 0.837, top-5 0.832, top-10 0.806 —
#: judging more candidates lets a low-ranked memory that scores high by chance
#: inflate the maximum for a set that holds no answer.
DEFAULT_TOP_K = 3
#: Provisional. Chosen on the same 124 questions it was measured on (top-3: keeps
#: 40/41 answerable sets, abstains on 21/27 unanswerable). Not calibrated — the
#: response says so — until a held-out set confirms it. The judge reads its
#: threshold from judge_recipe.CALIBRATIONS; these two names stay for importers.
DEFAULT_THRESHOLD = 0.6
CALIBRATION_STATUS = "measured_small_sample_not_calibrated"

#: A Hugging Face cache snapshot: .../models--<org>--<name>/snapshots/<commit>.
_HF_SNAPSHOT = re.compile(
    r"models--(?P<org>[A-Za-z0-9][\w.-]*?)--(?P<name>[A-Za-z0-9][\w.-]*)"
    r"[/\\]snapshots[/\\](?P<rev>[0-9a-fA-F]{7,64})(?:[/\\]|$)"
)
#: "org/name" or a bare legacy name. Never starts with "/", "~" or ".".
_REPO_ID = re.compile(r"^(?:[A-Za-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]*$")

_LOAD_TIMEOUT_S = 180.0
_WARMUP_ATTEMPTS = 3
_WARMUP_BACKOFF_S = 5.0
#: After a failed warm-up cycle or a lost worker, the next cycle waits this
#: long, doubling with each failure in a row up to ``_COOLDOWN_MAX_S``. A load
#: that is failing for good (but did not say so) costs a few attempts an hour
#: instead of three every thirty seconds.
_COOLDOWN_BASE_S = 60.0
_COOLDOWN_MAX_S = 1800.0
#: A request abandoned at its deadline whose reply has not arrived after this
#: long means the worker is stuck, not slow: it is replaced. Twenty times the
#: default judgement timeout, so a slow judgement is never mistaken for it.
_WEDGED_S = 30.0
#: Memory typing borrows the worker between recalls. It never waits on the
#: lock (a recall may hold it); it tries this many times, waiting for recalls
#: to finish in between, then gives up and the rules answer instead.
_BACKGROUND_ATTEMPTS = 3
#: Pause after giving the worker back, so a recall blocked on the lock gets it
#: before typing's next chunk does (the lock itself is not first-come-first-served).
_BACKGROUND_YIELD_S = 0.005


@dataclass(frozen=True)
class SufficiencyVerdict:
    """What the judge said about one recall.

    ``calibration_status`` travels with the verdict because it belongs to the
    backend and recipe that produced it, not to the response format.
    """

    probabilities: tuple[float, ...]
    threshold: float
    calibration_id: str
    calibration_status: str = CALIBRATION_STATUS
    backend: str = "laya"
    #: Memories an explicit rule recognised as answering (answer_question_forms):
    #: sufficient whatever the model's own numbers, which are kept as they were.
    rule_support: tuple[int, ...] = ()

    @property
    def answer_confidence(self) -> float:
        return max(self.probabilities)

    @property
    def insufficient(self) -> bool:
        return self.answer_confidence < self.threshold and not self.rule_support


@runtime_checkable
class SufficiencyJudge(Protocol):
    """The answer check, whichever model runs it.

    One engine holds at most one judge. Every implementation fails open:
    ``judge`` returns None whenever it cannot answer now, and None always means
    the recall is reported exactly as it would have been without a judge.
    """

    backend: str
    top_k: int

    @property
    def ready(self) -> bool: ...

    def judge(self, query: str,
              documents: Sequence[JudgeDocument]) -> SufficiencyVerdict | None: ...

    def shutdown(self) -> None: ...

    # Optional, and used when present: ``assess(query, documents, *,
    # deadline=None) -> JudgeOutcome`` — the verdict plus why there is none.


def laya_supported() -> bool:
    """Laya-MLX ships Apple-Silicon wheels only."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def model_identity(model: str) -> tuple[str, str]:
    """(repo, revision) of the weights, for naming a measurement.

    Never a filesystem path: the id travels in every recall response, and a
    path would put the user's home folder in each one. A local folder that is
    not a Hugging Face snapshot is just "local"; a repo id loads whatever
    revision is cached, so its revision is "unpinned".
    """
    text = str(model or "").strip()
    found = _HF_SNAPSHOT.search(text)
    if found:
        return f"{found['org']}/{found['name']}", found["rev"].lower()
    if _REPO_ID.match(text) and ".." not in text:
        return text, "unpinned"
    return "local", "unpinned"


def calibration_for_weights(model: str, recipe: JudgeRecipe) -> Calibration:
    """The recipe's measured threshold — only for the weights it was measured on.

    The threshold was measured on one revision of the weights. Any other —
    another snapshot, a folder that is not a snapshot, a repo id that loads
    whatever happens to be cached — reports confidence but cannot abstain,
    the same rule as any pair nobody measured.
    """
    from superlocalmemory.core.laya_runtime import LAYA_MODEL_REPO, LAYA_MODEL_REVISION

    repo, revision = model_identity(model)
    if repo != LAYA_MODEL_REPO or revision != LAYA_MODEL_REVISION.lower():
        return UNMEASURED
    return calibration_or_unmeasured("laya", recipe)


def laya_calibration_id(model: str, recipe: JudgeRecipe, top_k: int) -> str:
    repo, revision = model_identity(model)
    return f"laya:{repo}@{revision[:12]}:{recipe.recipe_id}:top{top_k}"


def _valid_probabilities(value: object, expected: int) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != expected or not value:
        return None
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        f = float(item)
        if not math.isfinite(f) or not 0.0 <= f <= 1.0:
            return None
        out.append(f)
    return tuple(out)


#: How long a process that found Laya already running elsewhere waits before
#: looking again. Checking is cheap, but a recall must not start a thread each time.
_SLOT_RETRY_S = 60.0


def _take_slot():
    """A claim on running Laya for this SLM data folder, held for the judge's lifetime.

    Every SLM process can build an engine — the daemon, a recall worker, a
    one-shot CLI command — and each would otherwise load its own copy of the
    model next to the daemon's. An exclusive lock on one file in the data
    folder allows exactly one, and the operating system drops it the moment its
    holder dies, so a crashed daemon never leaves the next one locked out. None
    when another process holds it.

    The scope is the data folder, not the computer: every profile, the CLI and
    the MCP server of one install share it, so an install runs one model. A
    second install with its own data folder is a separate SLM and gets its own.
    """
    try:
        import fcntl

        from superlocalmemory.infra.data_root import state_path
        path = state_path(".laya-judge.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 — held open on purpose
    except (ImportError, OSError, ValueError):
        return None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _release_slot(handle) -> None:
    if handle is None:
        return
    try:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError, ValueError):
        pass
    try:
        handle.close()
    except OSError:
        pass


class LayaSufficiencyJudge:
    """Asks a resident Laya worker whether recalled memories answer a question.

    The threshold is not a constructor argument: it belongs to the (backend,
    recipe) pair it was measured on, in judge_recipe.CALIBRATIONS, and only to
    the weights it was measured on. Any other recipe or weights still report
    confidence, but can never abstain.

    Recovery rules (each one a fix):

    * A load that can never succeed — the library, the weights, the folder or
      the interpreter is missing — is tried once; the judge then stays off
      until its setup changes (a new setup builds a new judge).
    * Other load failures retry within a cycle, and each failed cycle waits
      longer than the last before the next one may start.
    * A judgement that misses its deadline costs that recall its verdict only.
      The worker keeps the model; its late reply is read and discarded by the
      next request. Only a worker that has not answered for ``_WEDGED_S`` is
      replaced.
    """

    backend = "laya"

    def __init__(
        self,
        *,
        python: str | None = None,
        model: str = DEFAULT_MODEL,
        hf_home: str = "",
        timeout_s: float = 1.5,
        memory_limit_mb: int = 2048,
        top_k: int = DEFAULT_TOP_K,
        recipe: JudgeRecipe = ACTIVE_RECIPE,
        worker_path: Path | None = None,
        start: bool = True,
    ) -> None:
        self._python = python or sys.executable
        self._worker_path = Path(worker_path) if worker_path else WORKER_PATH
        self._model = model
        self._hf_home = hf_home
        self._timeout_s = timeout_s
        #: The least time worth asking with, after waiting for a busy worker:
        #: the recall floor, or half this judge's own timeout if that is shorter.
        self._min_ask_s = min(JUDGE_FLOOR_S, max(0.0, float(timeout_s)) / 2)
        self._memory_limit_mb = memory_limit_mb
        self._recipe = recipe
        calibration = calibration_for_weights(model, recipe)
        self.top_k = top_k
        self.threshold = calibration.threshold
        self.calibration_status = calibration.status
        self.calibration_id = laya_calibration_id(model, recipe, top_k)
        self._proc: subprocess.Popen | None = None
        self._reader: tuple[subprocess.Popen, LineReader] | None = None
        self._ready = False
        self._loading = False
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._slot = None
        self._slot_retry_at = 0.0
        self._ids = itertools.count(1)
        #: (id, sent at) of a request a recall gave up on; its reply is still due.
        self._stale: tuple[int, float] | None = None
        self._permanent_error = ""
        self._failures = 0
        self._next_attempt_at = 0.0
        if start:
            self.start_warmup()

    # -- readiness --------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        return self._loading

    @property
    def closed(self) -> bool:
        """Shut down for good. A closed judge answers None to everything."""
        return self._shutdown.is_set()

    def start_warmup(self) -> None:
        """Load the model in the background. Idempotent; never blocks.

        The check-and-set is locked: two recalls arriving on a cold judge
        would otherwise each start a warm-up and load the model twice. Never
        after a failure that retrying cannot fix, nor during a cooldown.
        """
        with self._state_lock:
            if self._shutdown.is_set() or self._loading or self._ready:
                return
            if self._permanent_error or time.monotonic() < self._next_attempt_at:
                return
            if self._slot is None and not self._claim_slot():
                return
            self._loading = True
        threading.Thread(target=self._warmup, daemon=True,
                         name="laya-sufficiency-warmup").start()

    def _claim_slot(self) -> bool:
        """Take this data folder's one Laya slot, or note when to look again.

        Called with ``_state_lock`` held. A process that finds the slot taken
        reports its recalls unjudged, exactly as before the answer check.
        """
        now = time.monotonic()
        if now < self._slot_retry_at:
            return False
        self._slot = _take_slot()
        if self._slot is None:
            if self._slot_retry_at == 0.0:
                logger.info("Answer check runs in another SLM process using this data "
                            "folder; this one reports recalls unjudged")
            self._slot_retry_at = now + _SLOT_RETRY_S
            return False
        return True

    def _warmup(self) -> None:
        try:
            for attempt in range(1, _WARMUP_ATTEMPTS + 1):
                if self._shutdown.is_set():
                    return
                resp, spawn_error = self._load_once()
                if resp and resp.get("ok") and not self._shutdown.is_set():
                    self._ready = True
                    logger.info("Laya sufficiency judge ready (%s)", self._model)
                    return
                self._kill()
                if spawn_error or failure_kind(resp) == KIND_PERMANENT:
                    self._stop_for_good(spawn_error or str((resp or {}).get("error", "")))
                    return
                logger.warning("Laya sufficiency judge load failed (attempt %d/%d): %s",
                               attempt, _WARMUP_ATTEMPTS,
                               (resp or {}).get("error", "no response"))
                self._shutdown.wait(_WARMUP_BACKOFF_S * attempt)
            self._note_failure()
        finally:
            self._loading = False

    def _load_once(self) -> tuple[dict | None, str]:
        """(reply, "") or (None, why the worker could not even start)."""
        try:
            return self._request({"cmd": "load", "model": self._model,
                                  "hf_home": self._hf_home,
                                  "memory_limit_mb": self._memory_limit_mb},
                                 deadline=time.monotonic() + _LOAD_TIMEOUT_S), ""
        except _SpawnFailed as exc:
            return None, str(exc)

    def _stop_for_good(self, error: str) -> None:
        self._permanent_error = error or "the model cannot be loaded"
        logger.warning("Answer check (Laya) off until its setup changes: %s",
                       self._permanent_error[:200])

    def _note_failure(self) -> None:
        """One more failure in a row: the next warm-up waits longer."""
        self._failures += 1
        self._next_attempt_at = time.monotonic() + cooldown_s(
            self._failures, base_s=_COOLDOWN_BASE_S, max_s=_COOLDOWN_MAX_S)

    # -- the decision -----------------------------------------------------

    def judge(self, query: str, documents: Sequence[JudgeDocument | str], *,
              deadline: float | None = None) -> SufficiencyVerdict | None:
        """A verdict on the top documents, or None when the judge cannot answer now."""
        return self.assess(query, documents, deadline=deadline).verdict

    reuses_after_lock = True  #: ``assess`` accepts ``reuse`` (answer_check_stage)

    def assess(self, query: str, documents: Sequence[JudgeDocument | str], *,
               deadline: float | None = None,
               reuse: Callable[[], SufficiencyVerdict | None] | None = None) -> JudgeOutcome:
        """A verdict on the top documents, and what became of the check.

        Never waits for a cold worker: if the model is not loaded this starts
        the warm-up and says "warming" — loading takes tens of seconds, beyond
        any recall budget. Waits for a worker busy with another recall, but
        only inside ``deadline`` (monotonic; the recall's), which also bounds
        the judgement itself together with this judge's own timeout.

        ``reuse`` is asked once the worker is this recall's, before anything is
        sent: a verdict that became known meanwhile is returned (``DETAIL_REUSED``).
        """
        docs = coerce_documents(documents[: self.top_k])
        if not isinstance(query, str) or not query or not docs:
            return JudgeOutcome(None, STATUS_SKIPPED)
        if self._shutdown.is_set():
            return JudgeOutcome(None, STATUS_UNAVAILABLE)
        if not self._ready:
            self.start_warmup()
            return JudgeOutcome(None, STATUS_WARMING if self._loading else STATUS_UNAVAILABLE)
        rendered = [self._recipe.render(d) for d in docs]
        return self._ask(query, rendered, effective_deadline(deadline, self._timeout_s),
                         reuse=reuse)

    def assess_if_idle(self, query: str,
                       documents: Sequence[JudgeDocument | str]) -> JudgeOutcome:
        """``assess`` for work nobody is waiting on, which never makes a recall wait.

        Asks only a warm worker that is free right now (``busy`` otherwise,
        without waiting), bounded by this judge's own timeout. Never starts a
        warm-up: a check finished after its recall returned is not a reason to
        load a model. Asked one memory at a time (``core.answer_check_deferred``).
        """
        docs = coerce_documents(documents[: self.top_k])
        if not isinstance(query, str) or not query or not docs:
            return JudgeOutcome(None, STATUS_SKIPPED)
        if self._shutdown.is_set() or not self._ready:
            return JudgeOutcome(None, STATUS_UNAVAILABLE)
        rendered = [self._recipe.render(d) for d in docs]
        return self._ask(query, rendered, effective_deadline(None, self._timeout_s),
                         wait=False)

    def _ask(self, query: str, rendered: list[str], deadline: float, *,
             wait: bool = True,
             reuse: Callable[[], SufficiencyVerdict | None] | None = None) -> JudgeOutcome:
        acquired = (self._lock.acquire(timeout=max(0.0, seconds_left(deadline) - self._min_ask_s))
                    if wait else self._lock.acquire(blocking=False))
        if not acquired:
            return JudgeOutcome(None, STATUS_BUSY)
        proc = self._proc  # read once: shutdown() swaps it without the lock
        try:
            known = reuse() if reuse is not None else None
            if isinstance(known, SufficiencyVerdict):  # became known while it waited
                return JudgeOutcome(known, STATUS_JUDGED, DETAIL_REUSED)
            return self._ask_locked(proc, query, rendered, deadline)
        except (BrokenPipeError, EOFError, OSError, ValueError) as exc:
            logger.warning("Laya sufficiency judge transport failed: %s", type(exc).__name__)
            self._worker_lost()
            return JudgeOutcome(None, STATUS_UNAVAILABLE)
        finally:
            self._after_use(proc)
            self._lock.release()
            if not self._ready and not self._loading:
                self.start_warmup()

    def _after_use(self, proc: subprocess.Popen | None) -> None:
        """A shutdown that found this request using the worker left its pipes
        to the request: closing them under a reader could hand the reader a
        reused descriptor."""
        if proc is not None and self._shutdown.is_set():
            _close_pipes(proc)

    def _ask_locked(self, proc: subprocess.Popen | None, query: str, rendered: list[str],
                    deadline: float) -> JudgeOutcome:
        if proc is None or proc.poll() is not None:
            if proc is not None:
                self._worker_lost()
            self._ready = False
            return JudgeOutcome(None, STATUS_UNAVAILABLE)
        if not self._drain_stale(proc, deadline):
            return JudgeOutcome(None, STATUS_BUSY if self._ready else STATUS_UNAVAILABLE)
        answers: list[tuple[float, ...]] = []
        # The question as typed, then any rewording of its shape: all or nothing.
        for asked in (query, *question_forms.reworded(query)):
            if seconds_left(deadline) < self._min_ask_s:
                return JudgeOutcome(None, STATUS_BUSY)
            got = self._probabilities(proc, asked, rendered, deadline)
            if got is None:
                return JudgeOutcome(None, STATUS_UNAVAILABLE)
            answers.append(got)
        probabilities = answers[0]
        for more in answers[1:]:
            probabilities = question_forms.merged(probabilities, more)
        self._failures = 0
        forms_used = question_forms.applies(query)
        return JudgeOutcome(SufficiencyVerdict(
            probabilities, self.threshold,
            self.calibration_id + ("+" + question_forms.FORMS_ID if forms_used else ""),
            self.calibration_status, self.backend,
            question_forms.rule_support(query, rendered)), STATUS_JUDGED)

    def _probabilities(self, proc: subprocess.Popen, query: str, rendered: list[str],
                       deadline: float) -> tuple[float, ...] | None:
        resp = self._exchange(proc, {"cmd": "judge", "query": query, "documents": rendered,
                                     "question": self._recipe.question}, deadline)
        if resp is None:
            logger.info("Laya sufficiency judge did not answer in time; "
                        "this recall is reported unjudged")
            return None
        probabilities = (_valid_probabilities(resp.get("probabilities"), len(rendered))
                         if resp.get("ok") else None)
        if probabilities is None and resp.get("ok"):
            logger.warning("Laya sufficiency judge returned a malformed answer; ignoring it")
        return probabilities

    # -- memory typing (background only) ------------------------------------

    def ask_kinds(self, documents: Sequence[str], recipe: KindRecipe,
                  verify_indices: Sequence[int]) -> list[KindAnswer] | None:
        """One kind answer per document from the running worker, or None.

        Background threads only (``recall_gate.background_work()``); on any
        other thread nothing is sent. Borrows the worker the answer check
        already runs and never starts, warms or replaces one: a cold or closed
        judge answers None. Before each chunk of ``laya_kinds.CHUNK_DOCUMENTS``
        it waits until no recall is in flight, takes the worker only if it is
        free, and hands it back after the chunk — so a recall that arrives
        mid-typing waits for one short chunk, inside its own deadline, instead
        of going unjudged. All-or-nothing: any refused, late or malformed chunk
        makes the whole answer None.
        """
        if not recall_gate.is_background_work():
            return None
        chunks = laya_kinds.build_chunks(documents, recipe, verify_indices)
        if chunks is None:
            return None
        answers: list[KindAnswer] = []
        for req in chunks:
            if (self._shutdown.is_set() or not self._ready
                    or recall_gate.background_preempt_requested()):
                return None
            recall_gate.wait_for_foreground_idle()
            reply = self._background_exchange(
                req, laya_kinds.chunk_timeout_s(len(req["documents"])))
            parsed = laya_kinds.parse_reply(reply, req, recipe)
            if parsed is None:
                return None
            answers.extend(parsed)
            time.sleep(_BACKGROUND_YIELD_S)
        return answers

    def _background_exchange(self, req: dict, timeout_s: float) -> dict | None:
        """One request on the live worker if it is free; never spawns, never blocks on it."""
        for _ in range(_BACKGROUND_ATTEMPTS):
            if self._lock.acquire(blocking=False):
                break
            recall_gate.wait_for_foreground_idle()
            time.sleep(_BACKGROUND_YIELD_S)
        else:
            return None
        proc = self._proc
        try:
            if proc is None or proc.poll() is not None or not self._ready:
                return None
            deadline = time.monotonic() + timeout_s
            if not self._drain_stale(proc, deadline):
                return None
            return self._exchange(proc, req, deadline)
        except (BrokenPipeError, EOFError, OSError, ValueError) as exc:
            logger.warning("Laya worker transport failed during memory typing: %s",
                           type(exc).__name__)
            self._worker_lost()
            return None
        finally:
            self._after_use(proc)
            self._lock.release()

    # -- transport --------------------------------------------------------

    def _spawn(self) -> None:
        from superlocalmemory.core.laya_runtime import check_interpreter

        # An interpreter other accounts could change is never started: it would
        # run as this user, with this user's environment.
        refused = check_interpreter(self._python, strict_location=False)
        if refused:
            raise _SpawnFailed(refused)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if self._hf_home:
            env["HF_HOME"] = self._hf_home
        try:
            self._proc = subprocess.Popen(
                [self._python, str(self._worker_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0, env=env,
            )
        except OSError as exc:
            raise _SpawnFailed(f"{type(exc).__name__}: the interpreter cannot be started") from exc

    def _worker_for_load(self) -> subprocess.Popen | None:
        """A fresh worker to load into, or None. Called with ``_lock`` held.

        Never leaves a worker behind a shutdown that ran while it was
        starting — that shutdown found nothing to stop.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._ready = False
            self._spawn()
            proc = self._proc
            if self._shutdown.is_set():
                self._kill()
                return None
        if proc is None or proc.stdin is None or proc.stdout is None:
            return None
        return proc

    def _request(self, req: dict, *, deadline: float) -> dict | None:
        """The load request (it spawns). Blocks on the lock: the warm-up owns it."""
        if self._shutdown.is_set():
            return None
        with self._lock:
            proc = None
            try:
                proc = self._worker_for_load()
                if proc is None:
                    return None
                return self._exchange(proc, req, deadline)
            except (BrokenPipeError, EOFError, OSError, ValueError) as exc:
                logger.warning("Laya sufficiency judge transport failed: %s",
                               type(exc).__name__)
                self._kill()
                return None
            finally:
                self._after_use(proc)

    def _exchange(self, proc: subprocess.Popen, req: dict, deadline: float) -> dict | None:
        """Send ``req`` and read its own reply. On the deadline the request is
        recorded as stale — its reply is discarded when it arrives."""
        request_id = next(self._ids)
        write_request(proc.stdin, {**req, "id": request_id})
        sent_at = time.monotonic()
        reply = self._read_reply(proc, request_id, deadline)
        if reply is None and req.get("cmd") != "load":
            self._stale = (request_id, sent_at)
        return reply

    def _read_reply(self, proc: subprocess.Popen, request_id: int,
                    deadline: float) -> dict | None:
        reader = self._reader_for(proc)
        while True:
            line = reader.readline(deadline)
            if line is None:
                return None
            reply = json.loads(line)
            if not isinstance(reply, dict):
                raise ValueError("worker reply is not an object")
            if reply.get("id", request_id) == request_id:
                return reply
            # A late reply to a request a recall already gave up on.

    def _drain_stale(self, proc: subprocess.Popen, deadline: float) -> bool:
        """Read and discard the late reply to an abandoned request, if one is due.

        True when nothing is outstanding. False when the worker is still busy
        with it (the caller says "busy"), or has been stuck so long that it was
        replaced (``_ready`` is then False).
        """
        if self._stale is None:
            return True
        stale_id, sent_at = self._stale
        if time.monotonic() - sent_at > _WEDGED_S:
            logger.warning("Laya sufficiency judge stopped answering; replacing it")
            self._worker_lost()
            return False
        reader = self._reader_for(proc)
        wait_until = min(deadline - self._min_ask_s, sent_at + _WEDGED_S)
        while True:
            line = reader.readline(wait_until)
            if line is None:
                return False
            reply = json.loads(line)
            if isinstance(reply, dict) and reply.get("id", stale_id) == stale_id:
                self._stale = None
                return True

    def _reader_for(self, proc: subprocess.Popen) -> LineReader:
        current = self._reader
        if current is None or current[0] is not proc:
            current = (proc, LineReader(proc.stdout.fileno()))
            self._reader = current
        return current[1]

    def _worker_lost(self) -> None:
        """The worker died, broke the protocol or stopped answering."""
        self._kill()
        self._note_failure()

    def _kill(self, *, graceful: bool = False) -> None:
        """Stop the worker. Not graceful by default, and on purpose.

        Every caller but an orderly shutdown is reacting to a worker that did
        not answer. A stuck worker cannot read "quit", so asking it politely
        and waiting made the recall that hit the timeout pay that wait too.
        """
        proc, self._proc = self._proc, None
        self._ready = False
        self._stale = None
        self._reader = None
        _stop_process(proc, graceful=graceful)

    def shutdown(self) -> None:
        """Stop the worker without waiting on the lock.

        The warm-up holds the lock for up to the load timeout, so taking it here
        could stall a daemon shutdown for minutes. Killing the process directly
        unblocks any reader, and the request path cleans up after itself.
        """
        self._shutdown.set()
        proc, self._proc = self._proc, None
        self._ready = False
        _stop_process(proc, graceful=False, wait_s=2, close=False)
        # The pipes close here only when no request is using them; otherwise
        # the request closes them on its way out (``_after_use``).
        if proc is not None and self._lock.acquire(blocking=False):
            try:
                _close_pipes(proc)
            finally:
                self._lock.release()
        # Only after the worker is gone: the next holder must never overlap it.
        with self._state_lock:
            slot, self._slot = self._slot, None
        _release_slot(slot)


class _SpawnFailed(RuntimeError):
    """The configured interpreter could not be started at all."""
