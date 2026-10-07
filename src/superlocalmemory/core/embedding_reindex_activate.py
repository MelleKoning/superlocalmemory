# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The swap: tables and query model change in one step for every request.

``ProfileRuntime.reconfigure`` drains the requests already admitted and holds
new ones (they wait; nothing is refused). Inside that window the storage swap
commits in one transaction, config.json is rewritten, and a new engine built on
the new model is published. Requests resume only after that, so none can pair
the new model with the old vectors or the old model with the new ones.

If publishing the engine fails after the commit, the swap is reversed at once
(the previous space is complete at that instant) and the job fails. If the
process dies between the commit and the config write, the next start rolls
config.json forward from the database, which is authoritative.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from superlocalmemory.core import embedding_reindex_secrets as secrets
from superlocalmemory.core import embedding_reindex_steps as steps
from superlocalmemory.storage import embedding_spaces as sp
from superlocalmemory.storage.embedding_reindex_jobs import get_job, update_job
from superlocalmemory.storage.embedding_space_swap import NotCaughtUp, activate, reverse

logger = logging.getLogger(__name__)


class ActivationFailed(RuntimeError):
    """The swap was reversed; the job is already marked failed."""


def _load_config(runner: Any) -> Any:
    from superlocalmemory.core.config import SLMConfig

    return SLMConfig.load(runner.data_root / "config.json")


def write_live_config(base_dir: Path, emb_cfg: Any, signature: str) -> None:
    """config.json: embedding := emb_cfg and embedding_signature := signature."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.embedding_migrator import _write_stored_signature

    config = SLMConfig.load(Path(base_dir) / "config.json")
    config.embedding = emb_cfg
    config.save()
    _write_stored_signature(Path(base_dir), signature)


def _publish(runner: Any, job: dict, embedder: Any, target: Any, snapshot: Any) -> None:
    from superlocalmemory.core.embedding_live import offer_embedder
    from superlocalmemory.storage.embedding_migrator import _write_stored_signature

    new_config = _load_config(runner)
    previous_key = new_config.embedding.api_key
    new_config.embedding = target
    if snapshot is not None:
        new_config.active_profile = snapshot.profile_id
    callback = getattr(runner.app_state, "reconfigure_engine", None)
    if callable(callback):
        offer_embedder(new_config.db_path, job["to_signature"], embedder)
        callback(new_config, mode_change=False)  # builds, publishes, saves config.json
    else:
        new_config.save()
    _write_stored_signature(runner.data_root, job["to_signature"])
    secrets.put(runner.data_root, secrets.PREVIOUS, previous_key)
    secrets.drop(runner.data_root, secrets.TARGET)
    try:  # out-of-process fallbacks still hold the old model
        from superlocalmemory.core.worker_pool import WorkerPool

        WorkerPool.shared().shutdown()
    except Exception as exc:
        logger.debug("worker pool recycle after switch skipped: %s", exc)


def _reverse(runner: Any, conn: Any, job: dict, error: str) -> None:
    from superlocalmemory.core.embedding_live import take_preset_embedder

    source = sp.config_from_public(job["from_config"])
    with steps.write_txn(conn, runner.db_path):
        reverse(conn, job, model_name=source.model_name, dimension=source.dimension,
                live_cfg=json.loads(job["from_config"]))
        update_job(conn, job["job_id"], state="failed", error=error,
                   finished_at=time.time(), activated_at=None,
                   stats=json.dumps({"clear": {"cursor": 0}}))
    try:
        cfg = _load_config(runner)
        take_preset_embedder(cfg)  # never leave the handed-over model behind
        restored = sp.config_from_public(job["from_config"], cfg.embedding.api_key
                                         if sp.signature_of(cfg.embedding) == job["from_signature"]
                                         else secrets.get(runner.data_root, secrets.PREVIOUS))
        write_live_config(runner.data_root, restored, job["from_signature"])
    except Exception as exc:
        logger.error("config.json could not be restored after a reversed switch: %s", exc)


def activate_job(runner: Any, job: dict, embedder: Any, target: Any) -> dict:
    """One activation attempt. Returns the job as it stands afterwards."""
    from superlocalmemory.server.profile_runtime import (
        TransitionDrainTimeout,
        get_profile_runtime,
    )

    conn = sp.connect(runner.db_path)
    outcome: dict[str, Any] = {}
    try:
        def _commit(snapshot: Any = None) -> None:
            started = time.perf_counter()
            with steps.write_txn(conn, runner.db_path):
                outcome["swap"] = activate(
                    conn, job, model_name=target.model_name, dimension=target.dimension,
                    live_cfg=sp.public_config(target), prev_cfg=json.loads(job["from_config"]))
            outcome["swap_lock_ms"] = round((time.perf_counter() - started) * 1000, 1)
            published = time.perf_counter()
            try:
                _publish(runner, job, embedder, target, snapshot)
            except BaseException as exc:
                logger.exception("the new embedding model could not be published")
                _reverse(runner, conn, job, f"the new model could not be put in service: {exc}")
                raise ActivationFailed(str(exc)) from exc
            outcome["publish_ms"] = round((time.perf_counter() - published) * 1000, 1)

        held = time.perf_counter()
        try:
            if runner.app_state is not None:
                get_profile_runtime(runner.app_state).reconfigure(_commit)
            else:
                _commit(None)
        except NotCaughtUp as exc:
            logger.info("re-index job %s caught new writes at the swap (%s); catching up",
                        job["job_id"], exc)
            with steps.write_txn(conn, runner.db_path):
                update_job(conn, job["job_id"], state="catching_up")
            return get_job(conn, job["job_id"])
        except TransitionDrainTimeout as exc:
            logger.info("re-index job %s: requests did not drain (%s); retrying", job["job_id"], exc)
            return get_job(conn, job["job_id"])
        outcome["window_ms"] = round((time.perf_counter() - held) * 1000, 1)
        _record(runner, conn, job, outcome)
        _settle(runner, conn, job, target)
        logger.info("embedding re-index job %s activated: %s", job["job_id"], outcome)
        return get_job(conn, job["job_id"])
    finally:
        conn.close()


def _record(runner: Any, conn: Any, job: dict, outcome: dict) -> None:
    current = get_job(conn, job["job_id"])
    stats = json.loads(current["stats"]) if current.get("stats") else {}
    stats["activation"] = outcome
    stats["clear"] = {"cursor": 0}  # the twins now hold the replaced space's vectors
    with steps.write_txn(conn, runner.db_path):
        update_job(conn, job["job_id"], stats=json.dumps(stats))
        if job["kind"] == "rollback":  # the switch it undid
            previous = conn.execute(
                f"SELECT job_id FROM {sp.JOBS} WHERE kind = 'switch' AND state = 'activated' "
                "AND to_signature = ? AND job_id < ? ORDER BY job_id DESC LIMIT 1",
                (job["from_signature"], job["job_id"])).fetchone()
            if previous is not None:
                update_job(conn, int(previous[0]), state="rolled_back")


def _settle(runner: Any, conn: Any, job: dict, target: Any) -> None:
    engine = getattr(runner.app_state, "engine", None) if runner.app_state else None
    embedder = getattr(engine, "embedder", None)
    try:
        time.sleep(1.0)
        fixed = steps.settle(runner.db_path, embedder, target.model_name, target.dimension)
    except Exception as exc:
        logger.warning("post-switch settle pass failed: %s", exc)
        fixed = -1
    current = get_job(conn, job["job_id"])
    stats = json.loads(current["stats"]) if current.get("stats") else {}
    stats["settled"] = fixed
    with steps.write_txn(conn, runner.db_path):
        update_job(conn, job["job_id"], stats=json.dumps(stats))


__all__ = ["ActivationFailed", "activate_job", "write_live_config"]
