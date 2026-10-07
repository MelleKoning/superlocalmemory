# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The daemon's two start-up calls and its health line for the re-index.

``prepare_daemon_config`` runs BEFORE the daemon builds its engine: it records
the live space on a store's first 4.1.22 start, and rolls config.json forward
when the process died between a swap's commit and its config write.
``start_for_daemon`` runs AFTER: it starts the runner (which resumes a job at
its cursor) and turns a model named outside the job API into a job.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from superlocalmemory.core import embedding_reindex as er
from superlocalmemory.core import embedding_reindex_secrets as secrets
from superlocalmemory.storage import embedding_spaces as sp
from superlocalmemory.storage.embedding_reindex_jobs import active_job, latest_job, public_view

logger = logging.getLogger(__name__)


def _record_first_start(conn: Any, config: Any) -> None:
    from superlocalmemory.core.embedding_live import _infer_live, live_signature

    desired = config.embedding
    live_sig, _cfg = live_signature(config, conn)
    desired_sig = sp.signature_of(desired)
    if live_sig is None or sp.same_space(desired_sig, live_sig):
        live_sig, live = desired_sig, desired
    else:
        live = _infer_live(desired, live_sig)
    with _txn(conn, config):
        sp.write_space(conn, live_sig, sp.public_config(live))


def _txn(conn: Any, config: Any):
    from superlocalmemory.core.embedding_reindex_steps import write_txn

    return write_txn(conn, config.db_path)


def _roll_forward(conn: Any, config: Any, row: dict) -> bool:
    """A committed swap whose config write never happened: finish it now."""
    from superlocalmemory.core.embedding_reindex_activate import write_live_config

    job = latest_job(conn)
    desired_sig = sp.signature_of(config.embedding)
    if (job is None or job["state"] != "activated"
            or job["to_signature"] != row["live_signature"]
            or sp.same_space(desired_sig, row["live_signature"])
            or not sp.same_space(desired_sig, job["from_signature"])):
        return False
    role = secrets.TARGET if job["kind"] == "switch" else secrets.PREVIOUS
    key = secrets.get(config.base_dir, role)
    old_key = config.embedding.api_key
    live = sp.config_from_public(row["live_config"], key)
    write_live_config(config.base_dir, live, row["live_signature"])
    config.embedding = live
    secrets.put(config.base_dir, secrets.PREVIOUS, old_key)
    secrets.drop(config.base_dir, secrets.TARGET)
    logger.warning("finished an embedding switch interrupted after its swap: %s is live",
                   row["live_signature"])
    return True


def prepare_daemon_config(config: Any) -> Any:
    """Before the engine: never fails start-up; returns the config to build on."""
    db_path = Path(config.db_path)
    if not db_path.exists():
        return config
    try:
        conn = sp.connect(db_path)
    except Exception as exc:  # no sqlite-vec here: nothing to switch either
        logger.debug("embedding space check skipped: %s", exc)
        return config
    try:
        if not sp.table_exists(conn, "atomic_facts"):
            return config
        sp.ensure_control_tables(conn)
        row = sp.read_space(conn)
        if row is None:
            _record_first_start(conn, config)
        else:
            _roll_forward(conn, config, row)
    except Exception as exc:
        logger.error("embedding space check at start failed (serving anyway): %s", exc)
    finally:
        conn.close()
    return config


def start_for_daemon(app_state: Any, config: Any) -> er.ReindexRunner | None:
    """After the engine: start the runner and queue any model named meanwhile."""
    from superlocalmemory.core.embedding_live import PENDING_ATTR

    try:
        runner = er.ReindexRunner(db_path=config.db_path, data_root=config.base_dir,
                                  app_state=app_state)
        engine_config = getattr(getattr(app_state, "engine", None), "_config", None)
        conn = sp.connect(config.db_path)
        try:  # a fresh store only exists once the engine has created it
            sp.ensure_control_tables(conn)
            if sp.read_space(conn) is None:
                _record_first_start(conn, engine_config or config)
        finally:
            conn.close()
        er._RUNNER = runner
        app_state.embedding_reindex = runner
        for candidate in (engine_config, config):
            if candidate is not None and getattr(candidate, PENDING_ATTR, None) is not None:
                runner.adopt_pending(candidate)
                # config.json now names the live model again; the job holds the target.
                candidate.save()
                break
        runner.start()
        return runner
    except Exception as exc:
        logger.error("embedding re-index runner did not start: %s", exc)
        return None


def health_payload(app_state: Any) -> dict | None:
    """``embedding_reindex`` on /health: state, done, total, eta (None: no job ever)."""
    runner = getattr(app_state, "embedding_reindex", None)
    if runner is None:
        return None
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{runner.db_path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            view = public_view(active_job(conn) or latest_job(conn))
        finally:
            conn.close()
    except Exception as exc:
        return {"state": "unknown", "error": str(exc)}
    if view is None:
        return None
    payload = {k: view[k] for k in ("job_id", "kind", "state", "done", "total", "eta_seconds",
                                    "from", "to", "error")}
    payload["notice"] = getattr(runner, "notice", None)
    return payload


__all__ = ["health_payload", "prepare_daemon_config", "start_for_daemon"]
