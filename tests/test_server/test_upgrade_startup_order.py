# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Start-up order for upgrades: a requested restore runs before migrations (their
clean-up could delete the restore point), the one-time repair after them and
before the engine opens the store, and the re-import once the engine is up."""

from __future__ import annotations

import inspect

from superlocalmemory.server import unified_daemon


def test_restore_then_migrate_then_repair_then_engine_then_reimport() -> None:
    source = inspect.getsource(unified_daemon.lifespan)
    order = [source.index(marker) for marker in (
        "perform_pending_restore(",
        "apply_all(",
        "run_store_repair_once(",
        "MemoryEngine(config)",
        "run_pending_reimport(",
    )]
    assert order == sorted(order), order


def test_restore_and_facet_routes_answer() -> None:
    # FastAPI keeps included routers as lazy entries, so ask the app itself
    # rather than listing app.routes.
    from fastapi.testclient import TestClient

    client = TestClient(unified_daemon.create_app())
    for path in ("/api/upgrade/restore-points", "/api/v3/facets"):
        assert client.get(path).status_code != 404, path
    assert client.get("/api/upgrade/not-a-route").status_code == 404
